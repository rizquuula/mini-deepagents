"""The agent service: one chat turn in which the model may call tools.

The service runs the model and the tools in a loop, inside a hard token
budget. It saves only the user message and the final reply, never the tool
steps. It is pure domain code. It imports no framework and logs nothing.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncGenerator, Callable, Sequence
from contextlib import aclosing
from typing import Final

from mini_deepagents.agent.domain import AgentEvent, ToolFinished, ToolStarted
from mini_deepagents.agent.tools import ToolSet
from mini_deepagents.chat.context import BYTES_PER_TOKEN
from mini_deepagents.chat.domain import (
    Message,
    ReplyDone,
    Session,
    StreamEnd,
    TextDelta,
    ToolCall,
    ToolCallRequested,
    Usage,
)
from mini_deepagents.chat.ports import ChatModel, ChatRepository, TokenEstimator
from mini_deepagents.chat.service import ChatService, PreparedTurn
from mini_deepagents.errors import ContextOverflowError, ModelError, ToolConfigError

__all__ = ["TRUNCATION_MARKER", "AgentService"]

TRUNCATION_MARKER: Final = " [truncated]"


class AgentService:
    """Run chat turns in which the model may call tools in a loop.

    The service reserves ``tool_cost`` tokens of ``input_budget`` for the
    tool schemas: ``tool_overhead_tokens`` plus the estimate of each spec.
    An empty tool set costs 0. The history fits the rest of the budget.
    """

    def __init__(
        self,
        repository: ChatRepository,
        model: ChatModel,
        estimator: TokenEstimator,
        tools: ToolSet,
        *,
        input_budget: int,
        max_output_tokens: int,
        tool_overhead_tokens: int = 0,
        max_steps: int = 4,
        max_tool_result_tokens: int = 512,
        session_key_prefix: str = "mini-deepagents-",
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        """Wrap a ``ChatService`` with the budget left after the tool schemas.

        Raises:
            ToolConfigError: The tool schemas use the whole input budget,
                ``max_steps`` is below 1, or ``max_tool_result_tokens``
                leaves no room for the truncation marker.
        """
        tool_cost = tool_overhead_tokens + tools.prompt_tokens(estimator) if tools else 0
        if tools and tool_cost >= input_budget:
            raise ToolConfigError(
                f"tools need {tool_cost} tokens; input budget is {input_budget}"
            )
        if max_steps < 1:
            raise ToolConfigError(f"max_steps must be at least 1, got {max_steps}")
        result_limit = max_tool_result_tokens * BYTES_PER_TOKEN
        if result_limit < len(TRUNCATION_MARKER.encode("utf-8")):
            raise ToolConfigError(
                f"max_tool_result_tokens {max_tool_result_tokens} is too small"
            )
        self._chat = ChatService(
            repository,
            model,
            estimator,
            input_budget=input_budget - tool_cost,
            max_output_tokens=max_output_tokens,
            session_key_prefix=session_key_prefix,
        )
        self._repository = repository
        self._model = model
        self._estimator = estimator
        self._tools = tools
        self._specs = tools.specs()
        self._tool_cost = tool_cost
        self._budget = input_budget - tool_cost
        self._max_output_tokens = max_output_tokens
        self._max_steps = max_steps
        self._result_limit = result_limit
        self._session_key_prefix = session_key_prefix
        self._clock = clock

    async def create_session(self) -> Session:
        """Create a new empty session and return it."""
        return await self._chat.create_session()

    async def list_sessions(self) -> list[Session]:
        """Return all sessions, newest first."""
        return await self._chat.list_sessions()

    async def get_messages(self, session_id: int) -> list[Message]:
        """Return the messages of a session, oldest first.

        Raises:
            SessionNotFoundError: No session has ``session_id``.
        """
        return await self._chat.get_messages(session_id)

    async def prepare(self, session_id: int, content: str) -> PreparedTurn:
        """Fit the history plus the new user message into the reduced budget.

        The call writes nothing.

        Raises:
            SessionNotFoundError: No session has ``session_id``.
            ContextOverflowError: The new message alone exceeds the budget.
        """
        return await self._chat.prepare(session_id, content)

    async def stream_reply(self, turn: PreparedTurn) -> AsyncGenerator[AgentEvent, None]:
        """Run the model and the tools in a loop, then save the final reply.

        Each step calls the model once. A step without tool calls ends the
        turn: the service saves the exchange and yields ``ReplyDone``. The
        last step sends no tools. An error, an early close, or a cancellation
        saves nothing.

        Raises:
            ModelError: The model call failed, or the model requested a tool
                at the last step.
            ContextOverflowError: The tool steps outgrow the budget.
        """
        session_key = f"{self._session_key_prefix}{turn.session_id}"
        window = list(turn.window)
        peak = 0
        for step in range(1, self._max_steps + 1):
            window_tokens = self._check_fits(window, step)
            tools = self._specs if step < self._max_steps else ()
            peak = max(peak, window_tokens + (self._tool_cost if tools else 0))

            parts: list[str] = []
            calls: list[ToolCall] = []
            usage: Usage | None = None
            events = self._model.stream(
                tuple(window),
                max_tokens=self._max_output_tokens,
                session_key=session_key,
                tools=tools,
            )
            async with aclosing(events):
                async for event in events:
                    if isinstance(event, TextDelta):
                        parts.append(event.text)
                        yield event
                    elif isinstance(event, ToolCallRequested):
                        calls.append(event.call)
                    elif isinstance(event, StreamEnd):
                        usage = event.usage
            if usage is None:
                raise ModelError("model stream ended without StreamEnd")

            text = "".join(parts)
            if not calls:
                await asyncio.to_thread(
                    self._repository.add_exchange, turn.session_id, turn.content, text
                )
                yield ReplyDone(
                    reply_chars=len(text),
                    usage=usage,
                    steps=step,
                    peak_estimated_tokens=peak,
                )
                return
            if not tools:
                raise ModelError("model requested a tool at the last step")

            window.append(Message("assistant", text, tool_calls=tuple(calls)))
            for call in calls:
                yield ToolStarted(call)
                started = self._clock()
                ok, result = await self._run_tool(call)
                result = self._cut(result)
                yield ToolFinished(
                    call_id=call.id,
                    name=call.name,
                    ok=ok,
                    result_chars=len(result),
                    duration_ms=round((self._clock() - started) * 1000),
                )
                window.append(Message("tool", result, tool_call_id=call.id))

    def _check_fits(self, window: Sequence[Message], step: int) -> int:
        estimated = sum(self._estimator.estimate(message) for message in window)
        if step > 1 and estimated > self._budget:
            raise ContextOverflowError(
                f"tool steps need {estimated} tokens; budget is {self._budget}",
                estimated_tokens=estimated,
                budget_tokens=self._budget,
            )
        return estimated

    async def _run_tool(self, call: ToolCall) -> tuple[bool, str]:
        tool = self._tools.get(call.name)
        if tool is None:
            return False, f'error: unknown tool "{call.name}"'
        arguments = _parse_arguments(call.arguments)
        if arguments is None:
            return False, "error: arguments are not a JSON object"
        try:
            return True, await tool.run(arguments)
        except Exception as exc:  # noqa: BLE001 - the contract allows any exception
            return False, f"error: {type(exc).__name__}: {exc}"

    def _cut(self, result: str) -> str:
        encoded = result.encode("utf-8")
        if len(encoded) <= self._result_limit:
            return result
        room = self._result_limit - len(TRUNCATION_MARKER.encode("utf-8"))
        return encoded[:room].decode("utf-8", errors="ignore") + TRUNCATION_MARKER


def _parse_arguments(raw: str) -> dict[str, object] | None:
    """Return the JSON object in ``raw``, or None. Blank text means no arguments."""
    if not raw.strip():
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None

"""Fakes for the chat and agent ports, to test code without a real model, store, or tool."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Mapping, Sequence
from datetime import datetime, timedelta, timezone

from mini_deepagents.chat import (
    Message,
    ModelEvent,
    Session,
    StreamEnd,
    TextDelta,
    ToolCall,
    ToolCallRequested,
    ToolSpec,
    Usage,
)
from mini_deepagents.errors import ModelError, SessionNotFoundError

_EPOCH = datetime(2026, 1, 1, tzinfo=timezone.utc)


class FakeChatModel:
    """A ``ChatModel`` that streams fixed events and records each call.

    Without ``steps``, every call streams ``deltas``. With ``steps``, call N
    plays ``steps[N]``: a ``str`` yields ``TextDelta`` and a ``ToolCall``
    yields ``ToolCallRequested``. ``error`` and ``fail_after`` apply to every
    call. ``calls`` records the messages, max tokens, and session key of each
    call. ``tools_per_call`` records the tools of each call.
    """

    def __init__(
        self,
        deltas: Sequence[str] = ("Hello", " world"),
        *,
        usage: Usage = Usage(10, 2, 12),
        error: Exception | None = None,
        fail_after: int = 0,
        steps: Sequence[Sequence[str | ToolCall]] | None = None,
    ) -> None:
        self.deltas = tuple(deltas)
        self.usage = usage
        self.error = error
        self.fail_after = fail_after
        self.steps = None if steps is None else tuple(tuple(step) for step in steps)
        self.calls: list[tuple[list[Message], int, str]] = []
        self.tools_per_call: list[tuple[ToolSpec, ...]] = []
        self.closed_early = False
        self.aclose_calls = 0

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int,
        session_key: str,
        tools: Sequence[ToolSpec] = (),
    ) -> AsyncGenerator[ModelEvent, None]:
        items = self._items_for_call(len(self.calls))
        self.calls.append((list(messages), max_tokens, session_key))
        self.tools_per_call.append(tuple(tools))
        finished = False
        try:
            for index, item in enumerate(items):
                if self.error is not None and index == self.fail_after:
                    raise self.error
                yield TextDelta(item) if isinstance(item, str) else ToolCallRequested(item)
            if self.error is not None:
                raise self.error
            if not items:
                raise ModelError("empty reply")
            finished = True
            yield StreamEnd(self.usage)
        except GeneratorExit:
            if not finished:
                self.closed_early = True
            raise

    async def aclose(self) -> None:
        self.aclose_calls += 1

    def _items_for_call(self, index: int) -> tuple[str | ToolCall, ...]:
        if self.steps is None:
            return self.deltas
        if index >= len(self.steps):
            raise ModelError(f"no scripted step for call {index + 1}")
        return self.steps[index]


class FakeTool:
    """A ``Tool`` that returns a fixed result or raises, and records each call."""

    def __init__(
        self,
        name: str,
        result: str = "ok",
        *,
        error: Exception | None = None,
        parameters: Mapping[str, object] | None = None,
    ) -> None:
        self.spec = ToolSpec(
            name=name,
            description=f"The fake tool {name}.",
            parameters=parameters or {"type": "object", "properties": {}},
        )
        self.result = result
        self.error = error
        self.calls: list[Mapping[str, object]] = []

    async def run(self, arguments: Mapping[str, object]) -> str:
        self.calls.append(arguments)
        if self.error is not None:
            raise self.error
        return self.result


class InMemoryChatRepository:
    """A ``ChatRepository`` that keeps sessions in a dict."""

    def __init__(self) -> None:
        self._sessions: dict[int, Session] = {}
        self._messages: dict[int, list[Message]] = {}

    def create_session(self) -> Session:
        session_id = len(self._sessions) + 1
        created_at = (_EPOCH + timedelta(seconds=session_id)).isoformat()
        session = Session(id=session_id, created_at=created_at)
        self._sessions[session_id] = session
        self._messages[session_id] = []
        return session

    def list_sessions(self) -> list[Session]:
        return sorted(self._sessions.values(), key=lambda s: s.id, reverse=True)

    def get_messages(self, session_id: int) -> list[Message]:
        return list(self._messages_of(session_id))

    def add_exchange(
        self, session_id: int, user_content: str, assistant_content: str
    ) -> None:
        self._messages_of(session_id).extend(
            [
                Message(role="user", content=user_content),
                Message(role="assistant", content=assistant_content),
            ]
        )

    def _messages_of(self, session_id: int) -> list[Message]:
        try:
            return self._messages[session_id]
        except KeyError:
            raise SessionNotFoundError(f"session {session_id} not found") from None

from __future__ import annotations

import asyncio
import itertools
from collections.abc import Sequence

from mini_deepagents import (
    AgentEvent,
    AgentService,
    ByteTokenEstimator,
    Message,
    ReplyDone,
    TextDelta,
    ToolCall,
    ToolFinished,
    ToolSet,
    ToolStarted,
)
from mini_deepagents.testing import FakeChatModel, FakeTool, InMemoryChatRepository

ESTIMATOR = ByteTokenEstimator()
MAX_OUTPUT_TOKENS = 64
CLOCK_STEP_MS = 500
TIME_CALL = ToolCall(id="call_1", name="clock", arguments='{"tz": "UTC"}')


def make_service(
    model: FakeChatModel,
    tools: Sequence[FakeTool] = (),
    *,
    repository: InMemoryChatRepository | None = None,
    input_budget: int = 1000,
    **options: int,
) -> tuple[AgentService, InMemoryChatRepository]:
    repository = repository or InMemoryChatRepository()
    service = AgentService(
        repository,
        model,
        ESTIMATOR,
        ToolSet(tools),
        input_budget=input_budget,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        clock=itertools.count(0.0, CLOCK_STEP_MS / 1000).__next__,
        **options,
    )
    return service, repository


def run_turn(
    service: AgentService, repository: InMemoryChatRepository, content: str = "time?"
) -> list[AgentEvent]:
    session = repository.create_session()

    async def run() -> list[AgentEvent]:
        turn = await service.prepare(session.id, content)
        return [event async for event in service.stream_reply(turn)]

    return asyncio.run(run())


def one_tool_turn() -> tuple[FakeChatModel, FakeTool]:
    model = FakeChatModel(steps=[[TIME_CALL], ["It is", " noon"]])
    return model, FakeTool("clock", "12:00")


def test_one_tool_turn_yields_tool_events_then_text_then_done() -> None:
    model, tool = one_tool_turn()
    service, repository = make_service(model, [tool])

    events = run_turn(service, repository)

    assert events[:-1] == [
        ToolStarted(TIME_CALL),
        ToolFinished("call_1", "clock", ok=True, result_chars=5, duration_ms=CLOCK_STEP_MS),
        TextDelta("It is"),
        TextDelta(" noon"),
    ]
    done = events[-1]
    assert isinstance(done, ReplyDone)
    assert (done.reply_chars, done.steps) == (len("It is noon"), 2)
    assert tool.calls == [{"tz": "UTC"}]


def test_one_tool_turn_saves_only_the_user_message_and_the_final_reply() -> None:
    model, tool = one_tool_turn()
    service, repository = make_service(model, [tool])

    run_turn(service, repository, "time?")

    assert repository.get_messages(1) == [
        Message("user", "time?"),
        Message("assistant", "It is noon"),
    ]


def test_second_model_call_receives_the_tool_call_and_its_result() -> None:
    model, tool = one_tool_turn()
    service, repository = make_service(model, [tool])

    run_turn(service, repository, "time?")

    second_window = model.calls[1][0]
    assert second_window == [
        Message("user", "time?"),
        Message("assistant", "", tool_calls=(TIME_CALL,)),
        Message("tool", "12:00", tool_call_id="call_1"),
    ]

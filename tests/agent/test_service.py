from __future__ import annotations

import asyncio
import itertools
from collections.abc import Sequence
from dataclasses import replace

import pytest

from mini_deepagents import (
    AgentEvent,
    AgentService,
    ByteTokenEstimator,
    ChatService,
    ContextOverflowError,
    Message,
    ModelError,
    ReplyDone,
    TextDelta,
    ToolCall,
    ToolConfigError,
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


def test_last_step_sends_no_tools() -> None:
    model = FakeChatModel(steps=[[TIME_CALL], ["done"]])
    tool = FakeTool("clock")
    service, repository = make_service(model, [tool], max_steps=2)

    run_turn(service, repository)

    assert model.tools_per_call == [(tool.spec,), ()]


def test_a_tool_call_at_the_last_step_raises_model_error_and_saves_nothing() -> None:
    model = FakeChatModel(steps=[[TIME_CALL], [TIME_CALL]])
    service, repository = make_service(model, [FakeTool("clock")], max_steps=2)

    with pytest.raises(ModelError):
        run_turn(service, repository)
    assert repository.get_messages(1) == []


@pytest.mark.parametrize(
    ("call", "tool", "expected"),
    [
        (
            ToolCall("call_1", "missing", "{}"),
            FakeTool("clock"),
            'error: unknown tool "missing"',
        ),
        (
            ToolCall("call_1", "clock", "[1, 2]"),
            FakeTool("clock"),
            "error: arguments are not a JSON object",
        ),
        (
            ToolCall("call_1", "clock", "{not json"),
            FakeTool("clock"),
            "error: arguments are not a JSON object",
        ),
        (
            ToolCall("call_1", "clock", "{}"),
            FakeTool("clock", error=RuntimeError("clock is down")),
            "error: RuntimeError: clock is down",
        ),
    ],
)
def test_a_failed_call_gives_the_model_an_error_result(
    call: ToolCall, tool: FakeTool, expected: str
) -> None:
    model = FakeChatModel(steps=[[call], ["sorry"]])
    service, repository = make_service(model, [tool])

    events = run_turn(service, repository)

    finished = next(event for event in events if isinstance(event, ToolFinished))
    assert not finished.ok
    assert model.calls[1][0][-1] == Message("tool", expected, tool_call_id="call_1")


@pytest.mark.parametrize("result", ["x" * 100, "é" * 100])
def test_a_long_result_is_cut_to_the_limit_with_a_marker(result: str) -> None:
    model = FakeChatModel(steps=[[TIME_CALL], ["done"]])
    service, repository = make_service(
        model, [FakeTool("clock", result)], max_tool_result_tokens=10
    )

    run_turn(service, repository)

    seen = model.calls[1][0][-1].content
    assert seen.endswith(" [truncated]")
    assert len(seen.encode("utf-8")) <= 10 * 3
    assert result.startswith(seen.removesuffix(" [truncated]"))


def test_a_window_that_outgrows_the_budget_raises_and_saves_nothing() -> None:
    tool = FakeTool("clock", "x" * 600)
    tool_cost = ESTIMATOR.estimate_tool(tool.spec)
    model = FakeChatModel(steps=[[TIME_CALL], ["done"]])
    service, repository = make_service(model, [tool], input_budget=tool_cost + 60)

    with pytest.raises(ContextOverflowError) as caught:
        run_turn(service, repository)

    assert caught.value.budget_tokens == 60
    assert caught.value.estimated_tokens is not None
    assert caught.value.estimated_tokens > 60
    assert len(model.calls) == 1
    assert repository.get_messages(1) == []


def test_an_empty_tool_set_sends_no_tools_and_behaves_like_chat_service() -> None:
    agent_model = FakeChatModel(("Hello", " world"))
    service, repository = make_service(agent_model, tool_overhead_tokens=207)
    chat_model = FakeChatModel(("Hello", " world"))
    chat_repository = InMemoryChatRepository()
    chat = ChatService(
        chat_repository,
        chat_model,
        ESTIMATOR,
        input_budget=1000,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )

    agent_events = run_turn(service, repository, "hi")
    chat_events = run_turn(chat, chat_repository, "hi")  # type: ignore[arg-type]

    done = agent_events[-1]
    assert isinstance(done, ReplyDone)
    assert [*agent_events[:-1], replace(done, peak_estimated_tokens=None)] == chat_events
    assert agent_model.calls == chat_model.calls
    assert agent_model.tools_per_call == [()]
    assert repository.get_messages(1) == chat_repository.get_messages(1)


@pytest.mark.parametrize(
    "options",
    [
        {"tool_overhead_tokens": 1000 - ESTIMATOR.estimate_tool(FakeTool("clock").spec)},
        {"max_steps": 0},
        {"max_tool_result_tokens": 3},
    ],
)
def test_an_impossible_setup_raises_tool_config_error(options: dict[str, int]) -> None:
    with pytest.raises(ToolConfigError):
        make_service(FakeChatModel(), [FakeTool("clock")], input_budget=1000, **options)


def test_peak_estimate_includes_the_tool_cost() -> None:
    model, tool = one_tool_turn()
    service, repository = make_service(model, [tool], tool_overhead_tokens=207)

    done = run_turn(service, repository)[-1]

    tool_cost = 207 + ESTIMATOR.estimate_tool(tool.spec)
    largest_window = sum(ESTIMATOR.estimate(message) for message in model.calls[1][0])
    assert isinstance(done, ReplyDone)
    assert done.peak_estimated_tokens == largest_window + tool_cost

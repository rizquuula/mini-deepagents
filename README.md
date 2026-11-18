# mini-deepagents

A minimal agent harness for models with an **8K-token context window**.
It has zero runtime dependencies: it imports only the Python standard library.

## Why this project exists

[Deep Agents](https://github.com/langchain-ai/deepagents) is a strong agent harness, but it is context hungry.
It assumes a model with a window of 100K tokens or more.
Its 8 default tool schemas cost about 2,480 tokens before the user writes one word.
On an 8K model, that is about 30% of the window.

Many small and local models have a window of 8K tokens or less.
This framework targets them.

The bet: a small context window can still give good results with a good harness and a good agent loop.
The model does not need 100K+ tokens.
It needs the right tokens.

## What is inside today

- **The chat turn:** `ChatService` prepares one turn, streams the model reply, and saves the exchange.
- **The agent turn:** `AgentService` runs the model and native tool calls in a loop, inside the same hard budget.
- **The context fitting rule:** `fit_messages` keeps the newest messages that fit a token budget.
  `ByteTokenEstimator` estimates each message as `ceil(utf8_bytes / 3) + 4` tokens, on the safe side.
  It also counts the bytes of tool calls and estimates each tool schema.
- **Four ports:** `ChatModel`, `ChatRepository`, `TokenEstimator`, and `Tool`.
- **The domain errors:** `MiniDeepAgentsError` and its subclasses `SessionNotFoundError`, `ModelError`, `ContextOverflowError`, and `ToolConfigError`.
- **Fakes:** `FakeChatModel`, `InMemoryChatRepository`, and `FakeTool` in `mini_deepagents.testing`.

## Install

```sh
uv add "mini-deepagents @ git+https://github.com/rizquuula/mini-deepagents@v0.2.0"
```

## Usage

This example runs one chat turn with the fakes.
Replace the fakes with your own adapters in a real app.

```python
import asyncio

from mini_deepagents import ByteTokenEstimator, ChatService, ReplyDone, TextDelta
from mini_deepagents.testing import FakeChatModel, InMemoryChatRepository


async def main() -> None:
    service = ChatService(
        InMemoryChatRepository(),
        FakeChatModel(("Hello", " world")),
        ByteTokenEstimator(),
        input_budget=7168,  # 8192 window minus 1024 output tokens
        max_output_tokens=1024,
    )
    session = await service.create_session()
    turn = await service.prepare(session.id, "Hi!")
    print(f"window: {len(turn.window)} message(s), {turn.estimated_tokens} tokens")

    async for event in service.stream_reply(turn):
        if isinstance(event, TextDelta):
            print(event.text, end="", flush=True)
        elif isinstance(event, ReplyDone):
            print(f"\ndone: {event.reply_chars} chars")

    print(await service.get_messages(session.id))


asyncio.run(main())
```

Output:

```
window: 1 message(s), 5 tokens
Hello world
done: 11 chars
[Message(role='user', content='Hi!', tool_calls=(), tool_call_id=None), Message(role='assistant', content='Hello world', tool_calls=(), tool_call_id=None)]
```

## The agent loop

`AgentService` runs one chat turn in which the model may call tools.
It uses the native tool calling of the provider.
Each step is one model call:

1. The service sends the window and the tool specs.
2. It forwards each `TextDelta` at once.
3. When the step has no tool call, the service saves the exchange and yields `ReplyDone`. The turn ends.
4. Otherwise, it appends the assistant message with its `tool_calls` to the window.
5. It runs each call in order and yields `ToolStarted` and `ToolFinished`.
6. It appends one `"tool"` message per result, then starts the next step.

The service saves only the user message and the final reply.
It never saves the tool steps.
An error, an early close, or a cancellation saves nothing.

### Limits

- **`max_steps`** (default 4): the maximum number of model calls per turn.
  The last step sends no tools, so the model must answer in text.
  A tool call at the last step raises `ModelError`.
- **`max_tool_result_tokens`** (default 512): the maximum size of one tool result.
  The service cuts a longer result to `max_tool_result_tokens * 3` bytes, including the marker ` [truncated]`.
- **The budget:** the service reserves `tool_cost` tokens for the tool schemas.
  `tool_cost` is `tool_overhead_tokens` plus the estimate of each spec. An empty tool set costs 0.
  The history and the tool steps share `input_budget - tool_cost`.
  A window that outgrows that budget raises `ContextOverflowError` before the next model call.
- A setup with `tool_cost >= input_budget` raises `ToolConfigError`.

A failed tool call does not stop the turn.
An unknown tool name, arguments that are not a JSON object, and a tool that raises each give an `error: ...` result.
The model sees that result in the next step.

`ReplyDone.steps` counts the model calls.
`ReplyDone.peak_estimated_tokens` is the largest prompt estimate of one step, with `tool_cost` when the step sent tools.
Log it next to the real `prompt_tokens` to check the estimate.

### Example

This example runs one agent turn with the fakes.
The fake model calls the tool in the first step and answers in the second step.

```python
import asyncio

from mini_deepagents import (
    AgentService,
    ByteTokenEstimator,
    ReplyDone,
    TextDelta,
    ToolCall,
    ToolFinished,
    ToolSet,
    ToolStarted,
)
from mini_deepagents.testing import FakeChatModel, FakeTool, InMemoryChatRepository


async def main() -> None:
    clock = FakeTool("get_time", "12:00 UTC")
    model = FakeChatModel(
        steps=[
            [ToolCall(id="call_1", name="get_time", arguments="{}")],
            ["It is ", "12:00 UTC."],
        ]
    )
    service = AgentService(
        InMemoryChatRepository(),
        model,
        ByteTokenEstimator(),
        ToolSet([clock]),
        input_budget=7168,  # 8192 window minus 1024 output tokens
        max_output_tokens=1024,
        tool_overhead_tokens=207,  # measured on deepseek-v4.1-flash
    )
    session = await service.create_session()
    turn = await service.prepare(session.id, "What time is it?")
    print(f"budget after tools: {turn.budget_tokens} tokens")

    async for event in service.stream_reply(turn):
        if isinstance(event, ToolStarted):
            print(f"tool start: {event.call.name}")
        elif isinstance(event, ToolFinished):
            print(f"tool done: ok={event.ok}, {event.result_chars} chars")
        elif isinstance(event, TextDelta):
            print(event.text, end="", flush=True)
        elif isinstance(event, ReplyDone):
            print(f"\ndone: {event.steps} steps, peak {event.peak_estimated_tokens} tokens")

    print(await service.get_messages(session.id))


asyncio.run(main())
```

Output:

```
budget after tools: 6911 tokens
tool start: get_time
tool done: ok=True, 9 chars
It is 12:00 UTC.
done: 2 steps, peak 286 tokens
[Message(role='user', content='What time is it?', tool_calls=(), tool_call_id=None), Message(role='assistant', content='It is 12:00 UTC.', tool_calls=(), tool_call_id=None)]
```

## Tool overhead

A provider adds a fixed number of prompt tokens when a request carries any tools.
This overhead depends on the provider, so the framework does not guess it.
Pass it to `AgentService` as `tool_overhead_tokens`.

Measure it once per provider and model:

1. Send one small request with no tools. Note the `prompt_tokens` that the provider reports.
2. Send the same request with one small tool. Note the `prompt_tokens` again.
3. Subtract the first count from the second count.
4. Subtract the tool's own cost, which is about 46 tokens for a schema of about 200 bytes.

The result is the overhead.
For deepseek-v4.1-flash on OpenCode Go, the measured counts are 44 tokens with no tools and 297 with one tool.
Each further small tool adds about 46 tokens.
So the overhead is about 207 tokens.
`ByteTokenEstimator` prices one such tool at about 67 tokens, which is on the safe side.

## Breaking changes in 0.2.0

- `ChatModel.stream()` gains the keyword argument `tools: Sequence[ToolSpec] = ()`.
  It may also yield `ToolCallRequested` events. Update each `ChatModel` adapter.
- `TokenEstimator` gains `estimate_tool(spec)`. Update each custom estimator.
- `Role` gains `"tool"`. `Message` gains `tool_calls` and `tool_call_id`, with defaults.
- `ReplyDone` gains `steps` and `peak_estimated_tokens`, with defaults.

## Write an adapter

An adapter connects the framework to a real model or a real store.

1. Write a class that implements the `ChatModel`, the `ChatRepository`, or the `Tool` protocol.
2. Read the contract in each docstring of `mini_deepagents/chat/ports.py` and `mini_deepagents/agent/ports.py`.
   Keep every rule of that contract.
3. Translate each vendor error into a domain error, such as `ModelError`.
4. Pass an instance of your class to `ChatService` or `AgentService`.

Run your adapter and the matching fake through the same tests.
Then both keep the same contract.

## Reference app

The demo app at https://github.com/rizquuula/mini-deepagents-demo uses this framework.
It holds the real adapters:

- an OpenCode model adapter
- a SQLite store
- a FastAPI server that streams replies over SSE
- a vanilla JavaScript web chat

## Development

Requirements: [uv](https://docs.astral.sh/uv/) and `make`.

| Command | What it does |
|---|---|
| `make` | Lists the targets. |
| `make test` | Runs the test suite with pytest. |

## License

MIT. See `LICENSE`.

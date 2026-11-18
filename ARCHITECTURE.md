# Architecture

mini-deepagents is a Python framework with a hexagonal design.
It holds ports and domain logic only.
Apps supply the adapters, such as a model client or a database store.

## Package layout

```
src/mini_deepagents/
  __init__.py          the public API: every name an app may import
  errors.py            MiniDeepAgentsError and its subclasses
  py.typed             marks the package as typed
  chat/                the chat context: the message protocol
    domain.py          Message, Session, Usage, ToolSpec, ToolCall, the model events, ReplyDone
    ports.py           ChatModel, ChatRepository, TokenEstimator (Protocols)
    context.py         ByteTokenEstimator, fit_messages (the token budget)
    service.py         ChatService: prepare one turn, stream it, save it
  agent/               the agent context: tools in a loop
    domain.py          ToolStarted, ToolFinished, AgentEvent
    ports.py           Tool (Protocol)
    tools.py           ToolSet: tools with unique names and their prompt cost
    service.py         AgentService: the tool-call loop around ChatService
  testing/
    fakes.py           FakeChatModel, InMemoryChatRepository, FakeTool
tests/
  chat/                tests of the chat context
  agent/               tests of the agent context
  test_architecture.py stdlib-only imports, no logging
  test_public_api.py   the exact export set
```

## Ports

Each port lives in the `ports.py` of its context, on the consumer side.
Each port docstring states the full contract.
In short:

| Port | Contract |
|---|---|
| `ChatModel.stream()` | Yield `TextDelta` and `ToolCallRequested` events, then exactly one `StreamEnd`, last. Raise `ModelError` on any failure. A stream with no text and no tool call is a failure. Send `tools` to the provider only when the sequence is not empty. Stop the upstream call when the consumer closes the stream early. |
| `ChatModel.aclose()` | Release the resources of the model. |
| `ChatRepository` | Create and list sessions. Return messages oldest first. `add_exchange()` writes the user message and the reply in one transaction. Raise `SessionNotFoundError` for an unknown session. |
| `TokenEstimator.estimate()` | Return the token cost of one message, with its tool calls. Never underestimate. |
| `TokenEstimator.estimate_tool()` | Return the token cost of one tool schema, without the provider overhead. Never underestimate. |
| `Tool` | Expose a `spec`. `run(arguments)` returns the result text and raises any exception on failure. |

## One chat turn

1. **Prepare:** `ChatService.prepare()` loads the history and appends the new user message.
2. **Fit:** `fit_messages()` keeps the newest messages that fit the input budget. The window starts with a user message. The call writes nothing.
3. **Stream:** `ChatService.stream_reply()` calls `ChatModel.stream()` and yields each `TextDelta`.
4. **Save on `StreamEnd`:** the service calls `ChatRepository.add_exchange()` only after the model stream completes.
5. **`ReplyDone`:** the service yields `ReplyDone` with the reply length and the token usage.

An error, an early close, or a cancellation saves nothing.

## One agent turn

`AgentService` wraps a `ChatService`.
It reserves `tool_cost` tokens for the tool schemas: `tool_overhead_tokens` plus `ToolSet.prompt_tokens()`.
The wrapped service fits the history into `input_budget - tool_cost`.
An empty tool set costs 0.

1. **Prepare:** `AgentService.prepare()` delegates to `ChatService.prepare()`.
2. **Step:** `stream_reply()` calls `ChatModel.stream()` with the window and the tool specs. The last step, and every step with an empty set, sends `tools=()`.
3. **Text:** the service yields each `TextDelta` at once.
4. **Final reply:** a step without tool calls ends the turn. The service saves the user message and the text of that step, then yields `ReplyDone`.
5. **Tool calls:** otherwise, the service appends the assistant message with its `tool_calls`. It runs each call in order, yields `ToolStarted` and `ToolFinished`, and appends one `"tool"` message per result.
6. **Budget check:** before the next step, the whole window must fit `input_budget - tool_cost`. If not, the service raises `ContextOverflowError`.

A failed call becomes an `error: ...` result, and the loop continues.
The service cuts each result to `max_tool_result_tokens * 3` bytes, with the marker ` [truncated]`.
It saves only the user message and the final reply, never the tool steps.
An error, an early close, or a cancellation saves nothing.

## Dependency rule

- Apps depend on the framework. The framework never depends on an app.
- `agent/` depends on `chat/`. `chat/` never depends on `agent/`.
- The framework imports only the Python standard library and itself.
- The framework logs nothing. Apps log from the data that the service returns.
- `tests/test_architecture.py` enforces the import rules.

## Growth path

| Need | Path |
|---|---|
| A new model provider | The app implements the `ChatModel` port. The framework does not change. |
| A new store | The app implements the `ChatRepository` port. The framework does not change. |
| A real tokenizer | The app implements the `TokenEstimator` port and passes it to `ChatService` or `AgentService`. |
| A new tool | The app implements the `Tool` port and adds it to the `ToolSet`. The framework does not change. |
| A new provider with tools | The app measures `tool_overhead_tokens` once and passes it to `AgentService`. See "Tool overhead" in `README.md`. |
| A new domain error | Derive it from `MiniDeepAgentsError`. |

## Versioning

- The framework uses semantic versioning (semver).
- Each release is a git tag, such as `v0.2.0`.
- While the version is 0.x, a breaking change bumps the minor version.
- A removed or renamed export in `mini_deepagents/__init__.py` is a breaking change.
- Apps pin an exact git tag.

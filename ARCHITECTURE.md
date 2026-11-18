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
  chat/                the chat context
    domain.py          Message, Session, Usage, TextDelta, StreamEnd, ReplyDone
    ports.py           ChatModel, ChatRepository, TokenEstimator (Protocols)
    context.py         ByteTokenEstimator, fit_messages (the token budget)
    service.py         ChatService: prepare one turn, stream it, save it
  testing/
    fakes.py           FakeChatModel, InMemoryChatRepository
tests/
  chat/                tests of the chat context
  test_architecture.py stdlib-only imports, no logging
  test_public_api.py   the exact export set
```

## Ports

Each port lives in `chat/ports.py`, on the consumer side.
Each port docstring states the full contract.
In short:

| Port | Contract |
|---|---|
| `ChatModel.stream()` | Yield zero or more `TextDelta` events with non-empty text, then exactly one `StreamEnd`, last. Raise `ModelError` on any failure. An empty reply is a failure. Stop the upstream call when the consumer closes the stream early. |
| `ChatModel.aclose()` | Release the resources of the model. |
| `ChatRepository` | Create and list sessions. Return messages oldest first. `add_exchange()` writes the user message and the reply in one transaction. Raise `SessionNotFoundError` for an unknown session. |
| `TokenEstimator.estimate()` | Return the token cost of one message. Never underestimate. |

## One chat turn

1. **Prepare:** `ChatService.prepare()` loads the history and appends the new user message.
2. **Fit:** `fit_messages()` keeps the newest messages that fit the input budget. The window starts with a user message. The call writes nothing.
3. **Stream:** `ChatService.stream_reply()` calls `ChatModel.stream()` and yields each `TextDelta`.
4. **Save on `StreamEnd`:** the service calls `ChatRepository.add_exchange()` only after the model stream completes.
5. **`ReplyDone`:** the service yields `ReplyDone` with the reply length and the token usage.

An error, an early close, or a cancellation saves nothing.

## Dependency rule

- Apps depend on the framework. The framework never depends on an app.
- The framework imports only the Python standard library and itself.
- The framework logs nothing. Apps log from the data that the service returns.
- `tests/test_architecture.py` enforces the import rules.

## Growth path

| Need | Path |
|---|---|
| A new model provider | The app implements the `ChatModel` port. The framework does not change. |
| A new store | The app implements the `ChatRepository` port. The framework does not change. |
| A real tokenizer | The app implements the `TokenEstimator` port and passes it to `ChatService`. |
| Tools and an agent loop | Add the `agent/` context as a sibling of `chat/`, with a `Tool` port and an agent loop. |
| A new domain error | Derive it from `MiniDeepAgentsError`. |

## Versioning

- The framework uses semantic versioning (semver).
- Each release is a git tag, such as `v0.1.0`.
- While the version is 0.x, a breaking change bumps the minor version.
- A removed or renamed export in `mini_deepagents/__init__.py` is a breaking change.
- Apps pin an exact git tag.

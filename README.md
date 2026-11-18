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
- **The context fitting rule:** `fit_messages` keeps the newest messages that fit a token budget.
  `ByteTokenEstimator` estimates each message as `ceil(utf8_bytes / 3) + 4` tokens, on the safe side.
- **Three ports:** `ChatModel`, `ChatRepository`, and `TokenEstimator`.
- **The domain errors:** `MiniDeepAgentsError` and its subclasses `SessionNotFoundError`, `ModelError`, and `ContextOverflowError`.
- **Fakes:** `FakeChatModel` and `InMemoryChatRepository` in `mini_deepagents.testing`.

The framework has no tools and no agent loop yet.
Today it runs one chat turn inside a hard token budget.

## Install

```sh
uv add "mini-deepagents @ git+https://github.com/rizquuula/mini-deepagents@v0.1.0"
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
[Message(role='user', content='Hi!'), Message(role='assistant', content='Hello world')]
```

## Write an adapter

An adapter connects the framework to a real model or a real store.

1. Write a class that implements the `ChatModel` or the `ChatRepository` protocol.
2. Read the contract in each docstring of `mini_deepagents/chat/ports.py`.
   Keep every rule of that contract.
3. Translate each vendor error into a domain error, such as `ModelError`.
4. Pass an instance of your class to `ChatService`.

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

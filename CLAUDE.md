# mini-deepagents

A minimal agent harness framework for models with an 8K-token context window. It is a Python package, managed with `uv`, with zero runtime dependencies. Today it runs one chat turn inside a hard token budget, with or without native tool calls in a loop.

## Why this project exists

- Deep Agents assumes a window of 100K+ tokens. Its 8 default tool schemas alone cost about 2,480 tokens, about 30% of an 8K window.
- Many small and local models have a window of 8K tokens or less. This project targets them.
- The owner's bet: a small context window can still give good results with a good harness and a good agent loop.
- This project exists to prove that bet. See `README.md` for the full story.

## Design rules that follow from the motivation

- Every model request must fit in the context window, prompt plus output. Never break this cap.
- Treat each prompt token as a cost. Before you add prompt text, a system prompt, or a tool schema, count its tokens and state the count.
- Measure, do not guess. Check a change against the real `prompt_tokens` that the model reports in a reference app.
- Build the simplest working version first. Add a feature only when the user asks for it.

## Framework rules

Read `ARCHITECTURE.md` before you change the structure.

- Keep zero runtime dependencies. Import only the Python standard library. `tests/test_architecture.py` enforces this.
- Keep only ports and domain logic in the framework. Vendor adapters live in apps, such as the demo app.
- Log nothing in the framework. Do not import `logging` anywhere in `src/`.
- Define each port in the `ports.py` of its context, on the consumer side. State the contract in each port docstring.
- Derive every error from `MiniDeepAgentsError` in `errors.py`.
- Treat `mini_deepagents/__init__.py` as the public API. A removed or renamed export is a breaking change.
- Use semver. While the version is 0.x, a breaking change bumps the minor version. Apps pin an exact git tag.
- Give every port a fake in `mini_deepagents.testing`.
- Add a new bounded context as a sibling package of `chat/` and `agent/`. Dependencies point inward: `agent/` uses `chat/`, never the reverse.
- Count the tool schemas in the budget. `AgentService` reserves `tool_overhead_tokens` plus `ToolSet.prompt_tokens()`. Measure the overhead per provider, as `README.md` describes.
- Save only the user message and the final reply of an agent turn. Never save the tool steps.

## Commands

- Show the targets: `make` (the same as `make help`)
- Run all tests: `make test` (`uv run pytest -q`)

## Tests: keep them lean and quick

- Keep the full suite under 5 seconds. A slow test is a bug: fix it or delete it.
- Make no real network call. Use the fakes in `mini_deepagents.testing`.
- Write one test per behaviour. Name the test by that behaviour.
- Use `pytest.mark.parametrize` for input variants. Do not copy a test to change one value.
- Prefer small fakes over `unittest.mock`. Do not patch private functions.
- Test our code, not the libraries.
- Do not use `sleep`, real clocks, or timing-based asserts.
- Do not add hooks or flags to production code only for tests.
- Add a test only for a new behaviour or a fixed bug. Delete a test that checks the same thing as another test.
- Use pytest. Ask before you add a test dependency.
- Drive async code with `asyncio.run(...)`. Do not add `pytest-asyncio`.

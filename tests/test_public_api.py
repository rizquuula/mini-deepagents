from __future__ import annotations

import mini_deepagents

EXPECTED_EXPORTS = {
    "ByteTokenEstimator",
    "ChatEvent",
    "ChatModel",
    "ChatRepository",
    "ChatService",
    "ContextOverflowError",
    "Message",
    "MiniDeepAgentsError",
    "ModelError",
    "ModelEvent",
    "PreparedTurn",
    "ReplyDone",
    "Role",
    "Session",
    "SessionNotFoundError",
    "StreamEnd",
    "TextDelta",
    "TokenEstimator",
    "Usage",
    "__version__",
    "fit_messages",
}


def test_public_api_exports_exactly_the_expected_names() -> None:
    assert set(mini_deepagents.__all__) == EXPECTED_EXPORTS
    for name in EXPECTED_EXPORTS:
        getattr(mini_deepagents, name)

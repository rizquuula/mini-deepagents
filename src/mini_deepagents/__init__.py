"""mini-deepagents: a minimal agent harness for models with an 8K-token context window.

This module is the public API. A removed or renamed export is a breaking change.
"""

from __future__ import annotations

from mini_deepagents.chat import (
    ByteTokenEstimator,
    ChatEvent,
    ChatModel,
    ChatRepository,
    ChatService,
    Message,
    ModelEvent,
    PreparedTurn,
    ReplyDone,
    Role,
    Session,
    StreamEnd,
    TextDelta,
    TokenEstimator,
    ToolCall,
    ToolCallRequested,
    ToolSpec,
    Usage,
    fit_messages,
)
from mini_deepagents.errors import (
    ContextOverflowError,
    MiniDeepAgentsError,
    ModelError,
    SessionNotFoundError,
)

__version__ = "0.1.0"

__all__ = [
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
    "ToolCall",
    "ToolCallRequested",
    "ToolSpec",
    "Usage",
    "__version__",
    "fit_messages",
]

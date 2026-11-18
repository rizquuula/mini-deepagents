"""mini-deepagents: a minimal agent harness for models with an 8K-token context window.

This module is the public API. A removed or renamed export is a breaking change.
"""

from __future__ import annotations

from mini_deepagents.agent import (
    AgentEvent,
    AgentService,
    Tool,
    ToolFinished,
    ToolSet,
    ToolStarted,
)
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
    ToolConfigError,
)

__version__ = "0.2.0"

__all__ = [
    "AgentEvent",
    "AgentService",
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
    "Tool",
    "ToolCall",
    "ToolCallRequested",
    "ToolConfigError",
    "ToolFinished",
    "ToolSet",
    "ToolSpec",
    "ToolStarted",
    "Usage",
    "__version__",
    "fit_messages",
]

"""Value types of the chat domain. They hold data only and import no framework."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, TypeAlias

Role: TypeAlias = Literal["user", "assistant", "tool"]


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """The schema of one tool that the model may call.

    ``parameters`` is a JSON Schema object that describes the arguments.
    """

    name: str
    description: str
    parameters: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class ToolCall:
    """One tool call that the model requests.

    ``arguments`` is the raw JSON string from the model. It may be invalid.
    """

    id: str
    name: str
    arguments: str


@dataclass(frozen=True, slots=True)
class Message:
    """One chat message.

    An assistant message may carry ``tool_calls``. A ``"tool"`` message
    carries the result of one call and names it in ``tool_call_id``.
    """

    role: Role
    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None


@dataclass(frozen=True, slots=True)
class Session:
    """One chat session. ``created_at`` is an ISO 8601 string."""

    id: int
    created_at: str


@dataclass(frozen=True, slots=True)
class Usage:
    """Token counts that the model reports. A field is None when unknown."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class TextDelta:
    """One piece of reply text from the model."""

    text: str


@dataclass(frozen=True, slots=True)
class ToolCallRequested:
    """One complete tool call from the model stream."""

    call: ToolCall


@dataclass(frozen=True, slots=True)
class StreamEnd:
    """The last event of a model stream."""

    usage: Usage


@dataclass(frozen=True, slots=True)
class ReplyDone:
    """The last event that the chat service or the agent service yields.

    ``steps`` counts the model calls of the turn. ``peak_estimated_tokens`` is
    the largest prompt estimate of one model call, or None when the service
    does not report it.
    """

    reply_chars: int
    usage: Usage
    steps: int = 1
    peak_estimated_tokens: int | None = None


ModelEvent: TypeAlias = TextDelta | ToolCallRequested | StreamEnd
ChatEvent: TypeAlias = TextDelta | ReplyDone

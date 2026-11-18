"""Value types of the agent domain. They hold data only and import no framework."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from mini_deepagents.chat.domain import ReplyDone, TextDelta, ToolCall


@dataclass(frozen=True, slots=True)
class ToolStarted:
    """The agent service starts to run one tool call."""

    call: ToolCall


@dataclass(frozen=True, slots=True)
class ToolFinished:
    """The agent service finished one tool call.

    ``ok`` is False when the result is an error text. ``result_chars`` counts
    the characters of the result that the model sees, after the cut.
    """

    call_id: str
    name: str
    ok: bool
    result_chars: int
    duration_ms: int


AgentEvent: TypeAlias = TextDelta | ToolStarted | ToolFinished | ReplyDone

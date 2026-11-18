"""Ports of the chat domain.

The domain defines each port here, on the consumer side. Adapters implement
the ports. Each docstring states the contract that every implementation must
keep.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Sequence
from typing import Protocol

from mini_deepagents.chat.domain import Message, ModelEvent, Session, ToolSpec


class ChatModel(Protocol):
    """A model that streams a reply to a chat history."""

    def stream(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int,
        session_key: str,
        tools: Sequence[ToolSpec] = (),
    ) -> AsyncGenerator[ModelEvent, None]:
        """Stream the reply to ``messages``, oldest first.

        Contract:
            - Yield ``TextDelta`` events with non-empty text and
              ``ToolCallRequested`` events, each with one complete call.
            - Then yield exactly one ``StreamEnd`` event, last.
            - Raise ``ModelError`` on any failure. A stream with no text
              and no tool call is a failure.
            - Send ``tools`` to the provider only when the sequence is not
              empty.
            - Send each message with its ``tool_calls`` and its
              ``tool_call_id``.
            - When the consumer closes the iterator early, stop the
              upstream call.
        """
        ...

    async def aclose(self) -> None:
        """Release the resources of the model, such as an HTTP client."""
        ...


class ChatRepository(Protocol):
    """The store for chat sessions and their messages."""

    def create_session(self) -> Session:
        """Create a new empty session and return it."""
        ...

    def list_sessions(self) -> list[Session]:
        """Return all sessions, newest first."""
        ...

    def get_messages(self, session_id: int) -> list[Message]:
        """Return the messages of a session, oldest first.

        Raises:
            SessionNotFoundError: No session has ``session_id``.
        """
        ...

    def add_exchange(
        self, session_id: int, user_content: str, assistant_content: str
    ) -> None:
        """Append one user message and one assistant reply in one transaction.

        Raises:
            SessionNotFoundError: No session has ``session_id``. The call
                writes nothing.
        """
        ...


class TokenEstimator(Protocol):
    """A strategy that estimates the token cost of one message or one tool."""

    def estimate(self, message: Message) -> int:
        """Return the token estimate for ``message``. Never underestimate."""
        ...

    def estimate_tool(self, spec: ToolSpec) -> int:
        """Return the token estimate for the schema of one tool.

        Never underestimate. Exclude the fixed overhead that a provider adds
        when a request carries any tools.
        """
        ...

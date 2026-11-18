"""Ports of the agent domain.

The domain defines each port here, on the consumer side. Apps implement the
ports. Each docstring states the contract that every implementation must keep.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from mini_deepagents.chat.domain import ToolSpec


class Tool(Protocol):
    """One tool that the model may call."""

    @property
    def spec(self) -> ToolSpec:
        """The name, description, and JSON Schema that the model sees."""
        ...

    async def run(self, arguments: Mapping[str, object]) -> str:
        """Run the tool with the parsed arguments and return the result text.

        Contract:
            - ``arguments`` is the JSON object from the model. The tool
              checks the fields it needs.
            - Return the result as text. The agent service cuts a long
              result to its limit.
            - Raise any exception on failure. The agent service turns it
              into a failed result that the model sees.
        """
        ...

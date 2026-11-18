"""The set of tools that one agent may call."""

from __future__ import annotations

from collections.abc import Sequence

from mini_deepagents.agent.ports import Tool
from mini_deepagents.chat.domain import ToolSpec
from mini_deepagents.chat.ports import TokenEstimator
from mini_deepagents.errors import ToolConfigError

__all__ = ["ToolSet"]


class ToolSet:
    """An ordered set of tools with unique names."""

    def __init__(self, tools: Sequence[Tool]) -> None:
        """Build the set in the given order.

        Raises:
            ToolConfigError: Two tools have the same name.
        """
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            name = tool.spec.name
            if name in self._tools:
                raise ToolConfigError(f'duplicate tool name "{name}"')
            self._tools[name] = tool

    def __len__(self) -> int:
        return len(self._tools)

    def specs(self) -> tuple[ToolSpec, ...]:
        """Return the spec of each tool, in the given order."""
        return tuple(tool.spec for tool in self._tools.values())

    def get(self, name: str) -> Tool | None:
        """Return the tool with ``name``, or None when no tool has it."""
        return self._tools.get(name)

    def prompt_tokens(self, estimator: TokenEstimator) -> int:
        """Return the summed estimate of all specs. An empty set costs 0."""
        return sum(estimator.estimate_tool(spec) for spec in self.specs())

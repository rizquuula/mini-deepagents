from __future__ import annotations

import pytest

from mini_deepagents import ByteTokenEstimator, ToolConfigError, ToolSet
from mini_deepagents.testing import FakeTool

ESTIMATOR = ByteTokenEstimator()


def test_get_finds_a_tool_by_name_and_returns_none_for_an_unknown_name() -> None:
    clock, search = FakeTool("clock"), FakeTool("search")
    tools = ToolSet([clock, search])

    assert tools.get("search") is search
    assert tools.get("missing") is None
    assert tools.specs() == (clock.spec, search.spec)
    assert len(tools) == 2


def test_a_duplicate_name_raises_tool_config_error() -> None:
    with pytest.raises(ToolConfigError, match="clock"):
        ToolSet([FakeTool("clock"), FakeTool("clock")])


@pytest.mark.parametrize("names", [("clock",), ("clock", "search")])
def test_prompt_tokens_sums_the_estimate_of_each_spec(names: tuple[str, ...]) -> None:
    tools = ToolSet([FakeTool(name) for name in names])

    expected = sum(ESTIMATOR.estimate_tool(spec) for spec in tools.specs())
    assert tools.prompt_tokens(ESTIMATOR) == expected


def test_an_empty_set_costs_zero_tokens() -> None:
    assert ToolSet([]).prompt_tokens(ESTIMATOR) == 0

from __future__ import annotations

import ast
import sys
from pathlib import Path

ALLOWED = sys.stdlib_module_names | {"mini_deepagents"}
FORBIDDEN = frozenset({"logging"})
PACKAGE = Path(__file__).resolve().parents[1] / "src" / "mini_deepagents"


def _imported_modules(path: Path) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules.add(node.module)
            modules.update(f"{node.module}.{alias.name}" for alias in node.names)
    return modules


def _is_violation(module: str) -> bool:
    top_level = module.split(".")[0]
    return top_level not in ALLOWED or top_level in FORBIDDEN


def test_framework_imports_only_stdlib_and_itself_and_never_logging() -> None:
    violations = sorted(
        f"{path.relative_to(PACKAGE)}: {module}"
        for path in sorted(PACKAGE.rglob("*.py"))
        for module in _imported_modules(path)
        if _is_violation(module)
    )

    assert violations == []

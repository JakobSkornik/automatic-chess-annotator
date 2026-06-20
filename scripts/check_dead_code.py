#!/usr/bin/env python3
"""
Static dead-code detector for the `app/` package.

Line coverage gives false negatives here: a symbol exercised only by its unit
test (or a one-off script) still reports as "covered" while being unreachable
from the running service. This tool instead asks the structural question —
"is this reachable from the production entry point?" — by walking the import
graph and counting references, classifying every top-level symbol as live,
production-dead (kept alive only by tests/scripts), or fully dead.

Exit code is non-zero when fully-dead symbols exist, so CI can gate on it.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from collections import deque
from pathlib import Path

# --- constants -------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = REPO_ROOT / "app"
TEST_DIR = REPO_ROOT / "tests"
SCRIPT_DIR = REPO_ROOT / "scripts"

# The only production process: `uvicorn app.main:app`. Reachability starts here.
ENTRY_MODULES = ("app.main",)

# Decorators that make a symbol reachable through a framework rather than a
# direct call, so a zero reference count is expected and not evidence of death.
FRAMEWORK_DECORATORS = (
    "get", "post", "put", "patch", "delete", "websocket",
    "on_event", "exception_handler", "middleware", "fixture",
)

PACKAGE_PREFIX = "app"


# --- public API ------------------------------------------------------------


def module_name_of(path: Path) -> str:
    """Dotted module name for a file under the repo root (drops the .py suffix)."""
    return ".".join(path.relative_to(REPO_ROOT).with_suffix("").parts)


def unreachable_modules() -> list[str]:
    """App modules not import-reachable from the production entry point(s)."""
    graph = _import_graph()
    reached = _reachable_from(graph, ENTRY_MODULES)
    candidates = {m for m in graph if not m.endswith("__init__")}
    return sorted(candidates - reached)


def classify_symbols() -> tuple[list[Symbol], list[Symbol]]:
    """Split top-level app symbols into (fully_dead, production_dead)."""
    symbols = _collect_symbols()
    app_sources = _sources_in(APP_DIR)
    test_sources = _sources_in(TEST_DIR)
    script_sources = _sources_in(SCRIPT_DIR)

    fully_dead: list[Symbol] = []
    production_dead: list[Symbol] = []
    for symbol in symbols:
        if symbol.is_framework_reachable:
            continue
        symbol.attach_reference_counts(app_sources, test_sources, script_sources)
        if symbol.is_production_dead:
            (fully_dead if symbol.is_fully_dead else production_dead).append(symbol)
    return fully_dead, production_dead


# --- privates --------------------------------------------------------------


class Symbol:
    """A top-level function or class defined in the app package."""

    def __init__(self, name: str, path: Path, lineno: int, decorators: list[str]):
        self.name = name
        self.path = path
        self.lineno = lineno
        self._decorators = decorators
        self.production_refs = 0
        self.internal_refs = 0
        self.test_refs = 0
        self.script_refs = 0

    @property
    def is_framework_reachable(self) -> bool:
        return any(d in FRAMEWORK_DECORATORS for d in self._decorators)

    @property
    def has_live_callers(self) -> bool:
        """Referenced by other production modules or by sibling code in its own module."""
        return self.production_refs > 0 or self.internal_refs > 0

    @property
    def is_production_dead(self) -> bool:
        return not self.has_live_callers

    @property
    def is_fully_dead(self) -> bool:
        return self.is_production_dead and self.test_refs == 0 and self.script_refs == 0

    @property
    def location(self) -> str:
        return f"{self.path.relative_to(REPO_ROOT)}:{self.lineno}"

    @property
    def kept_alive_by(self) -> str:
        keepers = []
        if self.test_refs:
            keepers.append("tests")
        if self.script_refs:
            keepers.append("scripts")
        return ", ".join(keepers) or "nothing"

    def attach_reference_counts(self, app_src, test_src, script_src) -> None:
        self.production_refs = _count_references(
            self.name, {p: s for p, s in app_src.items() if p != self.path}
        )
        self.internal_refs = _count_references(
            self.name, {self.path: app_src[self.path]}, skip_line=self.lineno
        )
        self.test_refs = _count_references(self.name, test_src)
        self.script_refs = _count_references(self.name, script_src)


def _python_files(base: Path) -> list[Path]:
    return [p for p in base.rglob("*.py") if "__pycache__" not in p.parts]


def _sources_in(base: Path) -> dict[Path, str]:
    return {p: p.read_text(encoding="utf-8") for p in _python_files(base)}


def _app_imports(path: Path) -> set[str]:
    """App module names imported by a file (resolved against real modules)."""
    tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(a.name for a in node.names if a.name.startswith(PACKAGE_PREFIX))
        elif isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
            PACKAGE_PREFIX
        ):
            found.add(node.module)
            found.update(f"{node.module}.{a.name}" for a in node.names)
    return found


def _import_graph() -> dict[str, set[str]]:
    files = _python_files(APP_DIR)
    known = {module_name_of(p) for p in files}
    return {
        module_name_of(p): {dep for dep in _app_imports(p) if dep in known}
        for p in files
    }


def _reachable_from(graph: dict[str, set[str]], roots: tuple[str, ...]) -> set[str]:
    reached: set[str] = set()
    queue = deque(roots)
    while queue:
        module = queue.popleft()
        if module in reached:
            continue
        reached.add(module)
        queue.extend(graph.get(module, ()))
    return reached


def _decorator_names(node: ast.AST) -> list[str]:
    names = []
    for decorator in getattr(node, "decorator_list", []):
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        if isinstance(target, ast.Attribute):
            names.append(target.attr)
        elif isinstance(target, ast.Name):
            names.append(target.id)
    return names


def _collect_symbols() -> list[Symbol]:
    defined = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    symbols: list[Symbol] = []
    for path in _python_files(APP_DIR):
        if path.name == "__init__.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
        for node in tree.body:
            if isinstance(node, defined):
                symbols.append(
                    Symbol(node.name, path, node.lineno, _decorator_names(node))
                )
    return symbols


def _count_references(
    name: str, sources: dict[Path, str], skip_line: int | None = None
) -> int:
    pattern = re.compile(rf"\b{re.escape(name)}\b")
    total = 0
    for source in sources.values():
        for lineno, line in enumerate(source.splitlines(), start=1):
            if lineno == skip_line:
                continue
            total += len(pattern.findall(line))
    return total


def _print_report(fully_dead: list[Symbol], production_dead: list[Symbol]) -> None:
    orphans = unreachable_modules()
    print(f"Unreachable modules from {ENTRY_MODULES}: {len(orphans)}")
    for module in orphans:
        print(f"  {module}")

    print(f"\nFully dead symbols (no references anywhere): {len(fully_dead)}")
    for symbol in sorted(fully_dead, key=lambda s: s.location):
        print(f"  {symbol.name}  @ {symbol.location}")

    print(
        f"\nProduction-dead symbols (kept alive only by tests/scripts): "
        f"{len(production_dead)}"
    )
    for symbol in sorted(production_dead, key=lambda s: s.location):
        print(f"  {symbol.name}  @ {symbol.location}  [{symbol.kept_alive_by}]")


# --- main ------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="also fail when production-dead (test/script-only) symbols exist",
    )
    args = parser.parse_args()

    fully_dead, production_dead = classify_symbols()
    _print_report(fully_dead, production_dead)

    if fully_dead or (args.strict and production_dead):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Pick the gallery examples a change set needs to re-execute.

Reads changed file paths (one per line, relative to the repo root) on
stdin and prints a regex for sphinx-gallery's ``filename_pattern``, which
docs/source/conf.py reads from ``PHYSICSKIT_GALLERY_PATTERN``. Used by
.github/workflows/docs.yml on pull requests and pushes to main::

    git diff --name-only main... | python docs/select_examples.py

An example is selected when the change set edits it, or edits a
physicskit module it imports, directly or through other physicskit
modules. ``import physicskit as pk`` alone counts only the modules the
example then reaches as ``pk.a.b``, since the top-level package imports
every subpackage and would otherwise select the whole gallery. An edited
package ``__init__.py`` selects every example that imports anything
inside that package, because Python runs it first. A change to the docs
configuration, this script or the package dependencies selects the full
gallery, as does a change to the Docs workflow; a change set that
touches none of the above selects nothing.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "physicskit"
EXAMPLES = ROOT / "examples"

FULL_GALLERY = r"/plot_"  # conf.py's default filename_pattern
NO_EXAMPLES = r"(?!)"  # matches no path

# Files whose change can alter how every example runs.
FULL_GALLERY_TRIGGERS = {
    ".github/workflows/docs.yml",
    "docs/source/conf.py",
    "docs/select_examples.py",
    "pyproject.toml",
}


def _module_name(path: Path) -> str:
    parts = list(path.relative_to(ROOT).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _package_modules() -> dict[str, Path]:
    return {_module_name(p): p for p in PACKAGE.rglob("*.py") if "tests" not in p.relative_to(PACKAGE).parts}


def _resolve(name: str, modules: dict[str, Path]) -> str | None:
    """Longest prefix of dotted ``name`` that is a physicskit module."""
    parts = name.split(".")
    for n in range(len(parts), 0, -1):
        candidate = ".".join(parts[:n])
        if candidate in modules:
            return candidate
    return None


def _attribute_chain(node: ast.Attribute) -> str | None:
    parts = []
    current: ast.expr = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    parts.append(current.id)
    return ".".join(reversed(parts))


def _imports(path: Path, modules: dict[str, Path]) -> set[str]:
    """The physicskit modules ``path`` imports."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    is_package = path.name == "__init__.py"
    own = _module_name(path) if path.is_relative_to(PACKAGE) else ""
    found: set[str] = set()
    aliases: dict[str, str] = {}  # local name -> module it is bound to

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    aliases[alias.asname] = alias.name
                else:
                    # ``import a.b.c`` binds ``a``; the module itself is used.
                    target = _resolve(alias.name, modules)
                    if target:
                        found.add(target)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base_parts = own.split(".")
                drop = node.level - 1 if is_package else node.level
                base_parts = base_parts[: len(base_parts) - drop] if drop else base_parts
                base = ".".join(base_parts + ([node.module] if node.module else []))
            else:
                base = node.module or ""
            for alias in node.names:
                target = _resolve(f"{base}.{alias.name}", modules) or _resolve(base, modules)
                if target:
                    found.add(target)

    # ``import physicskit.rmt as rmt`` then ``rmt.ensembles.syk.f``: follow
    # the attribute chain to the deepest module it names.
    if aliases:
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                chain = _attribute_chain(node)
                if chain is None:
                    continue
                head, _, rest = chain.partition(".")
                if head in aliases and rest:
                    target = _resolve(f"{aliases[head]}.{rest}", modules)
                    if target:
                        found.add(target)
    return found


def _closure(start: set[str], graph: dict[str, set[str]]) -> set[str]:
    seen: set[str] = set()
    stack = list(start)
    while stack:
        mod = stack.pop()
        if mod in seen:
            continue
        seen.add(mod)
        stack.extend(graph.get(mod, ()))
    return seen


def _with_parents(modules: set[str]) -> set[str]:
    out = set(modules)
    for mod in modules:
        parts = mod.split(".")
        out.update(".".join(parts[:n]) for n in range(1, len(parts)))
    return out


def select(changed: list[str]) -> list[Path]:
    """Examples to execute for ``changed``, or every example for a full build."""
    all_examples = sorted(EXAMPLES.rglob("plot_*.py"))
    if any(f in FULL_GALLERY_TRIGGERS for f in changed):
        return all_examples

    modules = _package_modules()
    changed_modules: set[str] = set()
    selected: set[Path] = set()
    for f in changed:
        path = ROOT / f
        if path in modules.values():
            changed_modules.add(_module_name(path))
        elif path.is_relative_to(PACKAGE) and "tests" not in path.parts and path.exists():
            # A data file counts as a change to the package holding it.
            for parent in path.parents:
                if (parent / "__init__.py").exists():
                    changed_modules.add(_module_name(parent / "__init__.py"))
                    break
        elif path.is_relative_to(EXAMPLES) and path.name.startswith("plot_") and path.suffix == ".py" and path.exists():
            selected.add(path)

    if changed_modules:
        graph = {name: _imports(p, modules) for name, p in modules.items()}
        for example in all_examples:
            reached = _with_parents(_closure(_imports(example, modules), graph))
            if reached & changed_modules:
                selected.add(example)
    return sorted(selected)


def pattern(selected: list[Path]) -> str:
    if not selected:
        return NO_EXAMPLES
    if len(selected) == len(list(EXAMPLES.rglob("plot_*.py"))):
        return FULL_GALLERY
    return "|".join(re.escape(p.relative_to(ROOT).as_posix()) + "$" for p in selected)


def main() -> None:
    changed = [line.strip() for line in sys.stdin if line.strip()]
    selected = select(changed)
    for p in selected:
        print(p.relative_to(ROOT).as_posix(), file=sys.stderr)
    print(f"{len(selected)} example(s) selected", file=sys.stderr)
    print(pattern(selected))


if __name__ == "__main__":
    main()

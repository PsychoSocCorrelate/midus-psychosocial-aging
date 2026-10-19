#!/usr/bin/env python3
"""Fail when a registry step runs a script by path that cannot import the package.

Python puts a script's own directory on sys.path, not the repository root, so
`uv run python analysis/figures/x.py` cannot `from analysis.core import ...`
unless the script inserts the root itself. Steps invoked with `-m` are exempt.

    uv run python scripts/validation/check_script_imports.py
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "scripts" / "orchestration" / "registry.json"
PACKAGES = {"analysis", "config", "preprocess", "scripts"}


def _imports_package(tree: ast.AST) -> bool:
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and (n.module or "").split(".")[0] in PACKAGES:
            return True
        if isinstance(n, ast.Import) and any(
                a.name.split(".")[0] in PACKAGES for a in n.names):
            return True
    return False


def main() -> int:
    steps = json.loads(REGISTRY.read_text())["steps"]
    bad: list[str] = []
    checked = 0
    for s in steps:
        cmd = s.get("command") or []
        if cmd[:3] != ["uv", "run", "python"] or len(cmd) < 4 or cmd[3] == "-m":
            continue
        script = ROOT / cmd[3]
        if not script.exists():
            bad.append(f"{s['id']}: command names {cmd[3]}, which does not exist")
            continue
        src = script.read_text()
        checked += 1
        if _imports_package(ast.parse(src)) and "sys.path.insert" not in src:
            bad.append(
                f"{s['id']}: {cmd[3]} imports the package but never puts the "
                f"repository root on sys.path, so running it by path raises "
                f"ModuleNotFoundError"
            )

    for line in bad:
        print(f"  {line}")
    print(f"{checked} path-invoked step(s) checked, {len(bad)} problem(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

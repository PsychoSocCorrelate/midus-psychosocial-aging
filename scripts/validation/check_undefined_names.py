#!/usr/bin/env python3
"""Fail on any name used but never bound (ruff F821) in the tracked source.

py_compile does not execute code, so it passes these; the pipelines cannot be
smoke-tested by importing, because importing one runs it.

    uv run python scripts/validation/check_undefined_names.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGETS = ["analysis", "preprocess", "scripts", "config"]


def main() -> int:
    proc = subprocess.run(
        ["uv", "run", "--with", "ruff", "ruff", "check", "--select", "F821",
         "--no-cache", *TARGETS],
        cwd=ROOT,
    )
    if proc.returncode:
        print("\nF821 is a name that is used and never bound. It is a NameError "
              "waiting for the branch that reaches it, not a style preference.")
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())

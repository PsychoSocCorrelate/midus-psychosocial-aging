"""Orchestrate preprocessing and master dataset rebuild steps."""

from __future__ import annotations

import sys

from runner import main_for_groups


if __name__ == "__main__":
    sys.exit(
        main_for_groups(
            sys.argv[1:],
            ["preprocess", "build"],
            "Rebuild the raw inputs into master analysis files",
        )
    )

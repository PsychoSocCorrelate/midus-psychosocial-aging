#!/usr/bin/env python3
"""Orchestrate manuscript figure rendering."""

from __future__ import annotations

import sys

from runner import main_for_groups


if __name__ == "__main__":
    sys.exit(main_for_groups(sys.argv[1:], ["figures"], "Render/check manuscript figures"))

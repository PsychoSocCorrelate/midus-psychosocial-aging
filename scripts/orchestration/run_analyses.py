#!/usr/bin/env python3
"""Orchestrate analysis runs with dry-run and heavy-rerun guards."""

from __future__ import annotations

import sys

from runner import main_for_groups


if __name__ == "__main__":
    sys.exit(main_for_groups(sys.argv[1:], ["analyses"], "Run or list analysis steps"))

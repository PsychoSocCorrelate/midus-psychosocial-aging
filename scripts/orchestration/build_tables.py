#!/usr/bin/env python3
"""Orchestrate manuscript and supplement table builders."""

from __future__ import annotations

import sys

from runner import main_for_groups


if __name__ == "__main__":
    sys.exit(main_for_groups(sys.argv[1:], ["tables"], "Build manuscript and supplement tables"))

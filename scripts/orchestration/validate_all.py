#!/usr/bin/env python3
"""Run the non-mutating validation steps."""

from __future__ import annotations

import argparse
import sys

from runner import (
    add_common_args,
    load_registry,
    run_steps,
    select_steps,
    validate_step_ids,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the validation steps")
    add_common_args(parser)
    args = parser.parse_args(argv)

    registry = load_registry()
    validate_step_ids(registry, args.step)
    include_groups = list(args.include)
    steps = select_steps(registry, ["validation"], include_groups, args.step)
    return run_steps(
        steps,
        registry=registry,
        execute=args.execute,
        include_heavy=args.include_heavy,
        keep_going=args.keep_going,
    )


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

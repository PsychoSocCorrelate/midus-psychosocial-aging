#!/usr/bin/env python3
"""Shared helpers for registry-backed orchestration scripts."""

from __future__ import annotations

import argparse
from html.parser import HTMLParser
import json
from pathlib import Path
import subprocess
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "scripts" / "orchestration" / "registry.json"


def load_registry(path: Path = REGISTRY) -> dict:
    return json.loads(path.read_text())


def shell_join(cmd: list[str]) -> str:
    return " ".join(cmd)


def _tracked_python_under(prefixes: Iterable[str]) -> list[str]:
    prefix_tuple = tuple(p.rstrip("/") + "/" for p in prefixes)
    tracked = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.splitlines()
    return [
        p for p in tracked
        if p.endswith(".py") and (p in prefixes or p.startswith(prefix_tuple))
    ]


def select_steps(
    registry: dict,
    groups: Iterable[str],
    include_groups: Iterable[str] = (),
    ids: Iterable[str] = (),
) -> list[dict]:
    base_groups = set(groups)
    extra_groups = set(include_groups)
    requested_ids = set(ids)
    selected: list[dict] = []
    for step in registry["steps"]:
        if requested_ids:
            if step["id"] in requested_ids:
                selected.append(step)
            continue
        if step["group"] in base_groups and step.get("default", False):
            selected.append(step)
        elif step["group"] in extra_groups:
            selected.append(step)
    return selected


def _norm_path(path: str) -> str:
    """Strip a trailing slash."""
    return path.rstrip("/")


def _producers(registry: dict) -> dict[str, str]:
    """Map each declared output path to the step id that writes it."""
    out: dict[str, str] = {}
    for step in registry["steps"]:
        if step.get("kind") == "check_exists":
            continue  # its `outputs` are preconditions, not products
        for path in step.get("outputs", ()):
            out.setdefault(_norm_path(path), step["id"])
    return out


def _producer_of(producers: dict[str, str], path: str) -> str | None:
    """Return the step that writes `path` or its nearest declared parent directory."""
    path = _norm_path(path)
    if path in producers:
        return producers[path]
    parts = path.split("/")
    for i in range(len(parts) - 1, 0, -1):
        parent = "/".join(parts[:i])
        if parent in producers:
            return producers[parent]
    return None


def dependency_edges(registry: dict) -> set[tuple[str, str]]:
    """Edges producer -> consumer, from declared inputs and outputs."""
    producers = _producers(registry)
    edges: set[tuple[str, str]] = set()
    for step in registry["steps"]:
        for path in step.get("inputs", ()):
            src = _producer_of(producers, path)
            if src and src != step["id"]:
                edges.add((src, step["id"]))
    return edges


def find_cycles(registry: dict) -> list[list[str]]:
    """Return any dependency cycles among declared steps."""
    graph: dict[str, set[str]] = {}
    for src, dst in dependency_edges(registry):
        graph.setdefault(src, set()).add(dst)
    cycles: list[list[str]] = []
    state: dict[str, int] = {}

    def visit(node: str, path: list[str]) -> None:
        state[node] = 1
        path.append(node)
        for nxt in sorted(graph.get(node, ())):
            if state.get(nxt) == 1:
                cycles.append(path[path.index(nxt):] + [nxt])
            elif not state.get(nxt):
                visit(nxt, path)
        path.pop()
        state[node] = 2

    for step in registry["steps"]:
        if not state.get(step["id"]):
            visit(step["id"], [])
    return cycles


def check_order(registry: dict, steps: list[dict]) -> list[str]:
    """Report selected steps that consume an output produced later in the run.

    Steps run in list order, so such a step would read a stale copy or miss the input.
    """
    problems = [
        f"cycle: {' -> '.join(cycle)}" for cycle in find_cycles(registry)
    ]
    position = {step["id"]: i for i, step in enumerate(steps)}
    for src, dst in sorted(dependency_edges(registry)):
        if src in position and dst in position and position[src] > position[dst]:
            problems.append(
                f"{dst} consumes an output of {src}, which the registry runs after it"
            )
    return problems


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute selected steps. Omit for dry-run.",
    )
    parser.add_argument(
        "--include-heavy",
        action="store_true",
        help="Allow heavy model-fitting/bootstrap steps to execute.",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        metavar="GROUP",
        help="Include an optional registry group.",
    )
    parser.add_argument(
        "--step",
        action="append",
        default=[],
        metavar="ID",
        help="Run/list only a specific step id. Can be repeated.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue after a failed step and report failures at the end.",
    )


def describe_steps(steps: list[dict], include_heavy: bool) -> None:
    print(f"Selected {len(steps)} step(s)")
    for idx, step in enumerate(steps, 1):
        flags = []
        if step.get("heavy", False):
            flags.append("heavy")
            if not include_heavy:
                flags.append("guarded")
        if not step.get("default", False):
            flags.append("optional")
        flag_text = f" [{' '.join(flags)}]" if flags else ""
        if step["kind"] == "command":
            action = shell_join(step["command"])
        elif step["kind"] == "check_exists":
            action = "check exists: " + ", ".join(step.get("outputs", []))
        elif step["kind"] in {"html_parse", "inline_js_parse", "py_compile"}:
            action = f"{step['kind']}: " + ", ".join(step.get("paths", []))
        elif step["kind"] == "git_clean":
            action = "git status --short"
        else:
            action = step["kind"]
        print(f"{idx:02d}. {step['id']}{flag_text}")
        print(f"    {action}")


def _check_paths(paths: Iterable[str]) -> None:
    missing = [p for p in paths if not (ROOT / p).exists()]
    if missing:
        raise RuntimeError("missing required file(s): " + ", ".join(missing))


def _html_parse(paths: Iterable[str]) -> None:
    parser = HTMLParser()
    for rel in paths:
        parser.feed((ROOT / rel).read_text())


def _inline_js_parse(paths: Iterable[str]) -> None:
    node_script = r"""
const fs = require('fs');
const vm = require('vm');
for (const path of process.argv.slice(1)) {
  const html = fs.readFileSync(path, 'utf8');
  const scripts = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m => m[1]);
  for (const script of scripts) new vm.Script(script);
}
"""
    subprocess.run(["node", "-e", node_script, *[str(ROOT / p) for p in paths]], cwd=ROOT, check=True)


def _py_compile(paths: Iterable[str]) -> None:
    py_files = _tracked_python_under(paths)
    if not py_files:
        return
    subprocess.run(["uv", "run", "python", "-m", "py_compile", *py_files], cwd=ROOT, check=True)


def _git_clean() -> None:
    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    if status:
        raise RuntimeError("working tree is not clean:\n" + status)


def execute_step(step: dict) -> None:
    kind = step["kind"]
    if kind == "command":
        subprocess.run(step["command"], cwd=ROOT, check=True)
    elif kind == "check_exists":
        _check_paths(step.get("outputs", []))
    elif kind == "html_parse":
        _html_parse(step.get("paths", []))
    elif kind == "inline_js_parse":
        _inline_js_parse(step.get("paths", []))
    elif kind == "py_compile":
        _py_compile(step.get("paths", []))
    elif kind == "git_clean":
        _git_clean()
    else:
        raise RuntimeError(f"unknown orchestration step kind: {kind}")


def run_steps(
    steps: list[dict],
    *,
    registry: dict,
    execute: bool,
    include_heavy: bool,
    keep_going: bool,
) -> int:
    describe_steps(steps, include_heavy)

    order_problems = check_order(registry, steps)
    if order_problems:
        print("\nDependency order is not satisfiable:")
        for problem in order_problems:
            print(f"- {problem}")
        print("\nFix the declared inputs/outputs or the step order in registry.json.")
        return 1

    if not execute:
        print("\nDry-run only. Add --execute to run selected non-heavy steps.")
        if any(s.get("heavy", False) for s in steps):
            print("Heavy steps require --include-heavy in addition to --execute.")
        return 0

    failures: list[tuple[str, str]] = []
    for step in steps:
        if step.get("heavy", False) and not include_heavy:
            print(f"\n[SKIP heavy] {step['id']} (add --include-heavy to execute)")
            continue
        print(f"\n[RUN] {step['id']}")
        try:
            execute_step(step)
        except Exception as exc:
            failures.append((step["id"], str(exc)))
            print(f"[FAIL] {step['id']}: {exc}")
            if not keep_going:
                break
    if failures:
        print("\nFailed step(s):")
        for step_id, err in failures:
            print(f"- {step_id}: {err}")
        return 1
    print("\nOrchestration completed.")
    return 0


def validate_step_ids(registry: dict, ids: Iterable[str]) -> None:
    available = {s["id"] for s in registry["steps"]}
    missing = sorted(set(ids) - available)
    if missing:
        raise SystemExit(f"unknown step id(s): {', '.join(missing)}")


def main_for_groups(argv: list[str] | None, groups: list[str], description: str) -> int:
    parser = argparse.ArgumentParser(description=description)
    add_common_args(parser)
    args = parser.parse_args(argv)
    registry = load_registry()
    validate_step_ids(registry, args.step)
    steps = select_steps(registry, groups, args.include, args.step)
    return run_steps(
        steps,
        registry=registry,
        execute=args.execute,
        include_heavy=args.include_heavy,
        keep_going=args.keep_going,
    )

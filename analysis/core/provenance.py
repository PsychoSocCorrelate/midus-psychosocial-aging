"""Stamp each analysis output directory with the inputs it was built from.

File modification times do not show which data an output was built from (a
rebuild that changes no value still moves every mtime), so each pipeline writes a
small stamp naming the master it read by content hash, plus the commit and
whether the run finished.

The stamp is written at exit; a run that raised records `completed: false`.
"""
from __future__ import annotations

import atexit
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STAMP = "provenance.json"


# Directory inputs: one analysis may consume another's whole output directory.
# shap_raw is large and not read by any downstream step; skip it.
_DIGEST_SKIP = {"shap_raw"}


def _file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256(path: Path) -> str | None:
    """Content digest of a file, or of a directory's CSV contents.

    A directory digest folds in each CSV's path and its hash, so adding, removing or
    editing any of them changes it. Returns None when the path is absent, which is
    recorded as None rather than as a hash of nothing.
    """
    if not path.exists():
        return None
    if path.is_file():
        return _file_sha(path)[:16]
    parts = []
    for f in sorted(path.rglob("*.csv")):
        if _DIGEST_SKIP & set(f.relative_to(path).parts):
            continue
        parts.append(f"{f.relative_to(path)}:{_file_sha(f)}")
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16] if parts else None


def _commit() -> str | None:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip() or None
    except Exception:
        return None


def _dirty() -> bool | None:
    try:
        r = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                           capture_output=True, text=True, timeout=15)
        return bool(r.stdout.strip())
    except Exception:
        return None


def write_stamp(out_dir: Path | str, pipeline: str, inputs: list[str],
                completed: bool) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "pipeline": pipeline,
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "completed": completed,
        "commit": _commit(),
        "working_tree_dirty": _dirty(),
        # Content hashes, so a changed input is detectable.
        "inputs": {p: _sha256(ROOT / p) for p in inputs},
    }
    path = out_dir / STAMP
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def stamp_at_exit(out_dir: Path | str, pipeline: str,
                  inputs: list[str] | None = None) -> None:
    """Register the stamp to be written when the process ends, however it ends.

    `completed` is false if the interpreter is unwinding an exception.
    """
    ins = list(inputs or ["data/processed/master_analysis.csv"])

    def _hook() -> None:
        completed = sys.exc_info()[0] is None
        write_stamp(out_dir, pipeline, ins, completed)

    atexit.register(_hook)


def read_stamp(out_dir: Path | str) -> dict | None:
    p = Path(out_dir) / STAMP
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def is_stale(out_dir: Path | str, inputs: list[str] | None = None) -> bool | None:
    """True if any recorded input hash no longer matches the file on disk.

    None when there is no stamp.
    """
    st = read_stamp(out_dir)
    if st is None:
        return None
    recorded = st.get("inputs") or {}
    for path, digest in recorded.items():
        if _sha256(ROOT / path) != digest:
            return True
    return False

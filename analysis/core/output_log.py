"""Write a Markdown log of a script's run settings and outputs."""

from __future__ import annotations

from argparse import Namespace
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any


def _format_value(value: Any) -> str:
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(v) for v in value)
    return str(value)


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def _infer_metadata(path: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if not path.exists():
        return metadata

    metadata["size_kb"] = round(path.stat().st_size / 1024, 1)

    suffix = path.suffix.lower()
    if suffix == ".csv":
        try:
            import pandas as pd

            df = pd.read_csv(path, nrows=0)
            metadata["columns"] = len(df.columns)
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                metadata["rows"] = max(sum(1 for _ in f) - 1, 0)
        except Exception:
            pass
    elif suffix == ".npy":
        try:
            import numpy as np

            arr = np.load(path, mmap_mode="r")
            metadata["shape"] = "x".join(str(dim) for dim in arr.shape)
        except Exception:
            pass

    return metadata


def _normalize_outputs(output_dir: Path, outputs: Sequence[Any]) -> list[dict[str, Any]]:
    normalized = []
    for item in outputs:
        if isinstance(item, Mapping):
            entry = dict(item)
            path = Path(entry["path"])
        else:
            entry = {"path": item}
            path = Path(item)

        if not path.is_absolute():
            path = output_dir / path
        inferred = _infer_metadata(path)
        for key, value in inferred.items():
            entry.setdefault(key, value)

        entry["path"] = path
        entry["exists"] = path.exists()
        entry.setdefault("kind", path.suffix.lstrip(".").upper() or "file")
        normalized.append(entry)
    return normalized


def write_output_log(
    output_dir: str | Path,
    log_name: str,
    script: str,
    outputs: Sequence[Any],
    run_args: Mapping[str, Any] | Namespace | None = None,
    notes: Sequence[str] | None = None,
) -> Path:
    """Write a Markdown log listing a script's outputs.

    `notes` are written under a Notes heading.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if isinstance(run_args, Namespace):
        args = vars(run_args)
    else:
        args = dict(run_args or {})

    lines = [
        "# Output Log",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Script: `{script}`",
        "",
    ]

    if args:
        lines += ["## Run Settings", "", "| Setting | Value |", "|---------|-------|"]
        for key in sorted(args):
            lines.append(f"| `{key}` | `{_format_value(args[key])}` |")
        lines.append("")

    if notes:
        lines += ["## Notes", ""] + list(notes) + [""]

    rows = _normalize_outputs(output_path, outputs)
    lines += ["## Outputs", "", "| Path | Kind | Exists | Details |", "|------|------|--------|---------|"]
    for row in rows:
        details = []
        for key in ("status", "rows", "columns", "shape", "size_kb"):
            if key in row and row[key] is not None:
                label = "size KB" if key == "size_kb" else key
                details.append(f"{label}={row[key]}")
        lines.append(
            f"| `{_display_path(row['path'])}` | {row['kind']} | "
            f"{'yes' if row['exists'] else 'no'} | {'; '.join(details)} |"
        )

    path = output_path / log_name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved: {path}")
    return path

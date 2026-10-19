#!/usr/bin/env python3
"""
A12: repeated-cross-validation intervals for the reported delta-R2 values.

Repeats the tiered comparison over `--repeats` cross-validation partitions,
varying only the fold assignment. The seed for repeat r is SEED_BASE + r, so
repeat 0 uses random_state=42 (the production default) and reproduces the value
the corresponding pipeline reports.

The interval is the spread of delta-R2 across cross-validation partitions of one
fixed sample. It is not a confidence interval for a population parameter: there
is no resampling of participants, so it says nothing about sampling error. Both
the per-repeat values and the summary are written.

With 15 repeats the 2.5th and 97.5th percentiles sit at the extremes of what was
observed. The observed minimum and maximum are therefore reported beside them,
because at this many repeats they are nearly the same statement.

Usage:
    uv run python analysis/pipelines/a12_delta_r2_ci.py
    uv run python analysis/pipelines/a12_delta_r2_ci.py --arms conditional --repeats 15
    uv run python analysis/pipelines/a12_delta_r2_ci.py --outcomes ALcomp_W1 --repeats 5
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analysis.core import age_method, provenance
from analysis.core.data_loader import W1_TO_W2, load_master
from analysis.core.output_log import write_output_log
from analysis.core.xgboost_pipeline import XGBoostPipeline

OUTPUT_DIR = ROOT / "output/a12_delta_r2_ci"

# Production seed; repeat 0 reproduces the pipeline's reported value.
SEED_BASE = 42

# Wave-1 names; the conditional arm maps them to wave 2 through W1_TO_W2.
PRIMARY = ["FInoPsy_W1", "btactComp_W1", "ALcomp_W1"]

ARMS = {
    # label: (tier configs, the delta to report, loader kwargs)
    "concurrent": (
        {"T1": [1], "T1T2": [1, 2]},
        ("T1T2", "T1"),
        {},
    ),
    "conditional": (
        {"Yw1": [], "Yw1T1": [1], "Yw1T1T2": [1, 2]},
        ("Yw1T1T2", "Yw1T1"),
        {"longitudinal": True},
    ),
    # a4b: the conditional contrast with follow-up-wave predictors
    # (load_master concurrent=True). Not named "concurrent" to avoid clashing
    # with the cross-sectional arm above.
    "conditional_followup": (
        {"Yw1": [], "Yw1T1": [1], "Yw1T1T2": [1, 2]},
        ("Yw1T1T2", "Yw1T1"),
        {"longitudinal": True, "concurrent": True},
    ),
}


def parse_args():
    p = argparse.ArgumentParser(description="A12: repeated-CV intervals for delta-R2")
    p.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    p.add_argument("--outcomes", nargs="+", default=["all"])
    p.add_argument("--repeats", type=int, default=15)
    p.add_argument("--repeat_start", type=int, default=0,
                   help="First repeat index this shard fits. The seed depends only on "
                        "the index, so any partition of the range gives the same cells.")
    p.add_argument("--repeat_end", type=int, default=None,
                   help="One past the last repeat index this shard fits (default: --repeats).")
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--no_gpu", action="store_true")
    p.add_argument("--tag", default="",
                   help="Shard identity; each shard writes its own repeats file.")
    p.add_argument("--fresh", action="store_true",
                   help="Discard this shard's existing repeats and refit from scratch.")
    p.add_argument("--merge_only", action="store_true",
                   help="Fit nothing. Combine every shard's repeats and write the summary.")
    p.add_argument("--shards", type=int, default=1,
                   help="Number of worker processes (1 = run in this process).")
    p.add_argument("--gpus", default="",
                   help="Comma-separated CUDA device ids for workers, round robin "
                        "(default: all devices).")
    p.add_argument("--threads", type=int, default=0,
                   help="OMP_NUM_THREADS per worker (default: cores / shards).")
    p.add_argument("--worker", action="store_true",
                   help="Internal: set by the driver on each shard. Writes metadata "
                        "under shards/<tag>/ and leaves the summary to the driver.")
    return p.parse_args()


def resolve(names: list[str]) -> list[str]:
    return PRIMARY if "all" in names else names


def run_arm(arm: str, outcome_w1: str, repeats: int, n_folds: int, use_gpu: bool,
            done: set[tuple], sink: Path, lo: int = 0, hi: int | None = None) -> None:
    """Fit every (tier, repeat) cell for one outcome.

    The design matrix is built once per tier; each row is appended to `sink` as
    it finishes, so an interrupted run can resume.
    """
    tier_configs, _, loader_kwargs = ARMS[arm]
    # Name the outcome by the wave the loader actually fits.
    outcome = (W1_TO_W2.get(outcome_w1, outcome_w1)
               if ARMS[arm][2].get("longitudinal") else outcome_w1)

    for tier_label, tiers in tier_configs.items():
        hi_ = repeats if hi is None else hi
        pending = [r for r in range(lo, hi_)
                   if (arm, outcome, r, tier_label) not in done]
        if not pending:
            print(f"  [{arm}] {outcome} | {tier_label}: nothing pending in this shard")
            continue

        t0 = time.time()
        X, y, features, age = load_master(
            outcome, tiers=tiers, age_residualize=False, **loader_kwargs)
        print(f"  [{arm}] {outcome} | {tier_label}: loaded n={int(y.notna().sum())} "
              f"features={len(features)} in {time.time() - t0:.1f}s; "
              f"{len(pending)} of repeats [{lo},{hi_}) to fit")

        for r in pending:
            seed = SEED_BASE + r
            t1 = time.time()
            pipe = XGBoostPipeline(n_folds=n_folds, use_gpu=use_gpu, random_state=seed)
            cv = pipe.run_cv(X, y, features, age=age)
            row = {
                "arm": arm, "outcome": outcome, "repeat": r, "seed": seed,
                "tier_config": tier_label, "r2_mean": cv["r2_mean"],
                "r2_std_folds": cv["r2_std"], "n": int(y.notna().sum()),
                "n_features": len(features), "fit_seconds": round(time.time() - t1, 1),
            }
            append_row(row, sink)
            print(f"  [{arm}] {outcome} | {tier_label} | repeat {r} of [{lo},{hi_}) "
                  f"(seed {seed}): r2={cv['r2_mean']:.4f} in {row['fit_seconds']}s",
                  flush=True)

        del X, y, age
        gc.collect()


def append_row(row: dict, sink: Path) -> None:
    """Append one row to the repeats file."""
    header = not sink.exists()
    pd.DataFrame([row]).to_csv(sink, mode="a", header=header, index=False)


def load_done(sink: Path) -> set[tuple]:
    """Cells already on disk, so a resumed run refits nothing it has."""
    if not sink.exists():
        return set()
    d = pd.read_csv(sink)
    return {(a, o, int(r), t) for a, o, r, t
            in zip(d["arm"], d["outcome"], d["repeat"], d["tier_config"])}


def summarise(long: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (arm, outcome), g in long.groupby(["arm", "outcome"], sort=False):
        _, (hi_cfg, lo_cfg), _ = ARMS[arm]
        wide = g.pivot(index="repeat", columns="tier_config", values="r2_mean")
        if hi_cfg not in wide or lo_cfg not in wide:
            continue
        d = (wide[hi_cfg] - wide[lo_cfg]).dropna()
        out.append({
            "arm": arm, "outcome": outcome,
            "delta": f"{hi_cfg} - {lo_cfg}",
            "n_repeats": len(d),
            "n": int(g["n"].max()),
            # Repeat 0 is the production seed, so this column is the number the
            # corresponding pipeline reports, recomputed here.
            "point_seed42": float(d.loc[0]) if 0 in d.index else np.nan,
            "mean": float(d.mean()), "sd": float(d.std(ddof=1)),
            "p2.5": float(d.quantile(0.025)), "median": float(d.median()),
            "p97.5": float(d.quantile(0.975)),
            "observed_min": float(d.min()), "observed_max": float(d.max()),
        })
    return pd.DataFrame(out)


def detect_gpus() -> list[str]:
    """Every CUDA device on the host, as ids fit for CUDA_VISIBLE_DEVICES."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout
        return [ln.strip() for ln in out.splitlines() if ln.strip().isdigit()]
    except Exception:
        return []


def shard_ranges(repeats: int, shards: int) -> list[tuple[int, int]]:
    """Contiguous repeat ranges, one per worker.

    Cell cost does not depend on the repeat index, so equal ranges balance the load.
    """
    base, extra = divmod(repeats, shards)
    ranges, lo = [], 0
    for i in range(shards):
        hi = lo + base + (1 if i < extra else 0)
        if hi > lo:
            ranges.append((lo, hi))
        lo = hi
    return ranges


def consolidate_age_ledger(tags: list[str]) -> None:
    """Merge the per-worker age ledgers into one file and renumber `seq`."""
    parts = []
    for t in tags:
        p = OUTPUT_DIR / "shards" / t.lstrip("_") / "age_adjustment_method.csv"
        if p.exists():
            parts.append(pd.read_csv(p))
    if not parts:
        return
    merged = pd.concat(parts, ignore_index=True)
    merged["seq"] = range(len(merged))
    merged.to_csv(OUTPUT_DIR / "age_adjustment_method.csv", index=False)
    print(f"age ledger: {len(merged)} load_master call(s) from {len(parts)} worker(s)")


def run_sharded(args) -> int:
    """Fan the repeats out across processes, then merge. Returns an exit code."""
    ranges = shard_ranges(args.repeats, args.shards)
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()] or detect_gpus()
    threads = args.threads or max(2, (os.cpu_count() or 8) // max(1, len(ranges)))
    shard_dir = OUTPUT_DIR / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    # Worker logs hold per-run timings, so they go under logs/ rather than the output directory.
    log_dir = ROOT / "logs" / "a12_shards"
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"A12 driver: {len(ranges)} worker(s), {args.repeats} repeats, "
          f"{threads} thread(s) each")
    print(f"gpus      : {gpus or ['<none; cpu>']}")
    print(f"ranges    : {ranges}", flush=True)

    procs, tags = [], []
    for i, (lo, hi) in enumerate(ranges):
        tag = f"_s{i:02d}"
        tags.append(tag)
        cmd = [sys.executable, str(Path(__file__).resolve()),
               "--arms", *args.arms, "--outcomes", *args.outcomes,
               "--repeats", str(args.repeats),
               "--repeat_start", str(lo), "--repeat_end", str(hi),
               "--n_folds", str(args.n_folds), "--tag", tag, "--worker"]
        if args.no_gpu:
            cmd.append("--no_gpu")
        if args.fresh:
            cmd.append("--fresh")
        env = dict(os.environ)
        env["OMP_NUM_THREADS"] = str(threads)
        if gpus:
            env["CUDA_VISIBLE_DEVICES"] = gpus[i % len(gpus)]
        log = log_dir / f"log{tag}.txt"
        fh = open(log, "wb")
        procs.append((tag, subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                            env=env, cwd=str(ROOT)), fh, log))
        print(f"  worker {tag}: repeats [{lo},{hi}) on "
              f"CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', '<unset>')} "
              f"-> {log.relative_to(ROOT)}", flush=True)

    t0 = time.time()
    failed = []
    for tag, proc, fh, log in procs:
        rc = proc.wait()
        fh.close()
        print(f"  worker {tag}: exit {rc} at {(time.time() - t0) / 60:.1f} min", flush=True)
        if rc != 0:
            failed.append((tag, rc, log))

    for tag, rc, log in failed:
        tail = ""
        try:
            tail = log.read_text(errors="replace")[-600:]
        except Exception:
            pass
        print(f"\nworker {tag} failed (exit {rc}); tail of {log.relative_to(ROOT)}:\n{tail}")

    consolidate_age_ledger(tags)
    # Stage-level provenance; completed is false if any worker failed.
    provenance.write_stamp(OUTPUT_DIR, "a12_delta_r2_ci",
                           ["data/processed/master_analysis.csv",
                            "data/processed/master_analysis_concurrent.csv"],
                           completed=not failed)
    write_summary()
    write_output_log(OUTPUT_DIR, "a12_output_log.md",
                     "analysis/pipelines/a12_delta_r2_ci.py",
                     [OUTPUT_DIR / "a12_delta_r2_ci.csv"], run_args=args)
    print(f"\ndriver finished in {(time.time() - t0) / 60:.1f} min; "
          f"{len(procs) - len(failed)} of {len(procs)} worker(s) succeeded")
    return 1 if failed else 0


def capture_env() -> dict:
    """Software, git and GPU details for this run."""
    def cmd(*a):
        try:
            return subprocess.run(a, capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:
            return "<unavailable>"
    import xgboost, sklearn
    return {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_sha": cmd("git", "rev-parse", "HEAD"),
        "git_branch": cmd("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(cmd("git", "status", "--porcelain")),
        "python": platform.python_version(),
        "xgboost": xgboost.__version__,
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        "gpus": cmd("nvidia-smi", "--query-gpu=index,name,memory.total",
                    "--format=csv,noheader"),
    }


def write_summary() -> pd.DataFrame:
    """Combine every shard's repeats and summarise. Safe to call at any time."""
    parts = sorted(OUTPUT_DIR.glob("a12_delta_r2_repeats*.csv"))
    if not parts:
        print("no repeats on disk; nothing to summarise")
        return pd.DataFrame()
    long = pd.concat([pd.read_csv(f) for f in parts], ignore_index=True)
    long = long.drop_duplicates(subset=["arm", "outcome", "repeat", "tier_config"],
                                keep="last")
    summary = summarise(long)
    summary.to_csv(OUTPUT_DIR / "a12_delta_r2_ci.csv", index=False)
    print(f"\nsummarised {len(long)} cells from {len(parts)} shard file(s)")
    if len(summary):
        print(summary.round(4).to_string(index=False))
    print("\nInterval is the spread across CV partitions of one fixed sample, "
          "not sampling uncertainty.")
    return summary


def main() -> None:
    args = parse_args()
    use_gpu = not args.no_gpu

    if args.shards > 1 and not args.merge_only:
        sys.exit(run_sharded(args))

    # Workers write metadata under their shard directory; the driver consolidates it.
    meta_dir = (OUTPUT_DIR / "shards" / args.tag.lstrip("_")) if args.worker else OUTPUT_DIR
    # A merge-only pass fits nothing, so it writes no age ledger (which would
    # overwrite the fitting run's ledger with a placeholder).
    provenance.stamp_at_exit(meta_dir, "a12_delta_r2_ci",
                             ["data/processed/master_analysis.csv",
                              "data/processed/master_analysis_concurrent.csv"])
    if not args.merge_only:
        age_method.emit_at_exit(meta_dir, "a12_delta_r2_ci")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.merge_only:
        write_summary()
        return

    sink = OUTPUT_DIR / f"a12_delta_r2_repeats{args.tag}.csv"
    if args.fresh:
        sink.unlink(missing_ok=True)
    done = load_done(sink)

    env = capture_env()
    (OUTPUT_DIR / f"a12_run_meta{args.tag}.json").write_text(
        json.dumps({"env": env, "args": vars(args), "seed_base": SEED_BASE}, indent=2) + "\n")

    outcomes = resolve(args.outcomes)
    total = sum(len(ARMS[a][0]) for a in args.arms) * len(outcomes) * args.repeats
    print(f"A12: {args.repeats} repeats x {args.n_folds}-fold CV, seeds "
          f"{SEED_BASE}..{SEED_BASE + args.repeats - 1}")
    print(f"arms      : {args.arms}")
    print(f"outcomes  : {outcomes}")
    print(f"shard     : {args.tag or '<none>'}   CUDA_VISIBLE_DEVICES="
          f"{env['cuda_visible_devices']}")
    print(f"cells     : {len(done)} already on disk, {total - len(done)} to fit")
    print(f"git       : {env['git_branch']} {env['git_sha'][:9]}"
          f"{' DIRTY' if env['git_dirty'] else ''}", flush=True)

    t0 = time.time()
    for arm in args.arms:
        for outcome_w1 in outcomes:
            run_arm(arm, outcome_w1, args.repeats, args.n_folds, use_gpu, done, sink,
                    args.repeat_start, args.repeat_end)
    print(f"\nshard {args.tag or '<none>'} finished in {(time.time() - t0) / 60:.1f} min")

    if args.worker:
        return  # the driver summarises once, over every worker's repeats
    write_summary()
    write_output_log(OUTPUT_DIR, "a12_output_log.md",
                     "analysis/pipelines/a12_delta_r2_ci.py",
                     [sink, OUTPUT_DIR / "a12_delta_r2_ci.csv"], run_args=args)


if __name__ == "__main__":
    main()

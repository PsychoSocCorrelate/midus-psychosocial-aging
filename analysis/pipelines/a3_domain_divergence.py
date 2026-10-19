#!/usr/bin/env python3
"""
A3: Formal Domain Divergence Test

Tests whether the predictor importance structure differs across aging domains
using bootstrap-based null distributions from CS-1 bootstrap SHAP iterations.

For each pair of aging domains, computes Spearman ρ between their bootstrap
mean |SHAP| vectors. The bootstrap null resamples within each domain's
bootstrap iterations to build a continuous null distribution.

Additional analyses (--layer flag):
  Layer 1:         Top-k SHAP ρ sensitivity (k=10,20,30,50,75,100,ALL) with subdomain ρ
  Layer 2:         Subdomain-level divergence test (standalone; included in default run)
  Layer 3:         Cross-prediction test (A4 longitudinal: Y2 top-20 used to predict the other Y2s)
  Layer 3cs:       Cross-prediction test (CS-1: Y1 top-20 used to predict the other Y1s)
  Layer long:      Longitudinal divergence (A4 SHAP ρ_signal + ρ_sd_signal, 3 domains)
  Layer decompose: Core-periphery cross-prediction decomposition (shared/unique features)
  Layer null:      Random-partition null for shared feature transferability
  Layer log:       Write the output log from existing artifacts

The default run (no --layer) includes CS-1 feature-level + subdomain-level divergence,
plus a decomposition table. For internally consistent decomposition tables:
  - CS-1 decomposition: default run + --layer 3cs (all CS-1 metrics)
  - A4 decomposition: --layer long + --layer 3 (all A4 metrics)

Usage:
    uv run python analysis/pipelines/a3_domain_divergence.py                  # CS-1 divergence
    uv run python analysis/pipelines/a3_domain_divergence.py --layer 1        # top-k sensitivity (A4)
    uv run python analysis/pipelines/a3_domain_divergence.py --layer 3        # cross-prediction (A4)
    uv run python analysis/pipelines/a3_domain_divergence.py --layer 3cs      # cross-prediction (CS-1)
    uv run python analysis/pipelines/a3_domain_divergence.py --layer long     # longitudinal divergence (A4)
    uv run python analysis/pipelines/a3_domain_divergence.py --layer all      # all layers
    uv run python analysis/pipelines/a3_domain_divergence.py --layer log      # output log only

Requires CS-1 outputs in output/cs1_cross_sectional/.
A4 layers require outputs in output/a4_longitudinal/.
Output: output/a3_domain_divergence/
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analysis.core.output_log import write_output_log
from analysis.core import age_method
from analysis.core import provenance
from analysis.core import artifact_paths as ap
CS1_DIR = ROOT / "output/cs1_cross_sectional"
OUTPUT_DIR = ROOT / "output/a3_domain_divergence"

# Domain mapping: domain label to CS-1 outcome file stem
# For FInoPsy, use the adjusted SHAP (A2 overlap-excluded)
DOMAINS = {
    "Frailty": "FInoPsy_W1",
    "Cognition": "btactComp_W1",
    "Allostatic Load": "ALcomp_W1",
    "Epigenetic Aging": "phenoAge_res",
}

# Longitudinal domain mapping: domain label to A4 outcome
# Only 3 domains (EpiAging has no longitudinal outcome)
DOMAINS_LONGITUDINAL = {
    "Frailty": "FInoPsy_W2",
    "Cognition": "btactComp_W2",
    "Allostatic Load": "ALcomp_W2",
}

N_PERMUTATIONS = 1000


def load_bootstrap_shap(
    outcome: str, cs1_dir: Path, use_adjusted: bool = False
) -> tuple[np.ndarray, list[str]]:
    """Load raw bootstrap SHAP matrix and feature names from CS-1.

    Returns:
        shap_raw: (n_bootstrap, n_features) array of mean |SHAP| per iteration
        features: list of feature names
    """
    suffix = "_adjusted" if use_adjusted else ""
    raw_path = cs1_dir / "shap_raw" / f"{outcome}_T1T2{suffix}_bootstrap_shap_raw.npy"
    feat_path = cs1_dir / "shap" / f"{outcome}_T1T2{suffix}_features.csv"

    if not raw_path.exists():
        raise FileNotFoundError(f"Missing: {raw_path}")
    if not feat_path.exists():
        raise FileNotFoundError(f"Missing: {feat_path}")

    shap_raw = np.load(raw_path)
    features = pd.read_csv(feat_path, header=None)[0].tolist()

    return shap_raw, features


def load_bootstrap_shap_longitudinal(
    outcome: str, a4_dir: Path, use_adjusted: bool = False
) -> tuple[np.ndarray, list[str]]:
    """Load raw bootstrap SHAP matrix and feature names from A4 longitudinal.

    Same format as CS-1 but with '_long_' naming convention.
    """
    suffix = "_adjusted" if use_adjusted else ""
    raw_path = a4_dir / "shap_raw" / f"{outcome}_long{suffix}_bootstrap_shap_raw.npy"
    feat_path = a4_dir / "shap" / f"{outcome}_long{suffix}_features.csv"

    if not raw_path.exists():
        raise FileNotFoundError(f"Missing: {raw_path}")
    if not feat_path.exists():
        raise FileNotFoundError(f"Missing: {feat_path}")

    shap_raw = np.load(raw_path)
    features = pd.read_csv(feat_path, header=None)[0].tolist()

    return shap_raw, features


def align_shap_vectors(
    domain_data: dict[str, tuple[np.ndarray, list[str]]]
) -> tuple[dict[str, np.ndarray], list[str]]:
    """Align all domain SHAP matrices to a common feature set.

    Features missing in a domain are filled with 0.0 (no importance).

    Returns:
        aligned: dict of domain to (n_bootstrap, n_common_features) arrays
        common_features: ordered list of all features across domains
    """
    # Union of all features
    all_features = []
    seen = set()
    for _, features in domain_data.values():
        for f in features:
            if f not in seen:
                all_features.append(f)
                seen.add(f)

    aligned = {}
    for domain, (shap_raw, features) in domain_data.items():
        n_boot = shap_raw.shape[0]
        aligned_matrix = np.zeros((n_boot, len(all_features)))
        feat_to_idx = {f: i for i, f in enumerate(features)}
        for j, f in enumerate(all_features):
            if f in feat_to_idx:
                aligned_matrix[:, j] = shap_raw[:, feat_to_idx[f]]
        aligned[domain] = aligned_matrix

    return aligned, all_features


def compute_pairwise_rho(
    mean_shap: dict[str, np.ndarray],
    domains: list[str],
) -> dict[tuple[str, str], float]:
    """Compute Spearman ρ for all domain pairs."""
    rho_dict = {}
    for d1, d2 in itertools.combinations(domains, 2):
        rho, _ = spearmanr(mean_shap[d1], mean_shap[d2])
        rho_dict[(d1, d2)] = float(rho)
    return rho_dict


def bootstrap_null(
    aligned: dict[str, np.ndarray],
    domains: list[str],
    n_perm: int = 1000,
    seed: int = 42,
) -> dict[tuple[str, str], np.ndarray]:
    """Build pooled-permutation null distribution for pairwise Spearman ρ.

    Tests: "If all domains shared the same importance structure, what ρ
    would we expect between their mean SHAP vectors?"

    Procedure (per iteration):
        1. Pool all bootstrap SHAP iterations across all domains
        2. Randomly assign pooled iterations to pseudo-domains (same n_boot each)
        3. Compute mean |SHAP| per pseudo-domain
        4. Compute all pairwise ρ values

    Under this null, ρ should be high (~1.0) because pseudo-domains
    draw from the same pool. If observed ρ is significantly lower,
    the domains are genuinely divergent.

    Returns dict of (domain1, domain2) to array of null ρ values.
    """
    rng = np.random.default_rng(seed)
    pairs = list(itertools.combinations(domains, 2))
    null_rho = {pair: np.zeros(n_perm) for pair in pairs}

    n_domains = len(domains)
    n_boot = next(iter(aligned.values())).shape[0]

    # Pool all iterations across domains
    pooled = np.concatenate([aligned[d] for d in domains], axis=0)  # (n_domains*n_boot, n_features)
    total = pooled.shape[0]

    for i in range(n_perm):
        if (i + 1) % 200 == 0:
            print(f"  Permutation {i+1}/{n_perm}")

        # Shuffle and split into pseudo-domains
        perm_idx = rng.permutation(total)
        pseudo_means = {}
        for d_i, domain in enumerate(domains):
            start = d_i * n_boot
            end = start + n_boot
            pseudo_means[domain] = pooled[perm_idx[start:end]].mean(axis=0)

        # Compute pairwise ρ
        for pair in pairs:
            rho, _ = spearmanr(pseudo_means[pair[0]], pseudo_means[pair[1]])
            null_rho[pair][i] = rho

    return null_rho


def holm_bonferroni(p_values: list[float]) -> list[float]:
    """Apply Holm-Bonferroni correction to a list of p-values."""
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * n

    for rank, (orig_idx, p) in enumerate(indexed):
        adj_p = min(1.0, p * (n - rank))
        # Enforce monotonicity
        if rank > 0:
            prev_idx = indexed[rank - 1][0]
            adj_p = max(adj_p, adjusted[prev_idx])
        adjusted[orig_idx] = adj_p

    return adjusted


# Layer 2: Subdomain-level divergence test + stability-filtered analysis

CONFIG_PATH = ROOT / "config/variable_config.yaml"

# Thresholds for signal-feature filtering (fraction of domain's total |SHAP|)
SIGNAL_THRESHOLDS = [0.0, 0.0025, 0.005, 0.01]
SIGNAL_LABELS = ["all", ">=0.25%", ">=0.5%", ">=1.0%"]


def load_subdomain_mapping() -> dict[str, list[str]]:
    """Load subdomain-to-feature-list mapping from variable_config.yaml."""
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    return {k: v for k, v in cfg["subdomain"].items() if v}


def _aggregate_to_subdomain(
    shap_vec: np.ndarray,
    common_features: list[str],
    subdomain_map: dict[str, list[str]],
    subdomain_names: list[str],
    signal_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Aggregate a feature-level SHAP vector to subdomain level.

    Args:
        shap_vec: 1-D array (n_features,) of mean |SHAP|
        signal_mask: optional boolean mask; if given, only signal features
            contribute to subdomain aggregation (mean over signal features
            within each subdomain). If None, all features contribute.

    Returns 1-D array (n_subdomains,).
    """
    feat_to_idx = {f: i for i, f in enumerate(common_features)}
    agg = np.zeros(len(subdomain_names))
    for si, sd_name in enumerate(subdomain_names):
        indices = [feat_to_idx[f] for f in subdomain_map[sd_name] if f in feat_to_idx]
        if not indices:
            continue
        vals = shap_vec[indices]
        if signal_mask is not None:
            sd_signal = signal_mask[indices]
            vals = vals[sd_signal]
        if len(vals) > 0:
            nonzero = vals > 0
            if nonzero.any():
                agg[si] = vals[nonzero].mean()
    return agg


def compute_signal_mask(
    mean_shap_d1: np.ndarray,
    mean_shap_d2: np.ndarray,
    total_d1: float,
    total_d2: float,
    threshold: float,
) -> np.ndarray:
    """Return boolean mask: True for features carrying >= threshold fraction
    of total |SHAP| in either domain of the pair."""
    if threshold <= 0:
        return np.ones(len(mean_shap_d1), dtype=bool)
    pct1 = mean_shap_d1 / total_d1 if total_d1 > 0 else mean_shap_d1
    pct2 = mean_shap_d2 / total_d2 if total_d2 > 0 else mean_shap_d2
    return (pct1 >= threshold) | (pct2 >= threshold)


def run_subdomain_divergence(
    aligned: dict[str, np.ndarray],
    common_features: list[str],
    domains: list[str],
    observed_feature_rho: dict[tuple[str, str], float],
    n_perm: int = 1000,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Layer 2: Subdomain-level divergence + stability-filtered analysis.

    Two diagnostics addressing distinct limitations:

    A) Subdomain aggregation: tests whether divergence persists at the
       informational level (14 psychosocial subdomains) or collapses when
       correlated features are grouped.

    B) Stability filtering: tests whether divergence holds among features
       with reliable SHAP signal, or is inflated by noise-floor agreement
       (many features near 0 in both domains inflate ρ through tied ranks).

    Both diagnostics are applied independently and jointly (filtered
    subdomain ρ) to give a complete picture.

    Returns:
        stability_df: ρ at multiple signal thresholds (feature + subdomain)
        subdomain_df: subdomain-level pairwise ρ with permutation null
            (at ≥0.5% signal filter)
        decomposition_df: combined summary
    """
    print("\nLAYER 2: SUBDOMAIN DIVERGENCE + STABILITY FILTER")

    subdomain_map = load_subdomain_mapping()
    subdomain_names = sorted(subdomain_map.keys())
    print(f"  Subdomains: {len(subdomain_names)}")

    mean_shap = {d: aligned[d].mean(axis=0) for d in domains}
    totals = {d: mean_shap[d].sum() for d in domains}
    pairs = list(itertools.combinations(domains, 2))

    # Part A: Stability-filtered ρ at multiple thresholds
    print("\n  --- Signal-Filtered Divergence ---")
    print(f"  {'Pair':<35s} {'Filter':<12s} {'N':>5s} {'ρ_feat':>8s} {'ρ_sub':>8s}")
    print("  " + "-" * 72)

    stability_rows = []
    for d1, d2 in pairs:
        for thresh, label in zip(SIGNAL_THRESHOLDS, SIGNAL_LABELS):
            mask = compute_signal_mask(mean_shap[d1], mean_shap[d2], totals[d1], totals[d2], thresh)
            n_feat = int(mask.sum())

            # Feature-level ρ on filtered set
            if n_feat >= 5:
                rho_f, p_f = spearmanr(mean_shap[d1][mask], mean_shap[d2][mask])
            else:
                rho_f, p_f = float("nan"), float("nan")

            # Subdomain-level ρ on filtered set
            sd1 = _aggregate_to_subdomain(mean_shap[d1], common_features, subdomain_map, subdomain_names, mask)
            sd2 = _aggregate_to_subdomain(mean_shap[d2], common_features, subdomain_map, subdomain_names, mask)
            active = (sd1 > 0) | (sd2 > 0)
            if active.sum() >= 3:
                rho_s, p_s = spearmanr(sd1[active], sd2[active])
                n_sd = int(active.sum())
            else:
                rho_s, p_s, n_sd = float("nan"), float("nan"), 0

            stability_rows.append({
                "domain_1": d1, "domain_2": d2,
                "filter": label, "n_features": n_feat,
                "rho_feature": float(rho_f), "p_feature": float(p_f),
                "rho_subdomain": float(rho_s), "p_subdomain": float(p_s),
                "n_subdomains": n_sd,
            })

            rf_str = f"{rho_f:>8.4f}" if not np.isnan(rho_f) else "     N/A"
            rs_str = f"{rho_s:>8.4f}" if not np.isnan(rho_s) else "     N/A"
            print(f"  {d1} vs {d2:<18s} {label:<12s} {n_feat:>5d} {rf_str} {rs_str}")
        print()

    stability_df = pd.DataFrame(stability_rows)
    stability_df.to_csv(OUTPUT_DIR / "a3_stability_filtered.csv", index=False)
    print(f"  Saved: a3_stability_filtered.csv")

    # Part B: Permutation null at subdomain level (signal-filtered)
    # Use ≥0.5% threshold as the primary signal filter for null testing
    PRIMARY_THRESH = 0.005
    print(f"\n  --- Subdomain Null Distribution (signal ≥0.5% filter) ---")
    print(f"  Building permutation null ({n_perm} iterations)...")

    rng = np.random.default_rng(seed)
    feat_to_idx = {f: i for i, f in enumerate(common_features)}

    # Pre-compute signal masks per pair
    pair_masks = {}
    for d1, d2 in pairs:
        pair_masks[(d1, d2)] = compute_signal_mask(
            mean_shap[d1], mean_shap[d2], totals[d1], totals[d2], PRIMARY_THRESH
        )

    # Observed subdomain ρ at primary threshold
    observed_sd_rho = {}
    for d1, d2 in pairs:
        mask = pair_masks[(d1, d2)]
        sd1 = _aggregate_to_subdomain(mean_shap[d1], common_features, subdomain_map, subdomain_names, mask)
        sd2 = _aggregate_to_subdomain(mean_shap[d2], common_features, subdomain_map, subdomain_names, mask)
        active = (sd1 > 0) | (sd2 > 0)
        if active.sum() >= 3:
            rho, _ = spearmanr(sd1[active], sd2[active])
        else:
            rho = float("nan")
        observed_sd_rho[(d1, d2)] = float(rho)

    # Permutation null
    n_boot = next(iter(aligned.values())).shape[0]
    pooled = np.concatenate([aligned[d] for d in domains], axis=0)
    total_rows = pooled.shape[0]
    null_sd_rho = {pair: np.zeros(n_perm) for pair in pairs}

    for i in range(n_perm):
        if (i + 1) % 200 == 0:
            print(f"    Permutation {i+1}/{n_perm}")

        perm_idx = rng.permutation(total_rows)
        pseudo_means = {}
        pseudo_totals = {}
        for d_i, domain in enumerate(domains):
            start = d_i * n_boot
            end = start + n_boot
            pseudo_means[domain] = pooled[perm_idx[start:end]].mean(axis=0)
            pseudo_totals[domain] = pseudo_means[domain].sum()

        for d1, d2 in pairs:
            # Recompute signal mask for pseudo-domains
            pmask = compute_signal_mask(
                pseudo_means[d1], pseudo_means[d2],
                pseudo_totals[d1], pseudo_totals[d2], PRIMARY_THRESH
            )
            sd1 = _aggregate_to_subdomain(pseudo_means[d1], common_features, subdomain_map, subdomain_names, pmask)
            sd2 = _aggregate_to_subdomain(pseudo_means[d2], common_features, subdomain_map, subdomain_names, pmask)
            active = (sd1 > 0) | (sd2 > 0)
            if active.sum() >= 3:
                rho, _ = spearmanr(sd1[active], sd2[active])
            else:
                rho = float("nan")
            null_sd_rho[(d1, d2)][i] = rho

    # P-values and correction
    p_values = []
    for pair in pairs:
        null = null_sd_rho[pair]
        obs = observed_sd_rho[pair]
        if np.isnan(obs):
            p_values.append(float("nan"))
        else:
            valid_null = null[~np.isnan(null)]
            p = float(np.mean(valid_null <= obs)) if len(valid_null) > 0 else float("nan")
            p_values.append(p)
    p_adjusted = holm_bonferroni([p if not np.isnan(p) else 1.0 for p in p_values])

    # Build subdomain results table
    sd_rows = []
    for i, (d1, d2) in enumerate(pairs):
        null = null_sd_rho[(d1, d2)]
        valid_null = null[~np.isnan(null)]
        sd_rows.append({
            "domain_1": d1,
            "domain_2": d2,
            "signal_filter": ">=0.5%",
            "observed_rho_subdomain": observed_sd_rho[(d1, d2)],
            "null_mean": float(valid_null.mean()) if len(valid_null) > 0 else float("nan"),
            "null_sd": float(valid_null.std()) if len(valid_null) > 0 else float("nan"),
            "p_value": p_values[i],
            "p_adjusted": p_adjusted[i],
            "significant": p_adjusted[i] < 0.05 if not np.isnan(p_adjusted[i]) else False,
        })
    subdomain_df = pd.DataFrame(sd_rows)
    subdomain_df.to_csv(OUTPUT_DIR / "a3_subdomain_divergence.csv", index=False)
    print(f"\n  Saved: a3_subdomain_divergence.csv")

    # Print subdomain results
    print(f"\n  Subdomain ρ (signal-filtered, ≥0.5%):")
    for _, row in subdomain_df.iterrows():
        sig = " ***" if row["significant"] else ""
        obs = row["observed_rho_subdomain"]
        obs_str = f"{obs:.4f}" if not np.isnan(obs) else "N/A"
        print(
            f"    {row['domain_1']:20s} vs {row['domain_2']:20s}: "
            f"ρ = {obs_str}  (null = {row['null_mean']:.4f} ± {row['null_sd']:.4f}, "
            f"p_adj = {row['p_adjusted']:.4f}){sig}"
        )

    # Part C: Decomposition table
    # Load CS-1 cross-prediction R² ratios if available (internal consistency)
    cross_pred_path = OUTPUT_DIR / "a3_cross_prediction_cs1.csv"
    cross_r2_map = {}
    if cross_pred_path.exists():
        cp_df = pd.read_csv(cross_pred_path)
        domain_to_outcome = {v: k for k, v in DOMAINS.items()}
        for _, row in cp_df.iterrows():
            src_d = domain_to_outcome.get(row["source_outcome"])
            tgt_d = domain_to_outcome.get(row["target_outcome"])
            if src_d and tgt_d:
                pair = tuple(sorted([src_d, tgt_d]))
                cross_r2_map.setdefault(pair, []).append(row["r2_ratio"])

    # Get filtered and unfiltered ρ from stability rows
    filtered_feat_rho = {}
    unfiltered_sd_rho = {}
    for row_dict in stability_rows:
        pair_key = (row_dict["domain_1"], row_dict["domain_2"])
        if row_dict["filter"] == ">=0.5%":
            filtered_feat_rho[pair_key] = row_dict["rho_feature"]
        if row_dict["filter"] == "all":
            unfiltered_sd_rho[pair_key] = row_dict["rho_subdomain"]

    decomp_rows = []
    for d1, d2 in pairs:
        rho_feat_all = observed_feature_rho[(d1, d2)]
        rho_feat_filt = filtered_feat_rho.get((d1, d2), float("nan"))
        rho_sd_all = unfiltered_sd_rho.get((d1, d2), float("nan"))
        rho_sd_filt = observed_sd_rho[(d1, d2)]

        pair_key = tuple(sorted([d1, d2]))
        cross_r2 = float(np.mean(cross_r2_map[pair_key])) if pair_key in cross_r2_map else float("nan")

        decomp_rows.append({
            "domain_1": d1,
            "domain_2": d2,
            "rho_feature_all": rho_feat_all,
            "rho_feature_signal": rho_feat_filt,
            "rho_subdomain_all": rho_sd_all,
            "rho_subdomain_signal": rho_sd_filt,
            "cross_r2_ratio": cross_r2,
        })
    decomposition_df = pd.DataFrame(decomp_rows)
    decomposition_df.to_csv(OUTPUT_DIR / "a3_divergence_decomposition.csv", index=False)
    print(f"\n  Saved: a3_divergence_decomposition.csv")

    # Print decomposition
    print("\n  Divergence Decomposition:")
    print(f"  {'Pair':<35s} {'ρ_all':>7s} {'ρ_sig':>7s} {'ρ_sd_all':>8s} {'ρ_sd_sig':>8s} {'Cross R²':>9s}")
    print("  " + "-" * 80)
    for _, row in decomposition_df.iterrows():
        rs = f"{row['rho_feature_signal']:>7.4f}" if not np.isnan(row["rho_feature_signal"]) else "    N/A"
        rsa = f"{row['rho_subdomain_all']:>8.4f}" if not np.isnan(row["rho_subdomain_all"]) else "     N/A"
        rsd = f"{row['rho_subdomain_signal']:>8.4f}" if not np.isnan(row["rho_subdomain_signal"]) else "     N/A"
        cr = f"{row['cross_r2_ratio']:.3f}" if not np.isnan(row["cross_r2_ratio"]) else "N/A"
        print(
            f"  {row['domain_1']} vs {row['domain_2']:<20s} "
            f"{row['rho_feature_all']:>7.4f} {rs} {rsa} {rsd} {cr:>9s}"
        )

    return stability_df, subdomain_df, decomposition_df


# Longitudinal divergence: A4 ρ_signal and ρ_sd_signal


def run_longitudinal_divergence(
    n_perm: int = 1000,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Longitudinal divergence: same analysis as CS-1 subdomain divergence
    but using A4 longitudinal bootstrap SHAP (3 domains, no EpiAging).

    Returns:
        stability_df, subdomain_df, decomposition_df
        (same structure as run_subdomain_divergence outputs, prefixed with 'long_')
    """
    print("\nLONGITUDINAL DIVERGENCE: A4 SHAP (3 domains)")

    # Load A4 bootstrap SHAP
    domain_data = {}
    for domain, outcome in DOMAINS_LONGITUDINAL.items():
        use_adjusted = (outcome == "FInoPsy_W2")
        try:
            shap_raw, features = load_bootstrap_shap_longitudinal(
                outcome, A4_DIR, use_adjusted=use_adjusted
            )
            domain_data[domain] = (shap_raw, features)
            print(f"  Loaded {domain} ({outcome}): {shap_raw.shape}")
        except FileNotFoundError as e:
            print(f"  WARNING: {e}")

    if len(domain_data) < 2:
        print("ERROR: Need at least 2 domains with A4 SHAP data.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    domains = list(domain_data.keys())
    aligned, common_features = align_shap_vectors(domain_data)
    print(f"  Common feature space: {len(common_features)} variables")
    print(f"  Domains: {domains}")

    mean_shap = {d: aligned[d].mean(axis=0) for d in domains}
    totals = {d: mean_shap[d].sum() for d in domains}
    pairs = list(itertools.combinations(domains, 2))

    # Feature-level ρ (all features)
    observed_rho = compute_pairwise_rho(mean_shap, domains)
    print("\n  Observed pairwise Spearman ρ (all features):")
    for (d1, d2), rho in observed_rho.items():
        print(f"    {d1} vs {d2}: ρ = {rho:.4f}")

    # Stability-filtered ρ
    subdomain_map = load_subdomain_mapping()
    subdomain_names = sorted(subdomain_map.keys())
    print(f"  Subdomains: {len(subdomain_names)}")

    print(f"\n  {'Pair':<35s} {'Filter':<12s} {'N':>5s} {'ρ_feat':>8s} {'ρ_sub':>8s}")
    print("  " + "-" * 72)

    stability_rows = []
    for d1, d2 in pairs:
        for thresh, label in zip(SIGNAL_THRESHOLDS, SIGNAL_LABELS):
            mask = compute_signal_mask(mean_shap[d1], mean_shap[d2], totals[d1], totals[d2], thresh)
            n_feat = int(mask.sum())

            if n_feat >= 5:
                rho_f, p_f = spearmanr(mean_shap[d1][mask], mean_shap[d2][mask])
            else:
                rho_f, p_f = float("nan"), float("nan")

            sd1 = _aggregate_to_subdomain(mean_shap[d1], common_features, subdomain_map, subdomain_names, mask)
            sd2 = _aggregate_to_subdomain(mean_shap[d2], common_features, subdomain_map, subdomain_names, mask)
            active = (sd1 > 0) | (sd2 > 0)
            if active.sum() >= 3:
                rho_s, p_s = spearmanr(sd1[active], sd2[active])
                n_sd = int(active.sum())
            else:
                rho_s, p_s, n_sd = float("nan"), float("nan"), 0

            stability_rows.append({
                "domain_1": d1, "domain_2": d2,
                "filter": label, "n_features": n_feat,
                "rho_feature": float(rho_f), "p_feature": float(p_f),
                "rho_subdomain": float(rho_s), "p_subdomain": float(p_s),
                "n_subdomains": n_sd,
            })

            rf_str = f"{rho_f:>8.4f}" if not np.isnan(rho_f) else "     N/A"
            rs_str = f"{rho_s:>8.4f}" if not np.isnan(rho_s) else "     N/A"
            print(f"  {d1} vs {d2:<18s} {label:<12s} {n_feat:>5d} {rf_str} {rs_str}")
        print()

    stability_df = pd.DataFrame(stability_rows)
    stability_df.to_csv(OUTPUT_DIR / "a3_longitudinal_stability.csv", index=False)
    print(f"  Saved: a3_longitudinal_stability.csv")

    # Permutation null at subdomain level (signal-filtered)
    PRIMARY_THRESH = 0.005
    print(f"\n  --- Subdomain Null Distribution (signal ≥0.5% filter) ---")
    print(f"  Building permutation null ({n_perm} iterations)...")

    rng = np.random.default_rng(seed)

    pair_masks = {}
    for d1, d2 in pairs:
        pair_masks[(d1, d2)] = compute_signal_mask(
            mean_shap[d1], mean_shap[d2], totals[d1], totals[d2], PRIMARY_THRESH
        )

    observed_sd_rho = {}
    for d1, d2 in pairs:
        mask = pair_masks[(d1, d2)]
        sd1 = _aggregate_to_subdomain(mean_shap[d1], common_features, subdomain_map, subdomain_names, mask)
        sd2 = _aggregate_to_subdomain(mean_shap[d2], common_features, subdomain_map, subdomain_names, mask)
        active = (sd1 > 0) | (sd2 > 0)
        if active.sum() >= 3:
            rho, _ = spearmanr(sd1[active], sd2[active])
        else:
            rho = float("nan")
        observed_sd_rho[(d1, d2)] = float(rho)

    n_boot = next(iter(aligned.values())).shape[0]
    pooled = np.concatenate([aligned[d] for d in domains], axis=0)
    total_rows = pooled.shape[0]
    null_sd_rho = {pair: np.zeros(n_perm) for pair in pairs}

    for i in range(n_perm):
        if (i + 1) % 200 == 0:
            print(f"    Permutation {i+1}/{n_perm}")

        perm_idx = rng.permutation(total_rows)
        pseudo_means = {}
        pseudo_totals = {}
        for d_i, domain in enumerate(domains):
            start = d_i * n_boot
            end = start + n_boot
            pseudo_means[domain] = pooled[perm_idx[start:end]].mean(axis=0)
            pseudo_totals[domain] = pseudo_means[domain].sum()

        for d1, d2 in pairs:
            pmask = compute_signal_mask(
                pseudo_means[d1], pseudo_means[d2],
                pseudo_totals[d1], pseudo_totals[d2], PRIMARY_THRESH
            )
            sd1 = _aggregate_to_subdomain(pseudo_means[d1], common_features, subdomain_map, subdomain_names, pmask)
            sd2 = _aggregate_to_subdomain(pseudo_means[d2], common_features, subdomain_map, subdomain_names, pmask)
            active = (sd1 > 0) | (sd2 > 0)
            if active.sum() >= 3:
                rho, _ = spearmanr(sd1[active], sd2[active])
            else:
                rho = float("nan")
            null_sd_rho[(d1, d2)][i] = rho

    p_values = []
    for pair in pairs:
        null = null_sd_rho[pair]
        obs = observed_sd_rho[pair]
        if np.isnan(obs):
            p_values.append(float("nan"))
        else:
            valid_null = null[~np.isnan(null)]
            p = float(np.mean(valid_null <= obs)) if len(valid_null) > 0 else float("nan")
            p_values.append(p)
    p_adjusted = holm_bonferroni([p if not np.isnan(p) else 1.0 for p in p_values])

    sd_rows = []
    for i, (d1, d2) in enumerate(pairs):
        null = null_sd_rho[(d1, d2)]
        valid_null = null[~np.isnan(null)]
        sd_rows.append({
            "domain_1": d1, "domain_2": d2,
            "signal_filter": ">=0.5%",
            "observed_rho_subdomain": observed_sd_rho[(d1, d2)],
            "null_mean": float(valid_null.mean()) if len(valid_null) > 0 else float("nan"),
            "null_sd": float(valid_null.std()) if len(valid_null) > 0 else float("nan"),
            "p_value": p_values[i],
            "p_adjusted": p_adjusted[i],
            "significant": p_adjusted[i] < 0.05 if not np.isnan(p_adjusted[i]) else False,
        })
    subdomain_df = pd.DataFrame(sd_rows)
    subdomain_df.to_csv(OUTPUT_DIR / "a3_longitudinal_subdomain.csv", index=False)
    print(f"\n  Saved: a3_longitudinal_subdomain.csv")

    for _, row in subdomain_df.iterrows():
        sig = " ***" if row["significant"] else ""
        obs = row["observed_rho_subdomain"]
        obs_str = f"{obs:.4f}" if not np.isnan(obs) else "N/A"
        print(
            f"    {row['domain_1']:20s} vs {row['domain_2']:20s}: "
            f"ρ = {obs_str}  (null = {row['null_mean']:.4f} ± {row['null_sd']:.4f}, "
            f"p_adj = {row['p_adjusted']:.4f}){sig}"
        )

    # Decomposition table (A4-consistent)
    # Read A4 cross-ΔR² from existing cross-prediction output
    cross_pred_path = OUTPUT_DIR / "a3_cross_prediction.csv"
    cross_delta_map = {}
    if cross_pred_path.exists():
        cp_df = pd.read_csv(cross_pred_path)
        if "delta_r2_ratio" not in cp_df.columns:
            print("  WARNING: a3_cross_prediction.csv missing delta_r2_ratio. Re-run --layer 3.")
        else:
            domain_to_outcome = {v: k for k, v in DOMAINS_LONGITUDINAL.items()}
            for _, row in cp_df.iterrows():
                src_d = domain_to_outcome.get(row["source_outcome"])
                tgt_d = domain_to_outcome.get(row["target_outcome"])
                if src_d and tgt_d:
                    pair = tuple(sorted([src_d, tgt_d]))
                    cross_delta_map.setdefault(pair, []).append(row["delta_r2_ratio"])

    filtered_feat_rho = {}
    for row_dict in stability_rows:
        if row_dict["filter"] == ">=0.5%":
            filtered_feat_rho[(row_dict["domain_1"], row_dict["domain_2"])] = row_dict["rho_feature"]

    decomp_rows = []
    for d1, d2 in pairs:
        rho_feat_all = observed_rho[(d1, d2)]
        rho_feat_filt = filtered_feat_rho.get((d1, d2), float("nan"))
        rho_sd_filt = observed_sd_rho[(d1, d2)]
        pair_key = tuple(sorted([d1, d2]))
        cross_delta = float(np.mean(cross_delta_map[pair_key])) if pair_key in cross_delta_map else float("nan")

        decomp_rows.append({
            "domain_1": d1, "domain_2": d2,
            "rho_feature_all": rho_feat_all,
            "rho_feature_signal": rho_feat_filt,
            "rho_subdomain_signal": rho_sd_filt,
            "cross_delta_r2_ratio": cross_delta,
        })
    decomposition_df = pd.DataFrame(decomp_rows)
    decomposition_df.to_csv(OUTPUT_DIR / "a3_longitudinal_decomposition.csv", index=False)
    print(f"\n  Saved: a3_longitudinal_decomposition.csv")

    print("\n  Longitudinal Divergence Decomposition:")
    print(f"  {'Pair':<35s} {'ρ_all':>7s} {'ρ_sig':>7s} {'ρ_sd':>7s} {'Cross ΔR²':>10s}")
    print("  " + "-" * 72)
    for _, row in decomposition_df.iterrows():
        rs = f"{row['rho_feature_signal']:>7.4f}" if not np.isnan(row["rho_feature_signal"]) else "    N/A"
        rsd = f"{row['rho_subdomain_signal']:>7.4f}" if not np.isnan(row["rho_subdomain_signal"]) else "    N/A"
        cr = f"{row['cross_delta_r2_ratio']:.3f}" if not np.isnan(row["cross_delta_r2_ratio"]) else "N/A"
        print(
            f"  {row['domain_1']} vs {row['domain_2']:<20s} "
            f"{row['rho_feature_all']:>7.4f} {rs} {rsd} {cr:>10s}"
        )

    return stability_df, subdomain_df, decomposition_df


# Layer 1: Top-k SHAP ρ sensitivity

TOPK_VALUES = [10, 20, 30, 50, 75, 100]  # "ALL" handled separately

# Domain mapping for longitudinal SHAP files (A4 output)
# Derived from DOMAINS_LONGITUDINAL.
LONGITUDINAL_DOMAINS = {v: v for v in DOMAINS_LONGITUDINAL.values()}

A4_DIR = ROOT / "output/a4_longitudinal"


def _load_longitudinal_shap(outcome: str) -> pd.Series:
    """Load mean |SHAP| from A4 longitudinal bootstrap SHAP CSV.

    Returns a Series indexed by variable name with mean_abs_shap values.
    Tries *_adjusted_* first (for FInoPsy), falls back to non-adjusted.
    """
    adj_path = ap.A4_DIR / "shap" / f"{outcome}_long_adjusted_bootstrap_shap.csv"
    plain_path = ap.A4_DIR / "shap" / f"{outcome}_long_bootstrap_shap.csv"

    if adj_path.exists():
        path = adj_path
    elif plain_path.exists():
        path = plain_path
    else:
        raise FileNotFoundError(
            f"No SHAP CSV found for {outcome} in {ap.A4_DIR / 'shap'}\n"
            f"  Tried: {adj_path.name}, {plain_path.name}"
        )

    df = pd.read_csv(path, index_col=0)
    return df["mean_abs_shap"]


def run_topk_sensitivity() -> pd.DataFrame:
    """Layer 1: Compute pairwise Spearman ρ at multiple top-k cutoffs.

    For each pair of longitudinal outcomes, ranks variables by average SHAP
    across both outcomes in the pair, takes top-k, and computes ρ on those
    k variables only. Also computes subdomain-level ρ at each k for
    comparison (to distinguish SHAP redistribution from genuine divergence).

    Returns DataFrame with columns: domain_1, domain_2, k, rho, rho_subdomain
    """
    print("\nLAYER 1: TOP-K SHAP RHO SENSITIVITY")

    # Load mean |SHAP| vectors for all longitudinal outcomes
    shap_vectors: dict[str, pd.Series] = {}
    for label, outcome in LONGITUDINAL_DOMAINS.items():
        try:
            sv = _load_longitudinal_shap(outcome)
            shap_vectors[label] = sv
            print(f"  Loaded {label}: {len(sv)} features")
        except FileNotFoundError as e:
            print(f"  WARNING: {e}")

    if len(shap_vectors) < 2:
        print("ERROR: Need at least 2 outcomes. Run A4 first.")
        return pd.DataFrame()

    # Align to common feature set (union, fill missing with 0)
    all_features = sorted(
        set().union(*(sv.index.tolist() for sv in shap_vectors.values()))
    )
    aligned: dict[str, np.ndarray] = {}
    for label, sv in shap_vectors.items():
        aligned[label] = np.array([sv.get(f, 0.0) for f in all_features])

    n_features = len(all_features)
    feature_arr = np.array(all_features)
    outcomes = list(aligned.keys())
    pairs = list(itertools.combinations(outcomes, 2))

    print(f"  Common feature space: {n_features} variables")
    print(f"  Pairs: {len(pairs)}")

    # Load subdomain mapping for subdomain-level ρ
    subdomain_map = load_subdomain_mapping()
    subdomain_names = sorted(subdomain_map.keys())
    feat_to_idx = {f: i for i, f in enumerate(all_features)}

    def _to_subdomain_vec(shap_vec: np.ndarray) -> np.ndarray:
        """Aggregate a feature-level SHAP vector to subdomain level."""
        agg = np.zeros(len(subdomain_names))
        for si, sd_name in enumerate(subdomain_names):
            indices = [feat_to_idx[f] for f in subdomain_map[sd_name] if f in feat_to_idx]
            if indices:
                vals = shap_vec[indices]
                nonzero = vals > 0
                if nonzero.any():
                    agg[si] = vals[nonzero].mean()
        return agg

    rows = []
    for d1, d2 in pairs:
        v1 = aligned[d1]
        v2 = aligned[d2]
        # Average SHAP across both outcomes for ranking
        avg_shap = (v1 + v2) / 2.0
        rank_order = np.argsort(avg_shap)[::-1]

        for k in TOPK_VALUES + [n_features]:
            k_label = k if k < n_features else "ALL"
            top_idx = rank_order[:k]

            # Feature-level ρ
            rho, _ = spearmanr(v1[top_idx], v2[top_idx])

            # Subdomain-level ρ: zero out features outside top-k, then aggregate
            v1_topk = np.zeros_like(v1)
            v2_topk = np.zeros_like(v2)
            v1_topk[top_idx] = v1[top_idx]
            v2_topk[top_idx] = v2[top_idx]
            sd1 = _to_subdomain_vec(v1_topk)
            sd2 = _to_subdomain_vec(v2_topk)
            # Only compute ρ on subdomains with non-zero values in at least one domain
            active = (sd1 > 0) | (sd2 > 0)
            if active.sum() >= 3:
                rho_sd, _ = spearmanr(sd1[active], sd2[active])
            else:
                rho_sd = float("nan")

            rows.append({
                "domain_1": d1,
                "domain_2": d2,
                "k": k_label,
                "rho": float(rho),
                "rho_subdomain": float(rho_sd),
            })

    result_df = pd.DataFrame(rows)
    out_path = OUTPUT_DIR / "a3_topk_sensitivity.csv"
    result_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    # Print results
    print("\n  Top-k Sensitivity Results:")
    print(f"  {'Pair':<35s} {'k':>5s}  {'ρ_feat':>7s}  {'ρ_sub':>7s}")
    print("  " + "-" * 60)
    for _, row in result_df.iterrows():
        rho_sd_str = f"{row['rho_subdomain']:>7.4f}" if not np.isnan(row["rho_subdomain"]) else "    N/A"
        print(
            f"  {row['domain_1']} vs {row['domain_2']:<15s} "
            f"{str(row['k']):>5s}  {row['rho']:>7.4f}  {rho_sd_str}"
        )

    return result_df


# Layer 3: Cross-prediction test

CROSS_PRED_OUTCOMES = list(DOMAINS_LONGITUDINAL.values())


def _get_top20_features(outcome: str) -> list[str]:
    """Get top-20 SHAP predictors for a longitudinal outcome from A4 output."""
    sv = _load_longitudinal_shap(outcome)
    return sv.sort_values(ascending=False).head(20).index.tolist()


def _get_own_r2(outcome: str) -> float:
    """Get the full-model pooled Yw1T1T2 R² for an outcome from a4_r2_table.csv."""
    r2_path = A4_DIR / "a4_r2_table.csv"
    if not r2_path.exists():
        raise FileNotFoundError(f"Missing: {r2_path}")
    df = pd.read_csv(r2_path)
    row = df[(df["outcome"] == outcome) & (df["cohort"] == "pooled") & (df["tier_config"] == "Yw1T1T2")]
    if row.empty:
        raise ValueError(f"No pooled Yw1T1T2 R² found for {outcome} in {r2_path}")
    return float(row["r2_mean"].iloc[0])


def _get_ar_baseline_r2(outcome: str) -> float:
    """Get the AR-only (Yw1) pooled R² for an outcome from a4_r2_table.csv."""
    r2_path = A4_DIR / "a4_r2_table.csv"
    if not r2_path.exists():
        raise FileNotFoundError(f"Missing: {r2_path}")
    df = pd.read_csv(r2_path)
    row = df[(df["outcome"] == outcome) & (df["cohort"] == "pooled") & (df["tier_config"] == "Yw1")]
    if row.empty:
        raise ValueError(f"No pooled Yw1 R² found for {outcome} in {r2_path}")
    return float(row["r2_mean"].iloc[0])


def run_cross_prediction() -> pd.DataFrame:
    """Layer 3: Cross-prediction test.

    For each pair of longitudinal outcomes:
      1. Take source outcome's top-20 SHAP features
      2. Train XGBoost 5-fold CV predicting the target outcome with those features
         (+ target's autoregressive W1 term if present)
      3. Compare to target's own full-model R²

    Returns DataFrame with cross-prediction results.
    """
    from analysis.core.data_loader import load_master
    from analysis.core.xgboost_pipeline import XGBoostPipeline

    print("\nLAYER 3: CROSS-PREDICTION TEST")

    pipeline = XGBoostPipeline(n_folds=5, n_bootstrap=0, use_gpu=True)

    # Pre-load top-20 features, own-R², and AR baselines for each outcome
    top20: dict[str, list[str]] = {}
    own_r2: dict[str, float] = {}
    ar_baselines: dict[str, float] = {}
    for outcome in CROSS_PRED_OUTCOMES:
        try:
            top20[outcome] = _get_top20_features(outcome)
            own_r2[outcome] = _get_own_r2(outcome)
            ar_baselines[outcome] = _get_ar_baseline_r2(outcome)
            own_delta = own_r2[outcome] - ar_baselines[outcome]
            print(f"  {outcome}: top-20 loaded, own R² = {own_r2[outcome]:.4f}, "
                  f"AR baseline = {ar_baselines[outcome]:.4f}, own ΔR² = {own_delta:.4f}")
            print(f"    Top-5: {top20[outcome][:5]}")
        except (FileNotFoundError, ValueError) as e:
            print(f"  WARNING: {e}")

    if len(top20) < 2:
        print("ERROR: Need at least 2 outcomes with SHAP data.")
        return pd.DataFrame()

    rows = []
    for source_outcome in CROSS_PRED_OUTCOMES:
        if source_outcome not in top20:
            continue
        source_features = top20[source_outcome]

        for target_outcome in CROSS_PRED_OUTCOMES:
            if target_outcome == source_outcome:
                continue
            if target_outcome not in own_r2:
                continue

            print(f"\n  Cross-prediction: {source_outcome} features -> {target_outcome}")

            # Load target data (longitudinal mode adds Y_W1 autoregressive term)
            try:
                X_full, y, all_features, age = load_master(
                    target_outcome, tiers=[1, 2], longitudinal=True, age_residualize=False
                )
            except Exception as e:
                print(f"    ERROR loading data: {e}")
                continue

            # Determine which source features are available in target's feature set
            available_features = set(all_features)
            selected = [f for f in source_features if f in available_features]

            # Always include the target's autoregressive W1 term if present
            # (e.g. FInoPsy_W1 when predicting FInoPsy_W2)
            w1_map = {
                "FInoPsy_W2": "FInoPsy_W1",
                "btactComp_W2": "btactComp_W1",
            }
            ar_term = w1_map.get(target_outcome)
            if ar_term and ar_term in available_features and ar_term not in selected:
                selected.append(ar_term)

            if len(selected) < 2:
                print(f"    WARNING: Only {len(selected)} features available, skipping.")
                continue

            X_subset = X_full[selected].copy()
            print(f"    Using {len(selected)} features ({len(source_features)} source + AR term)")

            # Run 5-fold CV
            cv_result = pipeline.run_cv(X_subset, y, selected, age=age)
            cross_r2 = cv_result["r2_mean"]
            target_r2 = own_r2[target_outcome]
            ratio = cross_r2 / target_r2 if target_r2 > 0 else float("nan")

            # Delta R² (incremental over autoregressive Y_W1 term)
            ar_base = ar_baselines[target_outcome]
            cross_delta = cross_r2 - ar_base
            own_delta = target_r2 - ar_base
            delta_ratio = cross_delta / own_delta if own_delta > 0.001 else float("nan")

            print(f"    Cross R² = {cross_r2:.4f} | AR baseline = {ar_base:.4f} | "
                  f"Cross ΔR² = {cross_delta:.4f} | Own ΔR² = {own_delta:.4f} | "
                  f"ΔR² ratio = {delta_ratio:.3f}")

            rows.append({
                "source_outcome": source_outcome,
                "target_outcome": target_outcome,
                "source_top20_features": ",".join(source_features),
                "cross_r2": cross_r2,
                "target_own_r2": target_r2,
                "r2_ratio": ratio,
                "ar_baseline_r2": ar_base,
                "cross_delta_r2": cross_delta,
                "own_delta_r2": own_delta,
                "delta_r2_ratio": delta_ratio,
            })

    result_df = pd.DataFrame(rows)
    out_path = OUTPUT_DIR / "a3_cross_prediction.csv"
    result_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    # Print summary table
    print("\n  Cross-Prediction Summary (ΔR² = incremental over AR term):")
    print(f"  {'Source':<18s} {'Target':<18s} {'Cross ΔR²':>10s} {'Own ΔR²':>9s} {'ΔR² Ratio':>10s}")
    print("  " + "-" * 70)
    for _, row in result_df.iterrows():
        print(
            f"  {row['source_outcome']:<18s} {row['target_outcome']:<18s} "
            f"{row['cross_delta_r2']:>10.4f} {row['own_delta_r2']:>9.4f} "
            f"{row['delta_r2_ratio']:>10.3f}"
        )

    return result_df


# CS-1 cross-prediction

CS1_CROSS_PRED_OUTCOMES = list(DOMAINS.values())  # W1 outcomes


def _get_cs1_top20_features(outcome: str) -> list[str]:
    """Get top-20 SHAP predictors for a CS-1 outcome from bootstrap SHAP CSV."""
    suffix = "_adjusted" if outcome == "FInoPsy_W1" else ""
    path = ap.CS1_DIR / "shap" / f"{outcome}_T1T2{suffix}_bootstrap_shap.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    df = pd.read_csv(path, index_col=0)
    return df["mean_abs_shap"].sort_values(ascending=False).head(20).index.tolist()


def _get_cs1_own_r2(outcome: str) -> float:
    """Get the T1T2 R² for a CS-1 outcome from cs1_r2_table.csv."""
    r2_path = CS1_DIR / "cs1_r2_table.csv"
    if not r2_path.exists():
        raise FileNotFoundError(f"Missing: {r2_path}")
    df = pd.read_csv(r2_path)
    row = df[(df["outcome"] == outcome) & (df["tier_config"] == "T1T2")]
    if row.empty:
        raise ValueError(f"No T1T2 R² found for {outcome} in {r2_path}")
    return float(row["r2_mean"].iloc[0])


def run_cross_prediction_cs1() -> pd.DataFrame:
    """CS-1 cross-prediction: same logic as Layer 3 but within CS-1 framework.

    For each pair of CS-1 outcomes (4 domains, 12 directed pairs):
      1. Take source outcome's top-20 CS-1 SHAP features
      2. Train XGBoost 5-fold CV predicting target W1 outcome (cross-sectional)
      3. Compare to target's own T1T2 R² from CS-1

    No autoregressive term (cross-sectional prediction).
    """
    from analysis.core.data_loader import load_master
    from analysis.core.xgboost_pipeline import XGBoostPipeline

    print("\nCS-1 CROSS-PREDICTION TEST")

    pipeline = XGBoostPipeline(n_folds=5, n_bootstrap=0, use_gpu=True)

    top20: dict[str, list[str]] = {}
    own_r2: dict[str, float] = {}
    for outcome in CS1_CROSS_PRED_OUTCOMES:
        try:
            top20[outcome] = _get_cs1_top20_features(outcome)
            own_r2[outcome] = _get_cs1_own_r2(outcome)
            print(f"  {outcome}: top-20 loaded, own R² = {own_r2[outcome]:.4f}")
            print(f"    Top-5: {top20[outcome][:5]}")
        except (FileNotFoundError, ValueError) as e:
            print(f"  WARNING: {e}")

    if len(top20) < 2:
        print("ERROR: Need at least 2 outcomes with CS-1 SHAP data.")
        return pd.DataFrame()

    rows = []
    for source_outcome in CS1_CROSS_PRED_OUTCOMES:
        if source_outcome not in top20:
            continue
        source_features = top20[source_outcome]

        for target_outcome in CS1_CROSS_PRED_OUTCOMES:
            if target_outcome == source_outcome:
                continue
            if target_outcome not in own_r2:
                continue

            print(f"\n  Cross-prediction: {source_outcome} features -> {target_outcome}")

            try:
                X_full, y, all_features, age = load_master(
                    target_outcome, tiers=[1, 2], longitudinal=False, age_residualize=True
                )
            except Exception as e:
                print(f"    ERROR loading data: {e}")
                continue

            available_features = set(all_features)
            selected = [f for f in source_features if f in available_features]

            if len(selected) < 2:
                print(f"    WARNING: Only {len(selected)} features available, skipping.")
                continue

            X_subset = X_full[selected].copy()
            print(f"    Using {len(selected)} of {len(source_features)} source features")

            cv_result = pipeline.run_cv(X_subset, y, selected, age=age)
            cross_r2 = cv_result["r2_mean"]
            target_r2 = own_r2[target_outcome]
            ratio = cross_r2 / target_r2 if target_r2 > 0 else float("nan")

            print(f"    Cross R² = {cross_r2:.4f} | Target own R² = {target_r2:.4f} | Ratio = {ratio:.3f}")

            rows.append({
                "source_outcome": source_outcome,
                "target_outcome": target_outcome,
                "source_top20_features": ",".join(source_features),
                "cross_r2": cross_r2,
                "target_own_r2": target_r2,
                "r2_ratio": ratio,
            })

    result_df = pd.DataFrame(rows)
    out_path = OUTPUT_DIR / "a3_cross_prediction_cs1.csv"
    result_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    print("\n  CS-1 Cross-Prediction Summary:")
    print(f"  {'Source':<18s} {'Target':<18s} {'Cross R²':>9s} {'Own R²':>8s} {'Ratio':>7s}")
    print("  " + "-" * 62)
    for _, row in result_df.iterrows():
        print(
            f"  {row['source_outcome']:<18s} {row['target_outcome']:<18s} "
            f"{row['cross_r2']:>9.4f} {row['target_own_r2']:>8.4f} {row['r2_ratio']:>7.3f}"
        )

    return result_df


# Core-Periphery Cross-Prediction Decomposition

# Autoregressive W1 term for each longitudinal outcome, derived from the two domain maps.
_W1_MAP = {DOMAINS_LONGITUDINAL[k]: DOMAINS[k] for k in DOMAINS_LONGITUDINAL if k in DOMAINS}

# Domain label lookups
_LABEL_A4 = {v: k for k, v in DOMAINS_LONGITUDINAL.items()}
_LABEL_CS1 = {v: k for k, v in DOMAINS.items()}


def _get_top_features(outcome: str, k: int, mode: str) -> list[str]:
    """Get top-K SHAP features for an outcome.

    Args:
        outcome: outcome name
        k: number of top features to return
        mode: 'a4' or 'cs1'
    """
    if mode == "a4":
        sv = _load_longitudinal_shap(outcome)
    else:
        suffix = "_adjusted" if outcome == "FInoPsy_W1" else ""
        path = ap.CS1_DIR / "shap" / f"{outcome}_T1T2{suffix}_bootstrap_shap.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing: {path}")
        sv = pd.read_csv(path, index_col=0)["mean_abs_shap"]
    return sv.sort_values(ascending=False).head(k).index.tolist()


def classify_pairwise_features(
    top_features: dict[str, list[str]],
    ar_terms: set[str] | None = None,
) -> pd.DataFrame:
    """Classify features as shared vs unique for each domain pair.

    Args:
        top_features: {outcome: [feature1, ...]}, top-K per outcome
        ar_terms: AR term names to exclude from classification

    Returns:
        DataFrame: outcome_a, outcome_b, feature, category (shared/unique_a/unique_b)
    """
    if ar_terms is None:
        ar_terms = set()

    psych = {
        o: [f for f in feats if f not in ar_terms]
        for o, feats in top_features.items()
    }

    outcomes = list(psych.keys())
    rows = []
    for oa, ob in itertools.combinations(outcomes, 2):
        sa, sb = set(psych[oa]), set(psych[ob])
        for f in sorted(sa & sb):
            rows.append({"outcome_a": oa, "outcome_b": ob, "feature": f, "category": "shared"})
        for f in sorted(sa - sb):
            rows.append({"outcome_a": oa, "outcome_b": ob, "feature": f, "category": "unique_a"})
        for f in sorted(sb - sa):
            rows.append({"outcome_a": oa, "outcome_b": ob, "feature": f, "category": "unique_b"})

    result = pd.DataFrame(rows)
    out_path = OUTPUT_DIR / "a3_pairwise_feature_sets.csv"
    result.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    label_map = {**_LABEL_A4, **_LABEL_CS1}
    for oa, ob in itertools.combinations(outcomes, 2):
        sub = result[(result["outcome_a"] == oa) & (result["outcome_b"] == ob)]
        ns = (sub["category"] == "shared").sum()
        na = (sub["category"] == "unique_a").sum()
        nb = (sub["category"] == "unique_b").sum()
        shared = sub[sub["category"] == "shared"]["feature"].tolist()
        la, lb = label_map.get(oa, oa), label_map.get(ob, ob)
        print(f"  {la} vs {lb}: {ns} shared, {na} unique_{la}, {nb} unique_{lb}")
        print(f"    Shared: {', '.join(shared[:7])}{'...' if ns > 7 else ''}")

    # Global overlap summary
    if len(outcomes) >= 3:
        from collections import Counter
        fc = Counter()
        for feats in psych.values():
            fc.update(feats)
        nd = len(outcomes)
        print(f"\n  Global overlap ({nd} domains):")
        for t in range(nd, 0, -1):
            feats_at = sorted([f for f, c in fc.items() if c == t])
            print(f"    {t}/{nd}: {len(feats_at)} features: {', '.join(feats_at[:6])}{'...' if len(feats_at) > 6 else ''}")

    return result


# Fixed hyperparameters: modal values from 260 tuned configs across
# CS-1, A4, A9 analyses (depth=3/lr=0.01 in 56.5% of runs).
# Using these avoids the 55× tuning overhead (18 combos × 3-fold inner CV)
# while matching the parameters that tuning selects most often.
_FIXED_XGB_PARAMS = {
    "max_depth": 3,
    "learning_rate": 0.01,
    "subsample": 1.0,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "n_estimators": 500,
}


def _pairwise_sets(
    pair_df: pd.DataFrame, oa: str, ob: str
) -> tuple[list[str], list[str], list[str]]:
    """Return (shared, unique_a, unique_b) for a directed pair."""
    mask = (pair_df["outcome_a"] == oa) & (pair_df["outcome_b"] == ob)
    if mask.any():
        sub = pair_df[mask]
        return (
            sub[sub["category"] == "shared"]["feature"].tolist(),
            sub[sub["category"] == "unique_a"]["feature"].tolist(),
            sub[sub["category"] == "unique_b"]["feature"].tolist(),
        )
    mask = (pair_df["outcome_a"] == ob) & (pair_df["outcome_b"] == oa)
    sub = pair_df[mask]
    return (
        sub[sub["category"] == "shared"]["feature"].tolist(),
        sub[sub["category"] == "unique_b"]["feature"].tolist(),
        sub[sub["category"] == "unique_a"]["feature"].tolist(),
    )


def run_cross_prediction_decomposed(
    mode: str = "a4",
    k: int = 20,
    smoke_test: bool = False,
    n_gpus: int = 2,
) -> pd.DataFrame:
    """Pairwise cross-prediction decomposed into shared/unique feature sets.

    Multi-GPU accelerated: distributes (source, target, feature_set) jobs
    across GPUs using ThreadPoolExecutor. Each GPU gets its own
    XGBoostPipeline instance.

    For each directed pair (source to target), fits XGBoost 5-fold CV:
      full:           all source top-K psychosocial features (+ AR for A4)
      shared:         features shared between source and target (+ AR)
      unique_source:  features unique to source (+ AR)
      unique_target:  features unique to target, predicting target itself (+ AR)

    Args:
        mode: 'a4' (longitudinal, primary) or 'cs1' (cross-sectional)
        k: number of top features per domain (default 20)
        smoke_test: subsample data to n=500 for fast verification
        n_gpus: number of CUDA devices (default 2)
    """
    from concurrent.futures import ThreadPoolExecutor
    import threading
    from analysis.core.data_loader import load_master
    from analysis.core.xgboost_pipeline import XGBoostPipeline

    print(f"\nCORE-PERIPHERY DECOMPOSITION: {mode.upper()} (top-{k}, {n_gpus} GPUs)")

    if mode == "a4":
        outcomes = CROSS_PRED_OUTCOMES
        ar_terms = set(_W1_MAP.values())
    else:
        outcomes = CS1_CROSS_PRED_OUTCOMES
        ar_terms = set()

    # Load top-K features, own R², AR baselines
    top_features: dict[str, list[str]] = {}
    own_r2: dict[str, float] = {}
    ar_baselines: dict[str, float] = {}
    for outcome in outcomes:
        top_features[outcome] = _get_top_features(outcome, k, mode)
        if mode == "a4":
            own_r2[outcome] = _get_own_r2(outcome)
            ar_baselines[outcome] = _get_ar_baseline_r2(outcome)
        else:
            own_r2[outcome] = _get_cs1_own_r2(outcome)
            ar_baselines[outcome] = 0.0
        print(f"  {outcome}: top-{k} loaded, own R² = {own_r2[outcome]:.4f}")

    # Classify features pairwise
    print("\n  Pairwise feature classification:")
    pair_df = classify_pairwise_features(top_features, ar_terms=ar_terms)

    # Pre-load all target data
    longitudinal = mode == "a4"
    data_cache: dict[str, tuple] = {}  # target: (X, y, features, age)
    for target in outcomes:
        try:
            X_full, y, all_features, age = load_master(
                target, tiers=[1, 2],
                longitudinal=longitudinal,
                age_residualize=not longitudinal,
            )
            if smoke_test:
                rng = np.random.default_rng(42)
                idx = rng.choice(len(y), min(500, len(y)), replace=False)
                X_full = X_full.iloc[idx].reset_index(drop=True)
                y = y.iloc[idx].reset_index(drop=True)
                if age is not None:
                    age = age[idx]
            data_cache[target] = (X_full, y, set(all_features), age)
        except Exception as e:
            print(f"  ERROR loading {target}: {e}")

    # Build job list
    Job = tuple  # (source, target, fset_name, feats, ar_term, ar_base, own_r2_tgt)
    jobs: list[Job] = []
    for source in outcomes:
        for target in outcomes:
            if source == target or target not in data_cache:
                continue

            shared, unique_src, unique_tgt = _pairwise_sets(pair_df, source, target)
            avail = data_cache[target][2]
            ar_term = _W1_MAP.get(target) if mode == "a4" else None
            src_psych = [f for f in top_features[source] if f not in ar_terms]

            feature_sets = {
                "full": [f for f in src_psych if f in avail],
                "shared": [f for f in shared if f in avail],
                "unique_source": [f for f in unique_src if f in avail],
                "unique_target": [f for f in unique_tgt if f in avail],
            }
            if ar_term and ar_term in avail:
                for fs in feature_sets.values():
                    if ar_term not in fs:
                        fs.append(ar_term)

            for fset_name, feats in feature_sets.items():
                if len(feats) >= 2:
                    jobs.append((source, target, fset_name, feats,
                                 ar_term, ar_baselines[target], own_r2[target]))

    print(f"\n  Total jobs: {len(jobs)}, distributing across {n_gpus} GPUs")

    # Multi-GPU worker
    results_lock = threading.Lock()
    all_rows: list[dict] = []
    progress = [0]

    def _worker(gpu_id: int, job_chunk: list[Job]) -> None:
        pipe = XGBoostPipeline(n_folds=5, n_bootstrap=0, use_gpu=True, tune=False)
        pipe.params.update(_FIXED_XGB_PARAMS)
        pipe.params["device"] = f"cuda:{gpu_id}"

        for source, target, fset_name, feats, ar_term, ar_base, own_r2_tgt in job_chunk:
            X_full, y, _, age = data_cache[target]
            X_sub = X_full[feats].copy()
            r2 = pipe.run_cv(X_sub, y, feats, age=age)["r2_mean"]
            dr2 = r2 - ar_base
            n_psych = len([f for f in feats if f != ar_term])

            row = {
                "source_outcome": source,
                "target_outcome": target,
                "feature_set": fset_name,
                "n_psych_features": n_psych,
                "features": ",".join(feats),
                "cross_r2": r2,
                "target_own_r2": own_r2_tgt,
                "r2_ratio": r2 / own_r2_tgt if own_r2_tgt > 0 else float("nan"),
                "ar_baseline_r2": ar_base,
                "delta_r2": dr2,
                "own_delta_r2": own_r2_tgt - ar_base,
                "delta_r2_ratio": (
                    dr2 / (own_r2_tgt - ar_base)
                    if (own_r2_tgt - ar_base) > 0.001
                    else float("nan")
                ),
            }
            with results_lock:
                all_rows.append(row)
                progress[0] += 1
                print(f"  [{progress[0]}/{len(jobs)}] GPU{gpu_id}  "
                      f"{source} -> {target} {fset_name:20s}: "
                      f"R²={r2:.4f}  ΔR²={dr2:.4f}  ({n_psych} psych)")

    # Distribute jobs round-robin across GPUs
    chunks: list[list[Job]] = [[] for _ in range(n_gpus)]
    for i, job in enumerate(jobs):
        chunks[i % n_gpus].append(job)

    with ThreadPoolExecutor(max_workers=n_gpus) as executor:
        futures = [executor.submit(_worker, gid, chunk)
                   for gid, chunk in enumerate(chunks)]
        for f in futures:
            f.result()  # propagate exceptions

    result_df = pd.DataFrame(all_rows)
    # Sort to deterministic order
    result_df = result_df.sort_values(
        ["source_outcome", "target_outcome", "feature_set"]
    ).reset_index(drop=True)
    out_path = OUTPUT_DIR / f"a3_cross_prediction_decomposed_{mode}.csv"
    result_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    # Summary table
    metric = "ΔR²" if mode == "a4" else "R²"
    print(f"\n  {'Source':<15s} {'Target':<15s} {'Set':<20s} {'N':>3s} {metric:>8s} {'Ratio':>7s}")
    print("  " + "-" * 72)
    for _, row in result_df.iterrows():
        val = row["delta_r2"] if mode == "a4" else row["cross_r2"]
        ratio = row["delta_r2_ratio"] if mode == "a4" else row["r2_ratio"]
        print(
            f"  {row['source_outcome']:<15s} {row['target_outcome']:<15s} "
            f"{row['feature_set']:<20s} {row['n_psych_features']:>3.0f} "
            f"{val:>8.4f} {ratio:>7.3f}"
        )

    return result_df


def run_partition_null(
    mode: str = "a4",
    n_perm: int = 100,
    smoke_test: bool = False,
    seed: int = 42,
    n_gpus: int = 2,
) -> pd.DataFrame:
    """Random-partition null for shared feature transferability.

    Multi-GPU accelerated: for each directed pair, distributes permutations
    across GPUs. Each GPU fits its share of random feature subsets.

    For each directed pair, randomly samples n_shared features from source's
    top-K psychosocial features, fits XGBoost on a single 80/20 split, and
    tests whether actual shared features transfer better than random subsets.
    """
    from concurrent.futures import ThreadPoolExecutor
    import threading
    from analysis.core.data_loader import load_master
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split
    import xgboost as xgb

    print(f"\nRANDOM-PARTITION NULL: {mode.upper()} ({n_perm} perms, {n_gpus} GPUs)")

    rng = np.random.default_rng(seed)

    if mode == "a4":
        outcomes = CROSS_PRED_OUTCOMES
        ar_terms = set(_W1_MAP.values())
    else:
        outcomes = CS1_CROSS_PRED_OUTCOMES
        ar_terms = set()

    top_features: dict[str, list[str]] = {}
    for outcome in outcomes:
        if mode == "a4":
            top_features[outcome] = _get_top20_features(outcome)
        else:
            top_features[outcome] = _get_cs1_top20_features(outcome)

    pair_df = classify_pairwise_features(top_features, ar_terms=ar_terms)

    def _get_shared(oa, ob):
        mask = (pair_df["outcome_a"] == oa) & (pair_df["outcome_b"] == ob)
        if mask.any():
            return pair_df[mask & (pair_df["category"] == "shared")]["feature"].tolist()
        mask = (pair_df["outcome_a"] == ob) & (pair_df["outcome_b"] == oa)
        return pair_df[mask & (pair_df["category"] == "shared")]["feature"].tolist()

    rows = []
    for source in outcomes:
        for target in outcomes:
            if source == target:
                continue

            shared = _get_shared(source, target)
            n_shared = len(shared)
            src_psych = [f for f in top_features[source] if f not in ar_terms]

            if n_shared < 2:
                print(f"  {source} -> {target}: skipping (n_shared={n_shared})")
                continue

            print(f"  {source} -> {target}: n_shared={n_shared}, pool={len(src_psych)}")

            longitudinal = mode == "a4"
            X_full, y, all_features, age = load_master(
                target, tiers=[1, 2],
                longitudinal=longitudinal,
                age_residualize=not longitudinal,
            )

            if smoke_test:
                idx = rng.choice(len(y), min(500, len(y)), replace=False)
                X_full = X_full.iloc[idx].reset_index(drop=True)
                y = y.iloc[idx].reset_index(drop=True)

            avail = set(all_features)
            ar_term = _W1_MAP.get(target) if mode == "a4" else None

            X_train, X_test, y_train, y_test = train_test_split(
                X_full, y, test_size=0.2, random_state=seed,
            )

            def _make_feats(feature_list: list[str]) -> list[str]:
                feats = [f for f in feature_list if f in avail]
                if ar_term and ar_term in avail and ar_term not in feats:
                    feats.append(ar_term)
                return feats

            # Actual shared R² (single fit on GPU 0)
            actual_feats = _make_feats(shared)
            if len(actual_feats) < 2:
                print(f"    skipping (<2 features)")
                continue
            model = xgb.XGBRegressor(
                **_FIXED_XGB_PARAMS,
                random_state=seed, n_jobs=-1, tree_method="hist",
                device="cuda:0", enable_categorical=True,
            )
            model.fit(X_train[actual_feats], y_train, verbose=False)
            actual_r2 = float(r2_score(y_test, model.predict(X_test[actual_feats])))

            # Pre-generate all random subsets for reproducibility
            all_subsets = [
                list(rng.choice(src_psych, size=n_shared, replace=False))
                for _ in range(n_perm)
            ]

            # Multi-GPU null distribution
            null_r2s = np.full(n_perm, np.nan)
            progress_lock = threading.Lock()
            progress = [0]

            def _null_worker(gpu_id: int, perm_indices: list[int]) -> None:
                for pi in perm_indices:
                    feats = _make_feats(all_subsets[pi])
                    if len(feats) < 2:
                        continue
                    m = xgb.XGBRegressor(
                        **_FIXED_XGB_PARAMS,
                        random_state=seed, n_jobs=-1, tree_method="hist",
                        device=f"cuda:{gpu_id}", enable_categorical=True,
                    )
                    m.fit(X_train[feats], y_train, verbose=False)
                    null_r2s[pi] = float(r2_score(y_test, m.predict(X_test[feats])))

                    with progress_lock:
                        progress[0] += 1
                        if progress[0] % 20 == 0:
                            print(f"    perm {progress[0]}/{n_perm}")

            perm_chunks: list[list[int]] = [[] for _ in range(n_gpus)]
            for i in range(n_perm):
                perm_chunks[i % n_gpus].append(i)

            with ThreadPoolExecutor(max_workers=n_gpus) as executor:
                futures = [executor.submit(_null_worker, gid, chunk)
                           for gid, chunk in enumerate(perm_chunks)]
                for f in futures:
                    f.result()

            p_val = float(np.mean(null_r2s >= actual_r2))
            print(f"    shared R²={actual_r2:.4f}, null={np.nanmean(null_r2s):.4f}±{np.nanstd(null_r2s):.4f}, p={p_val:.3f}")

            rows.append({
                "source_outcome": source,
                "target_outcome": target,
                "n_shared": n_shared,
                "n_source_psych": len(src_psych),
                "actual_shared_r2": actual_r2,
                "null_mean_r2": float(np.nanmean(null_r2s)),
                "null_sd_r2": float(np.nanstd(null_r2s)),
                "p_value": p_val,
                "n_perm": n_perm,
            })

    result_df = pd.DataFrame(rows)
    out_path = OUTPUT_DIR / f"a3_partition_null_{mode}.csv"
    result_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    return result_df


# Cross-domain counts over the three primary domains, as reported in the paper,
# computed from this pipeline's tables.

BACKGROUND_DOMAINS = ["Frailty", "Cognition"]
AL_CONSTRUCTIONS = ["Allostatic Load"]


def _reported_trios(w: str) -> dict[str, set[str]]:
    """Outcome columns of each three-domain set the manuscript reports."""
    src = DOMAINS if w == "W1" else DOMAINS_LONGITUDINAL
    return {al: {src[d] for d in BACKGROUND_DOMAINS + [al]}
            for al in AL_CONSTRUCTIONS if al in src}


def run_crossdomain_summary() -> pd.DataFrame:
    """Reduce the decomposition and partition-null tables to the stated counts."""
    rows = []
    for layer, mode, w in [("concurrent", "cs1", "W1"), ("longitudinal", "a4", "W2")]:
        dec_path = OUTPUT_DIR / f"a3_cross_prediction_decomposed_{mode}.csv"
        null_path = OUTPUT_DIR / f"a3_partition_null_{mode}.csv"
        if not (dec_path.exists() and null_path.exists()):
            print(f"  [{layer}] skipped: {dec_path.name} or {null_path.name} absent")
            continue
        dec = pd.read_csv(dec_path)
        nul = pd.read_csv(null_path)
        wide = dec.pivot_table(index=["source_outcome", "target_outcome"],
                               columns="feature_set", values="cross_r2")
        for al, trio in _reported_trios(w).items():
            sel = wide[[a in trio and b in trio for a, b in wide.index]]
            if sel.empty:
                print(f"  [{layer}] {al}: no decomposed pairs, skipped")
                continue
            ut = sel["unique_target"] > sel["shared"]
            full = ut & (sel["shared"] > sel["unique_source"])
            q = nul[nul.source_outcome.isin(trio) & nul.target_outcome.isin(trio)]
            nonsig = q.loc[q.p_value >= 0.05, "p_value"]
            rows.append({
                "layer": layer,
                "al_construction": al,
                "n_pairs": int(len(sel)),
                "k_unique_target_gt_shared": int(ut.sum()),
                "k_full_ordering": int(full.sum()),
                "n_pairs_null": int(len(q)),
                "k_partition_null_p05": int((q.p_value < 0.05).sum()),
                "closest_nonsig_p": float(nonsig.min()) if len(nonsig) else None,
            })

    result_df = pd.DataFrame(rows)
    out_path = OUTPUT_DIR / "a3_crossdomain_summary.csv"
    result_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    if not result_df.empty:
        print(result_df.to_string(index=False))
    return result_df


def a3_output_entries() -> list[dict]:
    names = [
        "a3_pairwise_rho.csv",
        "a3_bootstrap_nulls.csv",
        "a3_stability_filtered.csv",
        "a3_subdomain_divergence.csv",
        "a3_divergence_decomposition.csv",
        "a3_topk_sensitivity.csv",
        "a3_cross_prediction.csv",
        "a3_cross_prediction_cs1.csv",
        "a3_longitudinal_stability.csv",
        "a3_longitudinal_subdomain.csv",
        "a3_longitudinal_decomposition.csv",
        "a3_pairwise_feature_sets.csv",
        "a3_cross_prediction_decomposed_a4.csv",
        "a3_cross_prediction_decomposed_cs1.csv",
        "a3_partition_null_a4.csv",
        "a3_partition_null_cs1.csv",
        "a3_crossdomain_summary.csv",
    ]
    return [{"path": OUTPUT_DIR / name, "kind": "CSV"} for name in names]


def write_pipeline_log(run_args: dict | None = None) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_output_log(
        OUTPUT_DIR,
        "a3_output_log.md",
        Path(__file__).name,
        a3_output_entries(),
        run_args or {},
    )


def main():
    # Record input provenance and the age-adjustment method beside the outputs.
    provenance.stamp_at_exit(OUTPUT_DIR, "a3_domain_divergence", ["data/processed/master_analysis.csv", "data/processed/master_analysis_concurrent.csv", "output/a4_longitudinal", "output/cs1_cross_sectional"])
    age_method.emit_at_exit(OUTPUT_DIR, "a3_domain_divergence")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("A3: FORMAL DOMAIN DIVERGENCE TEST")

    # Load bootstrap SHAP data
    domain_data = {}
    for domain, outcome in DOMAINS.items():
        use_adjusted = (outcome == "FInoPsy_W1")
        try:
            shap_raw, features = load_bootstrap_shap(
                outcome, CS1_DIR, use_adjusted=use_adjusted
            )
            domain_data[domain] = (shap_raw, features)
            print(f"  Loaded {domain} ({outcome}): {shap_raw.shape}")
        except FileNotFoundError as e:
            print(f"  WARNING: {e}")

    if len(domain_data) < 2:
        print("ERROR: Need at least 2 domains. Run CS-1 first.")
        sys.exit(1)

    domains = list(domain_data.keys())
    n_pairs = len(list(itertools.combinations(domains, 2)))
    print(f"\n  Domains: {domains}")
    print(f"  Pairwise comparisons: {n_pairs}")

    # Align to common feature set
    aligned, common_features = align_shap_vectors(domain_data)
    print(f"  Common feature space: {len(common_features)} variables")

    # Compute observed mean |SHAP| per domain
    mean_shap = {d: aligned[d].mean(axis=0) for d in domains}

    # Observed pairwise ρ
    observed_rho = compute_pairwise_rho(mean_shap, domains)
    print("\n  Observed pairwise Spearman ρ:")
    for (d1, d2), rho in observed_rho.items():
        print(f"    {d1} vs {d2}: ρ = {rho:.4f}")

    # Bootstrap null
    print(f"\n  Building bootstrap null ({N_PERMUTATIONS} iterations)...")
    null_rho = bootstrap_null(aligned, domains, n_perm=N_PERMUTATIONS)

    # Compute p-values (one-sided: divergence = observed ρ lower than null)
    # Under the pooled null, ρ should be high (~1.0). If observed ρ is
    # significantly lower, domains are genuinely divergent.
    # p = fraction of null ρ ≤ observed ρ (small if observed is well below null)
    pairs = list(itertools.combinations(domains, 2))
    p_values = []
    for pair in pairs:
        obs = observed_rho[pair]
        null = null_rho[pair]
        p = float(np.mean(null <= obs))  # fraction of null ≤ observed
        p_values.append(p)

    # Holm-Bonferroni correction
    p_adjusted = holm_bonferroni(p_values)

    # Build results table
    result_rows = []
    for i, (d1, d2) in enumerate(pairs):
        result_rows.append({
            "domain_1": d1,
            "domain_2": d2,
            "observed_rho": observed_rho[(d1, d2)],
            "null_mean": float(null_rho[(d1, d2)].mean()),
            "null_sd": float(null_rho[(d1, d2)].std()),
            "p_value": p_values[i],
            "p_adjusted": p_adjusted[i],
            "significant": p_adjusted[i] < 0.05,
        })

    result_df = pd.DataFrame(result_rows)
    result_df.to_csv(OUTPUT_DIR / "a3_pairwise_rho.csv", index=False)
    print(f"\nSaved: a3_pairwise_rho.csv")

    # Save null distributions
    null_rows = []
    for pair in pairs:
        for i, rho in enumerate(null_rho[pair]):
            null_rows.append({
                "pair": f"{pair[0]}_vs_{pair[1]}",
                "iteration": i,
                "rho_null": rho,
            })
    null_df = pd.DataFrame(null_rows)
    null_df.to_csv(OUTPUT_DIR / "a3_bootstrap_nulls.csv", index=False)
    print(f"Saved: a3_bootstrap_nulls.csv")

    # Layer 2: Subdomain-level divergence + stability filter
    stability_df, subdomain_df, decomposition_df = run_subdomain_divergence(
        aligned, common_features, domains, observed_rho,
        n_perm=N_PERMUTATIONS,
    )

    write_pipeline_log({"layer": "default"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A3: domain divergence test")
    parser.add_argument(
        "--layer",
        type=str,
        default=None,
        choices=["1", "2", "3", "3cs", "long", "decompose", "null", "summary", "log", "all"],
        help=(
            "Run specific layer analyses: "
            "1=top-k sensitivity, 2=subdomain divergence (standalone), "
            "3=cross-prediction (A4 longitudinal), "
            "3cs=cross-prediction (CS-1 cross-sectional), "
            "long=longitudinal divergence (A4 ρ_signal + ρ_sd_signal), "
            "decompose=core-periphery cross-prediction decomposition (A4+CS-1), "
            "null=random-partition null for shared features (A4+CS-1), "
            "summary=reduce decompose+null to the counts the manuscript states, "
            "log=write output log only, "
            "all=the default analysis and every layer"
        ),
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Run with n=500 subsample for quick verification",
    )
    parser.add_argument(
        "--n-perm", type=int, default=100,
        help="Number of permutations for partition null (default: 100)",
    )
    parser.add_argument(
        "--n-gpus", type=int, default=2,
        help="Number of CUDA devices for decompose/null layers (default: 2)",
    )
    args = parser.parse_args()

    if args.layer is None:
        # Full analysis: feature-level + subdomain-level divergence (CS-1)
        main()
    elif args.layer == "2":
        # Standalone subdomain divergence (requires CS-1 data)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        domain_data = {}
        for domain, outcome in DOMAINS.items():
            use_adjusted = (outcome == "FInoPsy_W1")
            shap_raw, features = load_bootstrap_shap(outcome, CS1_DIR, use_adjusted=use_adjusted)
            domain_data[domain] = (shap_raw, features)
        aligned, common_features = align_shap_vectors(domain_data)
        domains = list(domain_data.keys())
        mean_shap = {d: aligned[d].mean(axis=0) for d in domains}
        observed_rho = compute_pairwise_rho(mean_shap, domains)
        run_subdomain_divergence(aligned, common_features, domains, observed_rho, n_perm=N_PERMUTATIONS)
        write_pipeline_log(vars(args))
    else:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        if args.layer == "log":
            write_pipeline_log(vars(args))
        if args.layer == "all":
            main()
        if args.layer in ("1", "all"):
            run_topk_sensitivity()
        if args.layer in ("3", "all"):
            run_cross_prediction()
        if args.layer in ("3cs", "all"):
            run_cross_prediction_cs1()
        if args.layer in ("long", "all"):
            run_longitudinal_divergence(n_perm=N_PERMUTATIONS)
        if args.layer in ("decompose", "all"):
            run_cross_prediction_decomposed(mode="a4", smoke_test=args.smoke_test, n_gpus=args.n_gpus)
            run_cross_prediction_decomposed(mode="cs1", smoke_test=args.smoke_test, n_gpus=args.n_gpus)
        if args.layer in ("null", "all"):
            run_partition_null(mode="a4", n_perm=args.n_perm, smoke_test=args.smoke_test, n_gpus=args.n_gpus)
            run_partition_null(mode="cs1", n_perm=args.n_perm, smoke_test=args.smoke_test, n_gpus=args.n_gpus)
        if args.layer in ("summary", "all"):
            run_crossdomain_summary()
        if args.layer != "log":
            write_pipeline_log(vars(args))

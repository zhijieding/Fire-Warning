#!/usr/bin/env python3
"""
Fire-evolution time-scale analysis from an event-level CSV.

Answers:
  1. How long after ignition each hazard variable typically reaches its worst state.
  2. How long after the peak each variable typically returns to a safe range.
  3. Implications for rescue, re-entry, and voyage-resumption decisions.

Development-stage statistics use only ``evolution_role == full_process`` because
post-fire recordings start after ignition (left-censored for escalation timing).
Recovery statistics include both full_process and post_fire experiments.

Usage (from Fire_prediction/)::

    python -m dual_track_analysis.analysis.fire_evolution_timescale_analysis

    python -m dual_track_analysis.analysis.fire_evolution_timescale_analysis \\
        --event_csv dual_track_analysis/outputs/evolution_analysis/evolution_event_table.csv \\
        --out_dir dual_track_analysis/outputs/evolution_analysis/final

All default paths are relative to ``dual_track_analysis/``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

_TRACK_ROOT = Path(__file__).resolve().parents[1]
_ROOT = _TRACK_ROOT.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

from dual_track_analysis.common.evolution_paper import (
    INDICATOR_ORDER,
    RECOVERY_ROLES,
    VARIABLE_TO_INDICATOR,
    plot_time_origin_for_row,
    recovery_records,
    time_axis_label,
    uses_ignition_origin,
)

ESCALATION_INTERPRETATIONS: dict[str, str] = {
    "Temperature": "Early thermal response",
    "Heat flux": "Early thermal exposure intensity",
    "CO": "Lagged toxic accumulation",
    "Transmittance": "Visibility deterioration",
}

RECOVERY_INTERPRETATIONS: dict[str, str] = {
    "Temperature": "Thermal decay after peak",
    "Heat flux": "Radiative exposure decay",
    "CO": "Toxic gas clearance",
    "Transmittance": "Visibility restoration",
}

N_BOOT = 5000
BOOT_RANDOM_STATE = 42

FIG_OUTPUT_STEM = "fig_timescale_statistics"
PAPER_XLSX = "fire_evolution_timescale_summary_for_paper.xlsx"

BOX_STYLE = dict(
    boxprops=dict(linewidth=1.3, color="black"),
    medianprops=dict(linewidth=1.8, color="#E69F00"),
    whiskerprops=dict(linewidth=1.3, color="black"),
    capprops=dict(linewidth=1.3, color="black"),
    meanprops=dict(
        marker="^",
        markerfacecolor="#2ca02c",
        markeredgecolor="#2ca02c",
        markersize=8,
    ),
    flierprops=dict(
        marker="o",
        markerfacecolor="white",
        markeredgecolor="black",
        markersize=5,
    ),
)


def setup_matplotlib() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["axes.linewidth"] = 1.2
    plt.rcParams["xtick.direction"] = "out"
    plt.rcParams["ytick.direction"] = "out"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["font.size"] = 9
    plt.rcParams["axes.labelsize"] = 10
    plt.rcParams["axes.titlesize"] = 10


# ---------------------------------------------------------------------------
# Data loading & cleaning
# ---------------------------------------------------------------------------


def _to_bool(val: Any) -> bool:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return False
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, np.integer)):
        return bool(val)
    if isinstance(val, (float, np.floating)):
        return bool(val)
    s = str(val).strip().lower()
    return s in ("true", "1", "yes", "t")


def _to_float(val: Any) -> float:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return np.nan
    try:
        return float(val)
    except (TypeError, ValueError):
        return np.nan


def load_event_table(path: str | Path) -> pd.DataFrame:
    """Load the evolution event CSV."""
    df = pd.read_csv(path)
    required = {
        "variable",
        "evolution_role",
        "experiment_id",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in event table: {sorted(missing)}")
    return df


def clean_event_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map variable names, coerce types, and add validity flags for downstream stats.

    full_process rows drive escalation timing; recovery flags apply to post-fire records.
    """
    out = df.copy()

    out["indicator"] = out["variable"].map(VARIABLE_TO_INDICATOR)
    unknown = out["indicator"].isna()
    if unknown.any():
        bad = sorted(out.loc[unknown, "variable"].astype(str).unique())
        raise ValueError(f"Unknown variable names: {bad}")

    for col in (
        "entered_danger",
        "recovered",
        "invalid_event_order",
    ):
        if col in out.columns:
            out[col] = out[col].map(_to_bool)
        else:
            out[col] = False

    for col in (
        "time_to_first_danger_s",
        "time_to_peak_s",
        "time_to_safe_s",
        "time_peak_to_safe_s",
        "time_danger_duration_s",
    ):
        if col in out.columns:
            out[col] = out[col].map(_to_float)
        else:
            out[col] = np.nan

    for col in ("t_first_danger", "t_peak", "t_safe", "t_origin", "t_evolution_origin", "t_ignition"):
        if col in out.columns:
            out[col] = out[col].map(_to_float)
        else:
            out[col] = np.nan

    out["t_plot_origin"] = out.apply(plot_time_origin_for_row, axis=1)
    out["time_to_first_danger_s"] = np.where(
        out["entered_danger"] & out["t_first_danger"].notna(),
        out["t_first_danger"] - out["t_plot_origin"],
        np.nan,
    )
    out["time_to_peak_s"] = np.where(
        out["t_peak"].notna(),
        out["t_peak"] - out["t_plot_origin"],
        np.nan,
    )
    out["time_to_safe_s"] = np.where(
        out["recovered"] & out["t_safe"].notna(),
        out["t_safe"] - out["t_plot_origin"],
        np.nan,
    )
    out["time_peak_to_safe_s"] = np.where(
        out["recovered"] & out["t_peak"].notna() & out["t_safe"].notna(),
        out["t_safe"] - out["t_peak"],
        np.nan,
    )
    out["time_danger_duration_s"] = np.where(
        out["recovered"]
        & out["entered_danger"]
        & out["t_first_danger"].notna()
        & out["t_safe"].notna(),
        out["t_safe"] - out["t_first_danger"],
        np.nan,
    )

    out["is_full_process"] = out["evolution_role"].astype(str) == "full_process"
    out["is_right_censored"] = ~out["recovered"]

    # First-danger-to-peak interval (kept even when ordering is invalid).
    out["time_first_danger_to_peak_s"] = out["time_to_peak_s"] - out["time_to_first_danger_s"]

    out["valid_first_danger"] = (
        out["is_full_process"]
        & out["entered_danger"]
        & out["t_first_danger"].notna()
        & out["time_to_first_danger_s"].notna()
    )

    out["valid_peak"] = (
        out["is_full_process"]
        & out["t_peak"].notna()
        & out["time_to_peak_s"].notna()
    )

    out["valid_first_to_peak"] = (
        out["is_full_process"]
        & out["entered_danger"]
        & out["t_first_danger"].notna()
        & out["t_peak"].notna()
        & out["time_to_peak_s"].notna()
        & out["time_to_first_danger_s"].notna()
        & (out["time_to_peak_s"] >= out["time_to_first_danger_s"])
    )

    bad_order = (
        out["is_full_process"]
        & out["entered_danger"]
        & out["t_first_danger"].notna()
        & out["t_peak"].notna()
        & (out["t_peak"] < out["t_first_danger"])
    )
    if "invalid_event_order" in out.columns:
        out.loc[bad_order, "invalid_event_order"] = True
    else:
        out["invalid_event_order"] = bad_order

    out["valid_observed_recovery"] = (
        out["recovered"]
        & out["t_peak"].notna()
        & out["t_safe"].notna()
        & (out["t_safe"] > out["t_peak"])
        & out["time_peak_to_safe_s"].notna()
        & (out["time_peak_to_safe_s"] > 0)
    )

    out["valid_danger_only_recovery"] = (
        out["entered_danger"]
        & out["recovered"]
        & out["t_peak"].notna()
        & out["t_safe"].notna()
        & (out["t_safe"] > out["t_peak"])
    )

    return out


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def bootstrap_ci(
    values: np.ndarray,
    stat: str = "mean",
    n_boot: int = N_BOOT,
    random_state: int = BOOT_RANDOM_STATE,
) -> tuple[float, float]:
    """Percentile bootstrap 95% CI for mean or median."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan, np.nan
    if len(arr) == 1:
        return float(arr[0]), float(arr[0])

    rng = np.random.default_rng(random_state)
    boot_stats = np.empty(n_boot, dtype=float)
    n = len(arr)
    for i in range(n_boot):
        sample = arr[rng.integers(0, n, size=n)]
        boot_stats[i] = float(np.mean(sample) if stat == "mean" else np.median(sample))
    lo, hi = np.percentile(boot_stats, [2.5, 97.5])
    return float(lo), float(hi)


def _empty_stat_row(indicator: str) -> dict[str, Any]:
    keys = [
        "indicator",
        "n",
        "mean",
        "std",
        "median",
        "q25",
        "q75",
        "iqr",
        "min",
        "max",
        "mean_ci_lo",
        "mean_ci_hi",
        "median_ci_lo",
        "median_ci_hi",
        "mean_std_fmt",
        "median_iqr_fmt",
        "combined_fmt",
    ]
    row = {k: np.nan for k in keys}
    row["indicator"] = indicator
    row["n"] = 0
    for fmt_key in ("mean_std_fmt", "median_iqr_fmt", "combined_fmt"):
        row[fmt_key] = "NA"
    return row


def _compute_stat_row(values: np.ndarray, indicator: str) -> dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return _empty_stat_row(indicator)

    q25, med, q75 = np.percentile(arr, [25, 50, 75])
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    mean_lo, mean_hi = bootstrap_ci(arr, stat="mean")
    med_lo, med_hi = bootstrap_ci(arr, stat="median")

    mean_std_fmt = f"{mean:.1f} ± {std:.1f}"
    median_iqr_fmt = f"{med:.1f} [{q25:.1f}, {q75:.1f}]"
    combined_fmt = f"{mean_std_fmt} / {med:.1f} [{q25:.1f}–{q75:.1f}]"

    return {
        "indicator": indicator,
        "n": int(len(arr)),
        "mean": mean,
        "std": std,
        "median": float(med),
        "q25": float(q25),
        "q75": float(q75),
        "iqr": float(q75 - q25),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean_ci_lo": mean_lo,
        "mean_ci_hi": mean_hi,
        "median_ci_lo": med_lo,
        "median_ci_hi": med_hi,
        "mean_std_fmt": mean_std_fmt,
        "median_iqr_fmt": median_iqr_fmt,
        "combined_fmt": combined_fmt,
    }


def summarize_metric(
    df: pd.DataFrame,
    value_col: str,
    group_col: str = "indicator",
    indicators: list[str] | None = None,
) -> pd.DataFrame:
    """Descriptive stats + bootstrap CIs for one numeric column, grouped by indicator."""
    indicators = indicators or INDICATOR_ORDER
    rows: list[dict[str, Any]] = []
    for ind in indicators:
        if group_col not in df.columns:
            rows.append(_empty_stat_row(ind))
            continue
        sub = df.loc[df[group_col] == ind, value_col]
        rows.append(_compute_stat_row(sub.values, ind))
    return pd.DataFrame(rows)


def _stat_lookup(stats_df: pd.DataFrame, indicator: str) -> dict[str, Any]:
    hit = stats_df.loc[stats_df["indicator"] == indicator]
    if hit.empty:
        return _empty_stat_row(indicator)
    return hit.iloc[0].to_dict()


def build_escalation_summary(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Development-stage summary (full_process only).

    post_fire experiments are excluded because ignition-aligned escalation times
    are left-censored when recording starts after the fire has already begun.
    """
    fp = df.loc[df["is_full_process"]].copy()

    first_danger = summarize_metric(
        fp.loc[fp["valid_first_danger"]], "time_to_first_danger_s"
    )
    peak = summarize_metric(fp.loc[fp["valid_peak"]], "time_to_peak_s")
    first_to_peak = summarize_metric(
        fp.loc[fp["valid_first_to_peak"]], "time_first_danger_to_peak_s"
    )

    rows: list[dict[str, Any]] = []
    for ind in INDICATOR_ORDER:
        fd = _stat_lookup(first_danger, ind)
        pk = _stat_lookup(peak, ind)
        ftp = _stat_lookup(first_to_peak, ind)
        rows.append(
            {
                "indicator": ind,
                "metric_first_danger": "time_to_first_danger_s",
                "n_first_danger": fd["n"],
                "first_danger_mean": fd["mean"],
                "first_danger_std": fd["std"],
                "first_danger_median": fd["median"],
                "first_danger_q25": fd["q25"],
                "first_danger_q75": fd["q75"],
                "first_danger_combined_fmt": fd["combined_fmt"],
                "metric_peak": "time_to_peak_s",
                "n_peak": pk["n"],
                "peak_mean": pk["mean"],
                "peak_std": pk["std"],
                "peak_median": pk["median"],
                "peak_q25": pk["q25"],
                "peak_q75": pk["q75"],
                "peak_combined_fmt": pk["combined_fmt"],
                "metric_first_to_peak": "time_first_danger_to_peak_s",
                "n_first_to_peak": ftp["n"],
                "first_to_peak_mean": ftp["mean"],
                "first_to_peak_std": ftp["std"],
                "first_to_peak_median": ftp["median"],
                "first_to_peak_q25": ftp["q25"],
                "first_to_peak_q75": ftp["q75"],
                "first_to_peak_combined_fmt": ftp["combined_fmt"],
                "interpretation": ESCALATION_INTERPRETATIONS.get(ind, ""),
            }
        )

    detail = pd.DataFrame(rows)

    # Paper-facing wide table for Excel.
    wide = pd.DataFrame(
        {
            "Indicator": INDICATOR_ORDER,
            "n_first_danger": [int(_stat_lookup(first_danger, i)["n"]) for i in INDICATOR_ORDER],
            "First danger time, s": [
                _stat_lookup(first_danger, i)["combined_fmt"] for i in INDICATOR_ORDER
            ],
            "n_peak": [int(_stat_lookup(peak, i)["n"]) for i in INDICATOR_ORDER],
            "Time to peak, s": [
                _stat_lookup(peak, i)["combined_fmt"] for i in INDICATOR_ORDER
            ],
            "n_first_to_peak": [
                int(_stat_lookup(first_to_peak, i)["n"]) for i in INDICATOR_ORDER
            ],
            "First danger to peak, s": [
                _stat_lookup(first_to_peak, i)["combined_fmt"] for i in INDICATOR_ORDER
            ],
            "Interpretation": [ESCALATION_INTERPRETATIONS[i] for i in INDICATOR_ORDER],
        }
    )
    return detail, wide, first_danger, peak, first_to_peak


def build_recovery_summary(
    df: pd.DataFrame,
    mode: str = "observed",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Recovery-stage summary.

    mode='observed': all roles, any post-peak return to safe threshold.
    mode='danger_only': subset that entered the danger threshold before recovery.
    """
    if mode not in ("observed", "danger_only"):
        raise ValueError(f"Unknown recovery mode: {mode}")

    flag = "valid_observed_recovery" if mode == "observed" else "valid_danger_only_recovery"
    pool = df.loc[df["evolution_role"].isin(["full_process", "post_fire"])].copy()

    detail_rows: list[dict[str, Any]] = []
    for ind in INDICATOR_ORDER:
        g = pool.loc[pool["indicator"] == ind]
        rec = g.loc[g[flag]]

        if mode == "observed":
            n_total = len(g)
            n_recovered = int(rec.shape[0])
        else:
            entered = g.loc[g["entered_danger"]]
            n_total = len(entered)
            n_recovered = int(rec.shape[0])

        n_censored = n_total - n_recovered
        rate = n_recovered / n_total if n_total > 0 else np.nan

        post_peak = summarize_metric(rec, "time_peak_to_safe_s")
        total_safe = summarize_metric(rec, "time_to_safe_s")
        pp = _stat_lookup(post_peak, ind)
        ts = _stat_lookup(total_safe, ind)

        detail_rows.append(
            {
                "indicator": ind,
                "mode": mode,
                "n_total": n_total,
                "n_recovered": n_recovered,
                "n_censored": n_censored,
                "recovery_rate": rate,
                "post_peak_n": pp["n"],
                "post_peak_mean": pp["mean"],
                "post_peak_median": pp["median"],
                "post_peak_combined_fmt": pp["combined_fmt"],
                "total_safe_n": ts["n"],
                "total_safe_mean": ts["mean"],
                "total_safe_median": ts["median"],
                "total_safe_combined_fmt": ts["combined_fmt"],
                "interpretation": RECOVERY_INTERPRETATIONS.get(ind, ""),
            }
        )

    detail = pd.DataFrame(detail_rows)
    wide = pd.DataFrame(
        {
            "Indicator": INDICATOR_ORDER,
            "n_total": detail["n_total"].values,
            "n_recovered": detail["n_recovered"].values,
            "n_censored": detail["n_censored"].values,
            "recovery_rate": detail["recovery_rate"].map(
                lambda x: f"{100 * x:.1f}%" if pd.notna(x) else "NA"
            ),
            "Post-peak recovery duration, s": detail["post_peak_combined_fmt"].values,
            "Total time to safe, s": detail["total_safe_combined_fmt"].values,
            "Interpretation": detail["interpretation"].values,
        }
    )
    post_peak_stats = summarize_metric(
        pool.loc[pool[flag]], "time_peak_to_safe_s"
    )
    total_safe_stats = summarize_metric(pool.loc[pool[flag]], "time_to_safe_s")
    return detail, wide, post_peak_stats, total_safe_stats


def build_post_fire_recovery_summary(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Panel (d) and paper-table recovery stats from peak-to-recovery/post-fire records only."""
    pool = recovery_records(df)
    detail_rows: list[dict[str, Any]] = []
    for ind in INDICATOR_ORDER:
        g = pool.loc[pool["indicator"] == ind]
        rec = g.loc[g["valid_observed_recovery"]]
        n_total = len(g)
        n_recovered = int(rec.shape[0])
        n_censored = n_total - n_recovered
        rate = n_recovered / n_total if n_total > 0 else np.nan
        post_peak = summarize_metric(rec, "time_peak_to_safe_s")
        pp = _stat_lookup(post_peak, ind)
        detail_rows.append(
            {
                "indicator": ind,
                "n_total": n_total,
                "n_recovered": n_recovered,
                "n_censored": n_censored,
                "recovery_rate": rate,
                "post_peak_n": pp["n"],
                "post_peak_mean": pp["mean"],
                "post_peak_median": pp["median"],
                "post_peak_combined_fmt": pp["combined_fmt"],
                "interpretation": RECOVERY_INTERPRETATIONS.get(ind, ""),
            }
        )
    detail = pd.DataFrame(detail_rows)
    post_peak_stats = summarize_metric(pool.loc[pool["valid_observed_recovery"]], "time_peak_to_safe_s")
    return detail, post_peak_stats, detail


def build_paper_main_table(
    escalation_detail: pd.DataFrame,
    recovery_detail: pd.DataFrame,
) -> pd.DataFrame:
    esc = escalation_detail.set_index("indicator")
    rec = recovery_detail.set_index("indicator")
    rows: list[dict[str, Any]] = []
    for ind in INDICATOR_ORDER:
        e = esc.loc[ind] if ind in esc.index else {}
        r = rec.loc[ind] if ind in rec.index else {}
        extreme_col = (
            "Time to minimum transmittance, s"
            if ind == "Transmittance"
            else "Time to extreme state, s"
        )
        row: dict[str, Any] = {
            "Indicator": ind,
            "n_dev": int(e.get("n_first_danger", 0)),
            "n_peak": int(e.get("n_peak", 0)),
            "n_first_to_peak": int(e.get("n_first_to_peak", 0)),
            "First danger time, s": e.get("first_danger_combined_fmt", "NA"),
            "First danger to extreme state, s": e.get("first_to_peak_combined_fmt", "NA"),
            "n_recovery_total": int(r.get("n_total", 0)),
            "n_recovered": int(r.get("n_recovered", 0)),
            "n_censored": int(r.get("n_censored", 0)),
            "Observed post-peak recovery duration, s": r.get("post_peak_combined_fmt", "NA"),
            "Recovery rate": (
                f"{100 * float(r['recovery_rate']):.1f}%"
                if pd.notna(r.get("recovery_rate"))
                else "NA"
            ),
            "Interpretation": (
                f"{ESCALATION_INTERPRETATIONS.get(ind, '')}; "
                f"{RECOVERY_INTERPRETATIONS.get(ind, '')}"
            ).strip("; "),
        }
        row[extreme_col] = e.get("peak_combined_fmt", "NA")
        rows.append(row)
    return pd.DataFrame(rows)


def collect_panel_sample_counts(cleaned: pd.DataFrame) -> dict[str, dict[str, int]]:
    fp = cleaned.loc[cleaned["is_full_process"]]
    rec = recovery_records(cleaned)
    counts: dict[str, dict[str, int]] = {}
    for ind in INDICATOR_ORDER:
        g_fp = fp.loc[fp["indicator"] == ind]
        g_rec = rec.loc[rec["indicator"] == ind]
        counts[ind] = {
            "n_dev_first_danger": int(g_fp["valid_first_danger"].sum()),
            "n_dev_peak": int(g_fp["valid_peak"].sum()),
            "n_dev_first_to_peak": int(g_fp["valid_first_to_peak"].sum()),
            "n_recovery_total": len(g_rec),
            "n_recovered": int(g_rec["valid_observed_recovery"].sum()),
            "n_censored": len(g_rec) - int(g_rec["valid_observed_recovery"].sum()),
        }
    return counts


def verify_sample_consistency(
    panel_counts: dict[str, dict[str, int]],
    paper_table: pd.DataFrame,
    fig_counts: dict[str, dict[str, int]],
) -> None:
    """Raise if figure axis labels and summary table disagree on sample sizes."""
    mismatches: list[str] = []
    for ind in INDICATOR_ORDER:
        row = paper_table.loc[paper_table["Indicator"] == ind]
        if row.empty:
            continue
        r = row.iloc[0]
        fc = fig_counts[ind]
        pc = panel_counts[ind]
        pairs = [
            ("n_dev (panel a)", int(r["n_dev"]), fc["panel_a"], pc["n_dev_first_danger"]),
            ("n_peak (panel b)", int(r["n_peak"]), fc["panel_b"], pc["n_dev_peak"]),
            ("n_first_to_peak (panel c)", int(r["n_first_to_peak"]), fc["panel_c"], pc["n_dev_first_to_peak"]),
            ("n_recovered (panel d)", int(r["n_recovered"]), fc["panel_d"], pc["n_recovered"]),
            ("n_recovery_total", int(r["n_recovery_total"]), pc["n_recovery_total"], pc["n_recovery_total"]),
            ("n_censored", int(r["n_censored"]), pc["n_censored"], pc["n_censored"]),
        ]
        for label, table_n, fig_n, expected in pairs:
            if table_n != expected:
                mismatches.append(f"{ind} {label}: table={table_n}, expected={expected}")
            if label in {"n_recovery_total", "n_censored"}:
                continue
            if fig_n != expected:
                mismatches.append(f"{ind} {label}: figure={fig_n}, expected={expected}")
    if mismatches:
        raise ValueError(
            "Sample size mismatch between figure and summary table:\n  "
            + "\n  ".join(mismatches)
        )


def build_event_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Sample counts per indicator × evolution_role."""
    rows: list[dict[str, Any]] = []
    for role_label in ("full_process", "post_fire", "all"):
        if role_label == "all":
            sub = df
        else:
            sub = df.loc[df["evolution_role"] == role_label]
        for ind in INDICATOR_ORDER:
            g = sub.loc[sub["indicator"] == ind]
            rows.append(
                {
                    "evolution_role": role_label,
                    "indicator": ind,
                    "n_total": len(g),
                    "n_entered_danger": int(g["entered_danger"].sum()),
                    "n_recovered": int(g["recovered"].sum()),
                    "n_invalid_event_order": int(g["invalid_event_order"].sum()),
                    "n_right_censored": int((~g["recovered"]).sum()),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _indicator_data(
    df: pd.DataFrame,
    value_col: str,
    mask: pd.Series | None = None,
) -> dict[str, np.ndarray]:
    data: dict[str, np.ndarray] = {}
    sub = df if mask is None else df.loc[mask]
    for ind in INDICATOR_ORDER:
        vals = sub.loc[sub["indicator"] == ind, value_col].dropna().values.astype(float)
        data[ind] = vals
    return data


def _has_detection_window_floor(data: dict[str, np.ndarray], floor: float = 20.0) -> bool:
    """True when early-response indicators cluster at the detection-window floor."""
    check_inds = ("Temperature", "Heat flux", "Transmittance")
    for ind in check_inds:
        vals = data.get(ind, np.array([]))
        if len(vals) == 0:
            continue
        frac_at_floor = np.mean(np.isclose(vals, floor, atol=0.5))
        if frac_at_floor >= 0.35:
            return True
    return False


def legend_handles() -> list[Line2D]:
    """Proxy artists for the combined 2x2 figure legend."""
    return [
        Line2D([0], [0], color="#E69F00", lw=2.0, label="Median"),
        Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor="#2ca02c",
            markeredgecolor="#2ca02c",
            markersize=8,
            linestyle="None",
            label="Mean",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#bdbdbd",
            markeredgecolor="#bdbdbd",
            markersize=7,
            alpha=0.75,
            linestyle="None",
            label="Sample",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=6,
            linestyle="None",
            label="Outlier",
        ),
    ]


def plot_box_with_points(
    ax: plt.Axes,
    data_dict: dict[str, np.ndarray],
    ylabel: str,
    title: str,
    panel_label: str,
    footnote: str | None = None,
) -> None:
    positions = list(range(1, len(INDICATOR_ORDER) + 1))
    series = [data_dict.get(ind, np.array([])) for ind in INDICATOR_ORDER]

    ax.boxplot(
        series,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showmeans=True,
        **BOX_STYLE,
    )

    rng = np.random.default_rng(BOOT_RANDOM_STATE)
    for pos, vals in zip(positions, series):
        if len(vals) == 0:
            continue
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(
            pos + jitter,
            vals,
            color="lightgray",
            alpha=0.55,
            s=20,
            linewidths=0,
            zorder=3,
        )

    labels = []
    for ind, vals in zip(INDICATOR_ORDER, series):
        labels.append(f"{ind}\n(n={len(vals)})")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9, pad=6)
    ax.text(0.02, 0.98, panel_label, transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")

    if footnote:
        ax.text(
            0.02,
            0.02,
            footnote,
            transform=ax.transAxes,
            fontsize=6.5,
            va="bottom",
            color="#444444",
            wrap=True,
        )


def make_2x2_figure(df: pd.DataFrame, out_dir: Path) -> tuple[list[Path], dict[str, dict[str, int]]]:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fp = df.loc[df["is_full_process"]]
    rec = recovery_records(df)

    data_a = _indicator_data(fp, "time_to_first_danger_s", fp["valid_first_danger"])
    data_b = _indicator_data(fp, "time_to_peak_s", fp["valid_peak"])
    data_c = _indicator_data(fp, "time_first_danger_to_peak_s", fp["valid_first_to_peak"])
    data_d = _indicator_data(rec, "time_peak_to_safe_s", rec["valid_observed_recovery"])

    fig_counts: dict[str, dict[str, int]] = {}
    for ind in INDICATOR_ORDER:
        fig_counts[ind] = {
            "panel_a": len(data_a.get(ind, [])),
            "panel_b": len(data_b.get(ind, [])),
            "panel_c": len(data_c.get(ind, [])),
            "panel_d": len(data_d.get(ind, [])),
        }

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 7.2))

    foot_a = None
    if _has_detection_window_floor(data_a):
        foot_a = "Negative values indicate threshold exceedance before the selected plot origin."

    plot_box_with_points(
        axes[0, 0],
        data_a,
        "Time (s)",
        "Time to first danger",
        "(a)",
        footnote=foot_a,
    )
    plot_box_with_points(
        axes[0, 1],
        data_b,
        "Time (s)",
        "Time to extreme state",
        "(b)",
        footnote="For transmittance, the extreme state denotes the minimum transmittance.",
    )
    plot_box_with_points(
        axes[1, 0],
        data_c,
        "Time (s)",
        "First danger to extreme state",
        "(c)",
    )
    plot_box_with_points(
        axes[1, 1],
        data_d,
        "Time (s)",
        "Observed post-peak recovery duration",
        "(d)",
    )

    title = fig.suptitle(
        "Time-scale statistics of hazard escalation and recovery",
        fontsize=11,
        y=0.97,
    )
    caption_en = (
        "Panels (a)–(c) are calculated from full-process experiments, whereas panel (d) uses "
        "peak-to-recovery/post-fire records with observed safe recovery. Unrecovered records "
        "are treated as right-censored and are reported in the summary table."
    )
    caption_zh = (
        "图中 (a)–(c) 基于全过程实验统计危险建立与极值时间，(d) 基于峰后恢复/火后实验统计观测峰后恢复时间。"
        "未在观测窗口内恢复至安全范围的样本视为右删失样本，并在汇总表中报告。"
    )
    fig.text(0.5, 0.045, caption_en, ha="center", va="bottom", fontsize=6.5, color="#444444", wrap=True)
    fig.text(0.5, 0.015, caption_zh, ha="center", va="bottom", fontsize=6.5, color="#444444", wrap=True)

    fig.subplots_adjust(
        left=0.11,
        right=0.98,
        top=0.92,
        bottom=0.16,
        hspace=0.38,
        wspace=0.28,
    )

    leg = fig.legend(
        handles=legend_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.08),
        ncol=4,
        frameon=False,
        fontsize=8.5,
    )

    paths: list[Path] = []
    stem = fig_dir / FIG_OUTPUT_STEM
    for ext in (".png", ".svg", ".pdf"):
        p = stem.with_suffix(ext)
        fig.savefig(
            p,
            dpi=600 if ext == ".png" else None,
            bbox_inches="tight",
            pad_inches=0.10,
            bbox_extra_artists=[leg, title],
        )
        paths.append(p)
    plt.close(fig)
    return paths, fig_counts


# ---------------------------------------------------------------------------
# Export & narrative
# ---------------------------------------------------------------------------


def export_tables(
    cleaned: pd.DataFrame,
    escalation_detail: pd.DataFrame,
    escalation_wide: pd.DataFrame,
    recovery_obs_detail: pd.DataFrame,
    recovery_obs_wide: pd.DataFrame,
    recovery_danger_detail: pd.DataFrame,
    recovery_danger_wide: pd.DataFrame,
    recovery_post_fire_detail: pd.DataFrame,
    paper_main_table: pd.DataFrame,
    event_counts: pd.DataFrame,
    out_dir: Path,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    cleaned_csv = out_dir / "cleaned_evolution_event_table.csv"
    cleaned.to_csv(cleaned_csv, index=False, encoding="utf-8-sig")
    paths.append(cleaned_csv)

    esc_csv = out_dir / "full_process_escalation_summary.csv"
    escalation_detail.to_csv(esc_csv, index=False, encoding="utf-8-sig")
    paths.append(esc_csv)

    obs_csv = out_dir / "recovery_observed_summary.csv"
    recovery_obs_detail.to_csv(obs_csv, index=False, encoding="utf-8-sig")
    paths.append(obs_csv)

    pf_csv = out_dir / "recovery_post_fire_summary.csv"
    recovery_post_fire_detail.to_csv(pf_csv, index=False, encoding="utf-8-sig")
    paths.append(pf_csv)

    danger_csv = out_dir / "recovery_danger_only_summary.csv"
    recovery_danger_detail.to_csv(danger_csv, index=False, encoding="utf-8-sig")
    paths.append(danger_csv)

    xlsx = out_dir / PAPER_XLSX
    xlsx_alt = out_dir / "fire_evolution_timescale_summary_for_paper_new.xlsx"
    try:
        with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
            paper_main_table.to_excel(writer, sheet_name="Paper_Main_Table", index=False)
            escalation_wide.to_excel(writer, sheet_name="FullProcess_Escalation", index=False)
            recovery_post_fire_detail.to_excel(writer, sheet_name="Recovery_PostFire", index=False)
            recovery_obs_wide.to_excel(writer, sheet_name="Recovery_Observed_AllRoles", index=False)
            recovery_danger_wide.to_excel(writer, sheet_name="Recovery_DangerOnly", index=False)
            event_counts.to_excel(writer, sheet_name="Event_Counts", index=False)
            cleaned.to_excel(writer, sheet_name="Cleaned_Event_Table", index=False)
        paths.append(xlsx)
    except PermissionError:
        with pd.ExcelWriter(xlsx_alt, engine="openpyxl") as writer:
            paper_main_table.to_excel(writer, sheet_name="Paper_Main_Table", index=False)
            escalation_wide.to_excel(writer, sheet_name="FullProcess_Escalation", index=False)
            recovery_post_fire_detail.to_excel(writer, sheet_name="Recovery_PostFire", index=False)
            recovery_obs_wide.to_excel(writer, sheet_name="Recovery_Observed_AllRoles", index=False)
            recovery_danger_wide.to_excel(writer, sheet_name="Recovery_DangerOnly", index=False)
            event_counts.to_excel(writer, sheet_name="Event_Counts", index=False)
            cleaned.to_excel(writer, sheet_name="Cleaned_Event_Table", index=False)
        paths.append(xlsx_alt)

    legacy_xlsx = out_dir / "fire_evolution_timescale_summary.xlsx"
    try:
        with pd.ExcelWriter(legacy_xlsx, engine="openpyxl") as writer:
            escalation_wide.to_excel(writer, sheet_name="FullProcess_Escalation", index=False)
            recovery_obs_wide.to_excel(writer, sheet_name="Recovery_Observed", index=False)
            recovery_danger_wide.to_excel(writer, sheet_name="Recovery_DangerOnly", index=False)
            event_counts.to_excel(writer, sheet_name="Event_Counts", index=False)
            cleaned.to_excel(writer, sheet_name="Cleaned_Event_Table", index=False)
        paths.append(legacy_xlsx)
    except PermissionError:
        pass

    return paths


def _fmt_median_or_na(val: float, n: int) -> str:
    if n == 0 or not np.isfinite(val):
        return "NA"
    return f"{val:.0f}"


def write_conclusion_draft(
    escalation_detail: pd.DataFrame,
    recovery_obs_detail: pd.DataFrame,
    out_path: Path,
) -> Path:
    """Auto-generate a Chinese conclusion paragraph from summary tables."""
    esc = escalation_detail.set_index("indicator")

    def peak_med(ind: str) -> tuple[float, int]:
        if ind not in esc.index:
            return np.nan, 0
        row = esc.loc[ind]
        return float(row["peak_median"]), int(row["n_peak"])

    def ftp_med(ind: str) -> tuple[float, int]:
        if ind not in esc.index:
            return np.nan, 0
        row = esc.loc[ind]
        return float(row["first_to_peak_median"]), int(row["n_first_to_peak"])

    t_med, t_n = peak_med("Temperature")
    h_med, h_n = peak_med("Heat flux")
    c_med, c_n = peak_med("CO")
    tr_med, tr_n = peak_med("Transmittance")
    co_ftp, co_ftp_n = ftp_med("CO")

    rec = recovery_obs_detail.set_index("indicator")

    def rec_med(ind: str) -> tuple[float, int, int, float]:
        if ind not in rec.index:
            return np.nan, 0, 0, np.nan
        row = rec.loc[ind]
        return (
            float(row["post_peak_median"]),
            int(row["post_peak_n"]),
            int(row["n_censored"]),
            float(row["recovery_rate"]) if pd.notna(row["recovery_rate"]) else np.nan,
        )

    h_rec, h_rec_n, h_cen, h_rate = rec_med("Heat flux")
    c_rec, c_rec_n, c_cen, c_rate = rec_med("CO")
    t_rec, t_rec_n, t_cen, t_rate = rec_med("Temperature")
    tr_rec, tr_rec_n, tr_cen, tr_rate = rec_med("Transmittance")

    co_ftp_text = (
        _fmt_median_or_na(co_ftp, co_ftp_n)
        if co_ftp_n >= 3
        else "有限样本下难以稳定估计"
    )

    lines = [
        "在本文实验条件下，舱室火灾危险变量表现出明显的阶段性演化特征。"
        "温度和辐射热流在着火后较早达到高水平，透光率随后恶化至最低水平，"
        "而 CO 浓度的积累相对滞后，通常在更晚阶段达到峰值。"
        f"统计结果显示，Temperature、Heat flux、CO 和 Transmittance 的达峰时间"
        f"（full_process 样本）分别为 {_fmt_median_or_na(t_med, t_n)}、"
        f"{_fmt_median_or_na(h_med, h_n)}、{_fmt_median_or_na(c_med, c_n)} 和 "
        f"{_fmt_median_or_na(tr_med, tr_n)} s"
        f"（n={t_n}/{h_n}/{c_n}/{tr_n}）。"
        "其中 Transmittance 的“达峰时间”在物理上对应达到最低透光率的时刻。"
    ]

    if co_ftp_n >= 3:
        lines.append(
            f"进一步地，CO 从首次危险到峰值仍存在约 {co_ftp_text} s 的持续恶化过程，"
            "表明火灾发生后的前数分钟是人员疏散和应急营救的关键窗口。"
        )
    else:
        lines.append(
            "CO 从首次危险到峰值的持续恶化时间在本文 full_process 样本中较少，"
            "营救窗口判断宜结合个案曲线与预警模型输出综合评估。"
        )

    lines.append("")
    lines.append(
        "在火后恢复阶段，不同危险变量的恢复过程并不同步。"
        "Observed recovery 统计显示，"
    )

    rec_parts: list[str] = []
    for label, med, n_rec, n_cen, rate in (
        ("Heat flux", h_rec, h_rec_n, h_cen, h_rate),
        ("CO", c_rec, c_rec_n, c_cen, c_rate),
        ("Temperature", t_rec, t_rec_n, t_cen, t_rate),
        ("Transmittance", tr_rec, tr_rec_n, tr_cen, tr_rate),
    ):
        if n_rec >= 3 and np.isfinite(med):
            rec_parts.append(f"{label} 从峰值回落至安全范围的中位时间约为 {med:.0f} s（n={n_rec}）")
        elif n_rec > 0:
            rec_parts.append(
                f"{label} 的完整恢复在观测窗口内较少被捕捉（已恢复 n={n_rec}，"
                f"右删失 n={n_cen}，恢复率 {100 * rate:.0f}%）"
                if np.isfinite(rate)
                else f"{label} 的完整恢复在观测窗口内较少被捕捉（n={n_rec}）"
            )

    if rec_parts:
        lines.append("；".join(rec_parts) + "。")
    else:
        lines.append("各变量在观测窗口内均较少观测到完整的峰后恢复过程。")

    lines.append(
        "对于观测窗口结束时仍未恢复至安全范围的样本，应视为右删失样本，"
        "因此恢复时间统计主要基于已观测到恢复的样本，同时报告未恢复比例以避免低估危险持续时间。"
        "火后再进入或复航判断不宜仅依赖单一指标，而应综合考虑热暴露、能见度和毒性气体风险，"
        "并优先关注恢复率较低或右删失比例较高的变量。"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines) + "\n"
    out_path.write_text(text, encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    setup_matplotlib()

    ap = argparse.ArgumentParser(
        description="Fire evolution time-scale analysis from event-level CSV."
    )
    ap.add_argument(
        "--event_csv",
        type=str,
        default="outputs/evolution_analysis/evolution_event_table.csv",
        help="Input evolution event table CSV",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="outputs/evolution_analysis/final",
        help="Output directory for tables, figures, and narrative",
    )
    args = ap.parse_args()

    event_csv = Path(args.event_csv)
    if not event_csv.is_file():
        event_csv = _TRACK_ROOT / args.event_csv
    if not event_csv.is_file():
        raise FileNotFoundError(
            f"Event CSV not found: {args.event_csv}\n"
            "Generate it first:\n"
            "  python -m dual_track_analysis.evolution.run_extract\n"
            "Or run the full lightweight pipeline:\n"
            "  python -m dual_track_analysis.run_all"
        )

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        rel = str(out_dir).replace("\\", "/")
        if rel.startswith("dual_track_analysis/"):
            rel = rel[len("dual_track_analysis/") :]
        out_dir = _TRACK_ROOT / rel
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading event table: {event_csv}")
    raw = load_event_table(event_csv)
    cleaned = clean_event_table(raw)

    (
        escalation_detail,
        escalation_wide,
        _fd,
        _pk,
        _ftp,
    ) = build_escalation_summary(cleaned)

    recovery_obs_detail, recovery_obs_wide, _, _ = build_recovery_summary(
        cleaned, mode="observed"
    )
    recovery_post_fire_detail, _, _ = build_post_fire_recovery_summary(cleaned)
    recovery_danger_detail, recovery_danger_wide, _, _ = build_recovery_summary(
        cleaned, mode="danger_only"
    )
    paper_main_table = build_paper_main_table(escalation_detail, recovery_post_fire_detail)
    event_counts = build_event_counts(cleaned)
    panel_counts = collect_panel_sample_counts(cleaned)

    table_paths = export_tables(
        cleaned,
        escalation_detail,
        escalation_wide,
        recovery_obs_detail,
        recovery_obs_wide,
        recovery_danger_detail,
        recovery_danger_wide,
        recovery_post_fire_detail,
        paper_main_table,
        event_counts,
        out_dir,
    )

    fig_paths: list[Path] = []
    fig_paths_data, fig_counts = make_2x2_figure(cleaned, out_dir)
    fig_paths.extend(fig_paths_data)
    verify_sample_consistency(panel_counts, paper_main_table, fig_counts)

    conclusion_path = write_conclusion_draft(
        escalation_detail,
        recovery_post_fire_detail,
        out_dir / "evolution_conclusion_draft.txt",
    )

    n_fp = cleaned.loc[cleaned["is_full_process"], "experiment_id"].nunique()
    n_rec_exps = recovery_records(cleaned)["experiment_id"].nunique()

    print("\n=== Quality check (Fig. Y) ===")
    print(f"Number of full-process experiments used in Fig. Y(a–c): {n_fp}")
    print(f"Number of recovery/post-fire records used in Fig. Y(d): {n_rec_exps}")
    print(f"Time axis label: {time_axis_label(cleaned)}")
    for ind in INDICATOR_ORDER:
        pc = panel_counts[ind]
        esc_row = escalation_detail.loc[escalation_detail["indicator"] == ind].iloc[0]
        rec_row = recovery_post_fire_detail.loc[recovery_post_fire_detail["indicator"] == ind].iloc[0]
        peak_name = "median minimum" if ind == "Transmittance" else "median extreme"
        rec_med = rec_row.get("post_peak_median", np.nan)
        rec_str = f"{rec_med:.1f} s" if pd.notna(rec_med) else "NA"
        print(
            f"  {ind}: n_dev(first)={pc['n_dev_first_danger']}, "
            f"n_dev(peak)={pc['n_dev_peak']}, n_dev(first-to-peak)={pc['n_dev_first_to_peak']}, "
            f"n_recovered={pc['n_recovered']}, n_censored={pc['n_censored']}, "
            f"median first danger={esc_row['first_danger_median']:.1f} s, "
            f"{peak_name}={esc_row['peak_median']:.1f} s, "
            f"median post-peak recovery={rec_str}"
        )
    print("Output paths:")
    all_outputs = table_paths + fig_paths + [conclusion_path]
    for p in all_outputs:
        print(f"  {p.resolve()}")


if __name__ == "__main__":
    main()

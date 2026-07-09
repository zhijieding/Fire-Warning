#!/usr/bin/env python3
"""
从 meta.csv 划分预警池 / 演化池，写出 experiment_registry.csv。

Usage (from Fire_prediction/):
    python -m dual_track_analysis.common.registry
    python -m dual_track_analysis.common.registry --meta dataset_ready/unified_1hz/meta.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG = _ROOT / "dual_track_analysis" / "config.json"


def _load_config(config_path: Path | None) -> dict:
    p = config_path or _DEFAULT_CONFIG
    if p.is_file():
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return {}


def experiment_id_from_row(row: pd.Series) -> str:
    fn = str(row.get("file_name", ""))
    stem = fn.replace("_clean.csv", "").replace(".csv", "")
    if stem == "fire_merged_1hz_reference":
        return "10"
    return stem


def load_meta(meta_path: Path) -> pd.DataFrame:
    df = pd.read_csv(meta_path)
    df["experiment_id"] = df.apply(experiment_id_from_row, axis=1)
    return df


def split_pools(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """预警池 vs 演化池。"""
    warn_mask = df["usable_for_warning"].astype(str).str.lower().isin(("true", "1", "yes"))
    evo_mask = df["usable_for_evolution"].astype(str).str.lower().isin(("true", "1", "yes"))

    # 预警：完整过程（含 reference_long 作为第 10 号长实验）
    warn_df = df[warn_mask].copy()
    warn_df["track"] = "warning"

    # 演化：全部 usable_for_evolution（69 post + 10 full，meta 中 10_clean 仅演化）
    evo_df = df[evo_mask].copy()
    evo_df["track"] = "evolution"
    evo_df["evolution_role"] = evo_df["category"].map(
        lambda c: "full_process" if str(c) in ("complete_full", "reference_long") else "post_fire"
    )
    return warn_df, evo_df


def write_registry(
    meta_path: Path,
    out_dir: Path,
) -> pd.DataFrame:
    df = load_meta(meta_path)
    warn_df, evo_df = split_pools(df)

    out_dir.mkdir(parents=True, exist_ok=True)

    warn_path = out_dir / "warning_experiments.csv"
    evo_path = out_dir / "evolution_experiments.csv"
    reg_path = out_dir / "experiment_registry.csv"

    cols = [
        "experiment_id", "file_name", "category", "task_category_zh",
        "duration", "event_time_sec", "usable_for_warning", "usable_for_evolution",
    ]
    cols = [c for c in cols if c in df.columns]

    warn_df[cols + ["track"]].to_csv(warn_path, index=False)
    evo_cols = cols + ["track"]
    if "evolution_role" in evo_df.columns:
        evo_cols.append("evolution_role")
    evo_df[evo_cols].to_csv(evo_path, index=False)

    reg = pd.concat([warn_df.assign(primary_track="warning"), evo_df], ignore_index=True)
    reg.to_csv(reg_path, index=False)

    print(f"预警池: {len(warn_df)} 实验 → {warn_path}")
    print(f"演化池: {len(evo_df)} 实验 → {evo_path}")
    print(f"  full_process: {(evo_df.get('evolution_role') == 'full_process').sum() if 'evolution_role' in evo_df.columns else 'n/a'}")
    print(f"  post_fire:    {(evo_df.get('evolution_role') == 'post_fire').sum() if 'evolution_role' in evo_df.columns else 'n/a'}")
    return reg


def warning_ids(meta_path: Path | None = None) -> list[str]:
    p = meta_path or (_ROOT / "dataset_ready/unified_1hz/meta.csv")
    warn_df, _ = split_pools(load_meta(p))
    return sorted(warn_df["experiment_id"].astype(str).unique())


def evolution_ids(meta_path: Path | None = None) -> list[str]:
    p = meta_path or (_ROOT / "dataset_ready/unified_1hz/meta.csv")
    _, evo_df = split_pools(load_meta(p))
    return sorted(evo_df["experiment_id"].astype(str).unique())


def main() -> None:
    cfg = _load_config(None)
    ap = argparse.ArgumentParser(description="Build warning/evolution experiment registry from meta.csv")
    ap.add_argument("--meta", type=Path, default=_ROOT / cfg.get("meta_csv", "dataset_ready/unified_1hz/meta.csv"))
    ap.add_argument(
        "--out-dir", type=Path,
        default=_ROOT / "dual_track_analysis" / "outputs" / "registry",
    )
    args = ap.parse_args()
    write_registry(args.meta.resolve(), args.out_dir.resolve())


if __name__ == "__main__":
    main()

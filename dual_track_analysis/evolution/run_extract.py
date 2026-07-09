#!/usr/bin/env python3
"""
Build evolution event table from unified 1 Hz experiment CSVs.

Outputs:
  - ``outputs/evolution_analysis/evolution_event_table.csv`` (input for time-scale analysis)
  - ``outputs/evolution/key_moments_long.csv`` (long-format key moments, legacy compat)

Usage (from Fire_prediction/)::

    python -m dual_track_analysis.evolution.run_extract

    python -m dual_track_analysis.analysis.fire_evolution_timescale_analysis
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_TRACK_ROOT = Path(__file__).resolve().parents[1]
_ROOT = _TRACK_ROOT.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dual_track_analysis.common.csv_locator import unified_csv_for_experiment
from dual_track_analysis.evolution.key_moments import (
    VARIABLE_SPECS,
    build_cfg,
    extract_experiment_events,
    load_track_config,
)


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    if p.is_file():
        return p.resolve()
    for base in (_TRACK_ROOT, _ROOT):
        cand = base / path
        if cand.is_file():
            return cand.resolve()
    return (_TRACK_ROOT / path).resolve()


def build_event_table(
    registry_csv: Path,
    *,
    out_event_csv: Path,
    out_long_csv: Path | None = None,
) -> pd.DataFrame:
    reg = pd.read_csv(registry_csv)
    if "experiment_id" not in reg.columns:
        raise ValueError(f"{registry_csv} missing experiment_id column")

    cfg = build_cfg()
    rows: list[dict] = []
    skipped: list[str] = []

    for _, row in reg.iterrows():
        eid = str(row["experiment_id"])
        role = str(row.get("evolution_role", "post_fire"))
        event_t = row.get("event_time_sec", None)
        try:
            event_t = float(event_t) if pd.notna(event_t) else None
        except (TypeError, ValueError):
            event_t = None

        csv_path = unified_csv_for_experiment(eid)
        if csv_path is None or not csv_path.is_file():
            skipped.append(eid)
            continue

        rows.extend(
            extract_experiment_events(
                csv_path,
                experiment_id=eid,
                evolution_role=role,
                event_time_sec=event_t,
                cfg=cfg,
            )
        )

    if not rows:
        raise RuntimeError("No evolution events extracted; check registry and unified CSV paths.")

    df = pd.DataFrame(rows)
    out_event_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_event_csv, index=False, encoding="utf-8-sig")

    if out_long_csv is not None:
        long_rows = []
        for r in rows:
            long_rows.append(
                {
                    "experiment_id": r["experiment_id"],
                    "evolution_role": r["evolution_role"],
                    "variable": r["variable"],
                    "t_origin": r["t_origin"],
                    "t_first_danger": r["t_first_danger"],
                    "t_peak": r["t_peak"],
                    "t_safe": r["t_safe"],
                    "time_to_first_danger_s": r["time_to_first_danger_s"],
                    "time_to_peak_s": r["time_to_peak_s"],
                    "time_peak_to_safe_s": r["time_peak_to_safe_s"],
                    "entered_danger": r["entered_danger"],
                    "recovered": r["recovered"],
                }
            )
        out_long_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(long_rows).to_csv(out_long_csv, index=False, encoding="utf-8-sig")

    if skipped:
        print(f"Warning: skipped {len(skipped)} experiments (CSV not found): {skipped[:10]}{'...' if len(skipped) > 10 else ''}")

    print(f"Extracted {len(df)} events ({df['experiment_id'].nunique()} experiments × {len(VARIABLE_SPECS)} variables)")
    print(f"  → {out_event_csv}")
    if out_long_csv is not None:
        print(f"  → {out_long_csv}")
    return df


def main() -> None:
    track_cfg = load_track_config()
    ap = argparse.ArgumentParser(description="Extract evolution key moments → event table")
    ap.add_argument(
        "--registry",
        type=str,
        default="outputs/registry/evolution_experiments.csv",
        help="Evolution experiment registry CSV",
    )
    ap.add_argument(
        "--out-event-csv",
        type=str,
        default="outputs/evolution_analysis/evolution_event_table.csv",
        help="Event-level CSV for time-scale analysis",
    )
    ap.add_argument(
        "--out-long-csv",
        type=str,
        default="outputs/evolution/key_moments_long.csv",
        help="Long-format key moments (legacy)",
    )
    args = ap.parse_args()

    registry = _resolve(args.registry)
    if not registry.is_file():
        raise FileNotFoundError(
            f"Registry not found: {args.registry}\n"
            "Run first: python -m dual_track_analysis.common.registry"
        )

    out_event = _TRACK_ROOT / args.out_event_csv
    out_long = _TRACK_ROOT / args.out_long_csv
    build_event_table(registry, out_event_csv=out_event, out_long_csv=out_long)


if __name__ == "__main__":
    main()

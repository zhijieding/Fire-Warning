"""定位 unified_1hz 与 usable_csv 下的实验 CSV 路径。"""
from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


def unified_csv_for_experiment(
    experiment_id: str,
    unified_dir: Path | None = None,
) -> Path | None:
    """按 experiment_id 查找 unified 宽表。"""
    unified_dir = unified_dir or (_ROOT / "dataset_ready/unified_1hz")
    eid = str(experiment_id)

    candidates = [
        unified_dir / f"{eid}_clean_unified_1hz.csv",
        unified_dir / f"{eid}_unified_1hz.csv",
    ]

    for p in candidates:
        if p.is_file():
            return p
    for p in sorted(unified_dir.glob(f"*{eid}*_unified_1hz.csv")):
        if p.is_file():
            return p
    return None


def usable_csv_for_experiment(
    experiment_id: str,
    usable_dir: Path | None = None,
) -> Path | None:
    usable_dir = usable_dir or (_ROOT / "dataset_ready/usable_csv")
    eid = str(experiment_id)
    candidates = [
        usable_dir / f"{eid}_clean.csv",
        usable_dir / f"{eid}.csv",
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None

"""Shared constants and helpers for paper-facing evolution figures."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

VARIABLE_TO_INDICATOR: dict[str, str] = {
    "T_max": "Temperature",
    "Heat_max": "Heat flux",
    "CO": "CO",
    "Trans": "Transmittance",
}

INDICATOR_ORDER: list[str] = ["Temperature", "Heat flux", "CO", "Transmittance"]

RECOVERY_ROLES: tuple[str, ...] = ("post_fire", "peak_to_recovery")

PANEL_COLORS: dict[str, str] = {
    "Temperature": "#D73027",
    "Heat flux": "#B2182B",
    "CO": "#1A9850",
    "Transmittance": "#2166AC",
}

TIME_LABEL_RECORDING = "Time from recording start (s)"
TIME_LABEL_IGNITION = "Time since ignition (s)"


def _to_bool(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return False
    return str(val).strip().lower() in {"1", "true", "yes", "t", "y"}


def _to_float(val: Any) -> float:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return np.nan
    try:
        return float(val)
    except (TypeError, ValueError):
        return np.nan


def uses_ignition_origin(df: pd.DataFrame) -> bool:
    return "t_ignition" in df.columns and df["t_ignition"].notna().any()


def plot_time_origin_for_row(row: pd.Series) -> float:
    if "t_ignition" in row.index and pd.notna(row.get("t_ignition")):
        return _to_float(row["t_ignition"])
    if "t_plot_origin" in row.index and pd.notna(row.get("t_plot_origin")):
        return _to_float(row["t_plot_origin"])
    return 0.0


def time_axis_label(df: pd.DataFrame) -> str:
    return TIME_LABEL_IGNITION if uses_ignition_origin(df) else TIME_LABEL_RECORDING


def select_full_process_df(df: pd.DataFrame) -> pd.DataFrame:
    """All rows belonging to full-process experiments (case-insensitive role)."""
    role = df["evolution_role"].astype(str).str.strip().str.lower()
    return df.loc[role.eq("full_process")].copy()


def full_process_experiment_ids(df: pd.DataFrame) -> list[str]:
    fp = select_full_process_df(df)
    if fp.empty:
        return []
    return sorted(fp["experiment_id"].astype(str).unique().tolist(), key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x))


def recovery_records(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df["evolution_role"].astype(str).isin(RECOVERY_ROLES)].copy()


def min_valid_samples(n_full_process: int) -> int:
    return max(3, int(np.ceil(0.5 * n_full_process)))


def safe_threshold_valid(direction: str, danger: float, safe: float) -> bool:
    if not (np.isfinite(danger) and np.isfinite(safe)):
        return False
    if direction == "down":
        return safe > danger
    return safe < danger

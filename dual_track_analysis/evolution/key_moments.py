"""
Per-variable fire-evolution key-moment extraction from unified 1 Hz CSV curves.

Used by ``run_extract`` to build ``evolution_event_table.csv`` for time-scale analysis.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import Config
from data_pipeline.labels import _first_sustained_above, _first_sustained_below

_TRACK_ROOT = Path(__file__).resolve().parents[1]
_ROOT = _TRACK_ROOT.parent
_DEFAULT_CONFIG = _TRACK_ROOT / "config.json"


def _hazard_window_params(cfg: Config) -> tuple[int, int]:
    """Return (window_seconds, min_hits) for fraction-based hazard onset."""
    window = int(cfg.sustain_seconds)
    frac = float(getattr(cfg, "hazard_window_fraction", 0.8))
    need = max(1, int(np.ceil(window * frac)))
    return window, need


def _metric_hazard_mask_fraction(
    time: np.ndarray,
    values: np.ndarray,
    threshold: float,
    min_start: int,
    window: int,
    need: int,
    *,
    above: bool,
) -> np.ndarray:
    """
    hazard_mask(t)=1 when the last ``window`` seconds contain at least ``need``
    samples meeting the threshold (>= for above; <= for below).
    """
    t = np.asarray(time, dtype=float)
    v = np.asarray(values, dtype=float)
    T = len(v)
    mask = np.zeros(T, dtype=np.int32)
    if window <= 0 or T < window:
        return mask

    for i in range(window - 1, T):
        if float(t[i]) < float(min_start):
            continue
        seg = v[i - window + 1 : i + 1]
        seg = seg[np.isfinite(seg)]
        if seg.size == 0:
            continue
        hits = (seg >= threshold) if above else (seg <= threshold)
        if int(np.sum(hits)) >= need:
            mask[i] = 1
    return mask


def _first_active_time(mask: np.ndarray, time: np.ndarray, min_start: int) -> float:
    t = np.asarray(time, dtype=float)
    m = np.asarray(mask, dtype=np.int32)
    for i in range(len(m)):
        if m[i] == 1 and float(t[i]) >= float(min_start):
            return float(t[i])
    return float("nan")


def _temp_series_for_hazard(df: pd.DataFrame) -> np.ndarray:
    """Prefer max layer mean; fallback to overall max temperature series."""
    lm = [c for c in ("LayerMean_1", "LayerMean_2", "LayerMean_3", "LayerMean_4") if c in df.columns]
    if lm:
        return df[lm].astype(float).max(axis=1).values
    return _t_max(df)


def hazard_onset_time_from_unified(df: pd.DataFrame, cfg: Config) -> float:
    """
    Hazard-process start time aligned with warning hazard definition:
    in a rolling window of length ``sustain_seconds``, at least 80% points
    meet the hazard threshold for any of the four indicators (OR).
    """
    if "time" not in df.columns:
        return float("nan")

    time = df["time"].astype(float).values
    min_s = int(cfg.min_start_time)
    window, need = _hazard_window_params(cfg)

    candidates: list[float] = []

    if "CO" in df.columns:
        co = df["CO"].astype(float).values
        m = _metric_hazard_mask_fraction(time, co, float(cfg.co_threshold), min_s, window, need, above=True)
        candidates.append(_first_active_time(m, time, min_s))

    if "Trans" in df.columns:
        trans = df["Trans"].astype(float).values
        m = _metric_hazard_mask_fraction(time, trans, float(cfg.trans_threshold), min_s, window, need, above=False)
        candidates.append(_first_active_time(m, time, min_s))

    heat = _heat_max(df)
    if np.isfinite(heat).any():
        m = _metric_hazard_mask_fraction(time, heat, float(cfg.heat_threshold), min_s, window, need, above=True)
        candidates.append(_first_active_time(m, time, min_s))

    temp = _temp_series_for_hazard(df)
    if np.isfinite(temp).any():
        m = _metric_hazard_mask_fraction(time, temp, float(cfg.temp_threshold), min_s, window, need, above=True)
        candidates.append(_first_active_time(m, time, min_s))

    valid = [c for c in candidates if np.isfinite(c)]
    return float(min(valid)) if valid else float("nan")


@dataclass(frozen=True)
class VariableSpec:
    variable: str
    label: str
    direction: str  # "up" | "down"
    danger_threshold: float | None  # None → relative to peak (T_max, Heat_max)
    safe_threshold: float | None


VARIABLE_SPECS: tuple[VariableSpec, ...] = (
    VariableSpec("T_max", "温度最大值", "up", None, None),
    VariableSpec("Heat_max", "热流最大值", "up", None, None),
    VariableSpec("CO", "CO", "up", None, None),
    VariableSpec("Trans", "透光率", "down", None, None),
)


def load_track_config(path: Path | None = None) -> dict[str, Any]:
    p = path or _DEFAULT_CONFIG
    if p.is_file():
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return {}


def build_cfg(track_cfg: dict[str, Any] | None = None) -> Config:
    """Merge dual_track_analysis/config.json into global Config."""
    track_cfg = track_cfg or load_track_config()
    cfg = Config()
    if "smooth_window" in track_cfg:
        cfg.smooth_window = int(track_cfg["smooth_window"])
    if "sustain_seconds" in track_cfg:
        cfg.sustain_seconds = int(track_cfg["sustain_seconds"])
    if "min_start_time" in track_cfg:
        cfg.min_start_time = int(track_cfg["min_start_time"])
    if "co_hazard_ppm" in track_cfg:
        cfg.co_threshold = float(track_cfg["co_hazard_ppm"])
    if "trans_hazard" in track_cfg:
        cfg.trans_threshold = float(track_cfg["trans_hazard"])
    if "co_safe_ppm" in track_cfg:
        cfg.co_safe_ppm = float(track_cfg["co_safe_ppm"])
    if "k_safe" in track_cfg:
        cfg.k_safe = float(track_cfg["k_safe"])
    if "temp_decay_frac" in track_cfg:
        cfg.temp_decay_frac = float(track_cfg["temp_decay_frac"])
    return cfg


def _smooth(y: np.ndarray, window: int) -> np.ndarray:
    w = max(1, int(window))
    return (
        pd.Series(np.asarray(y, dtype=float))
        .rolling(w, min_periods=1, center=True)
        .mean()
        .values
    )


def _heat_max(df: pd.DataFrame) -> np.ndarray:
    if "Heat_max" in df.columns:
        return df["Heat_max"].astype(float).values
    cols = [c for c in df.columns if c.startswith("Heat_") and not c.endswith("_mask")]
    if not cols:
        return np.full(len(df), np.nan)
    return df[cols].astype(float).max(axis=1).values


def _t_max(df: pd.DataFrame) -> np.ndarray:
    for name in ("温度最大值", "T_max"):
        if name in df.columns:
            return df[name].astype(float).values
    cols = [c for c in df.columns if c.startswith("T_m") and not c.endswith("_mask")]
    if not cols:
        return np.full(len(df), np.nan)
    return df[cols].astype(float).max(axis=1).values


def _series_for_variable(df: pd.DataFrame, variable: str) -> np.ndarray:
    if variable == "CO":
        return df["CO"].astype(float).values if "CO" in df.columns else np.full(len(df), np.nan)
    if variable == "Trans":
        if "Trans" in df.columns:
            return df["Trans"].astype(float).values
        if "k" in df.columns:
            return df["k"].astype(float).values
        return np.full(len(df), np.nan)
    if variable == "Heat_max":
        return _heat_max(df)
    if variable == "T_max":
        return _t_max(df)
    raise ValueError(f"Unknown variable: {variable}")


def _danger_threshold(spec: VariableSpec, cfg: Config) -> float:
    if spec.variable == "CO":
        return float(cfg.co_threshold)
    if spec.variable == "Trans":
        return float(cfg.trans_threshold)
    if spec.variable == "T_max":
        return float(cfg.temp_threshold)
    if spec.variable == "Heat_max":
        return float(cfg.heat_threshold)
    raise ValueError(spec.variable)


def _absolute_safe_threshold(spec: VariableSpec, cfg: Config) -> float | None:
    if spec.variable == "CO":
        return float(getattr(cfg, "co_safe_ppm", 50.0))
    if spec.variable == "Trans":
        return float(getattr(cfg, "k_safe", 0.7))
    return None


def _recovery_settings(track_cfg: dict[str, Any], cfg: Config) -> dict[str, float | int]:
    """
    Recovery-specific thresholds (separate from hazard-onset sustain/decay).

    Falls back to legacy keys when recovery_* are absent.
    """
    hazard_decay = float(track_cfg.get("temp_decay_frac", getattr(cfg, "temp_decay_frac", 0.35)))
    hazard_sustain = int(cfg.sustain_seconds)
    return {
        "sustain": int(track_cfg.get("recovery_sustain_seconds", hazard_sustain)),
        "decay_frac": float(track_cfg.get("recovery_decay_frac", hazard_decay)),
        "co_safe_ppm": float(
            track_cfg.get("recovery_co_safe_ppm", getattr(cfg, "co_safe_ppm", 50.0))
        ),
        "k_safe": float(track_cfg.get("recovery_k_safe", getattr(cfg, "k_safe", 0.7))),
    }


def _peak_time(
    time: np.ndarray,
    y_smooth: np.ndarray,
    t_origin: float,
    direction: str,
) -> tuple[float, float]:
    """Return (t_peak, value_at_peak) on smoothed curve for t >= t_origin."""
    t = np.asarray(time, dtype=float)
    y = np.asarray(y_smooth, dtype=float)
    mask = np.isfinite(y) & (t >= float(t_origin))
    if not np.any(mask):
        mask = np.isfinite(y)
    if not np.any(mask):
        return np.nan, np.nan
    idx = np.where(mask)[0]
    if direction == "down":
        i = idx[int(np.nanargmin(y[mask]))]
    else:
        i = idx[int(np.nanargmax(y[mask]))]
    return float(t[i]), float(y[i])


def _first_sustained_safe_after_peak(
    time: np.ndarray,
    values: np.ndarray,
    peak_t: float,
    sustain: int,
    *,
    safe_threshold: float,
    above: bool,
) -> float:
    """First time after peak where values stay on the safe side for ``sustain`` seconds."""
    t = np.asarray(time, dtype=float)
    v = np.asarray(values, dtype=float)
    if not np.isfinite(peak_t):
        return np.nan
    peak_i = int(np.argmin(np.abs(t - peak_t)))
    count = 0
    start_idx = -1
    for i in range(peak_i, len(t)):
        if not np.isfinite(v[i]):
            count = 0
            start_idx = -1
            continue
        ok = (v[i] >= safe_threshold) if above else (v[i] <= safe_threshold)
        if ok:
            if count == 0:
                start_idx = i
            count += 1
            if count >= sustain:
                return float(t[start_idx])
        else:
            count = 0
            start_idx = -1
    return np.nan


def extract_variable_events(
    time: np.ndarray,
    values: np.ndarray,
    spec: VariableSpec,
    cfg: Config,
    *,
    t_origin: float,
    evolution_role: str,
    experiment_id: str,
) -> dict[str, Any]:
    """Extract one row of the evolution event table for a single variable."""
    track_cfg = load_track_config()
    smooth_w = int(track_cfg.get("smooth_window", cfg.smooth_window))
    hazard_sustain = int(cfg.sustain_seconds)
    min_t = int(cfg.min_start_time)
    recovery = _recovery_settings(track_cfg, cfg)

    y_smooth = _smooth(values, smooth_w)
    danger_thr = _danger_threshold(spec, cfg)

    if spec.direction == "down":
        t_first = _first_sustained_below(time, y_smooth, danger_thr, min_t, hazard_sustain)
    else:
        t_first = _first_sustained_above(time, y_smooth, danger_thr, min_t, hazard_sustain)

    t_peak, peak_val = _peak_time(time, y_smooth, t_origin, spec.direction)

    recovery_sustain = int(recovery["sustain"])
    if spec.variable in ("T_max", "Heat_max"):
        row_safe_thr = (
            float(peak_val * recovery["decay_frac"]) if np.isfinite(peak_val) else np.nan
        )
        t_safe = _first_sustained_safe_after_peak(
            time,
            y_smooth,
            t_peak,
            recovery_sustain,
            safe_threshold=row_safe_thr,
            above=False,
        )
    elif spec.variable == "Trans":
        row_safe_thr = float(recovery["k_safe"])
        t_safe = _first_sustained_safe_after_peak(
            time,
            y_smooth,
            t_peak,
            recovery_sustain,
            safe_threshold=row_safe_thr,
            above=True,
        )
    else:
        row_safe_thr = float(recovery["co_safe_ppm"])
        t_safe = _first_sustained_safe_after_peak(
            time,
            y_smooth,
            t_peak,
            recovery_sustain,
            safe_threshold=row_safe_thr,
            above=False,
        )

    entered = bool(np.isfinite(t_first))
    recovered = bool(np.isfinite(t_safe) and np.isfinite(t_peak) and t_safe > t_peak)

    invalid = False
    if entered and np.isfinite(t_peak) and t_first > t_peak:
        invalid = True
    if recovered and t_safe <= t_peak:
        invalid = True
    if entered and recovered and np.isfinite(t_first) and t_safe < t_first:
        invalid = True

    def _rel(t_abs: float) -> float:
        if not np.isfinite(t_abs) or not np.isfinite(t_origin):
            return np.nan
        return float(t_abs - t_origin)

    time_to_first = _rel(t_first) if entered else np.nan
    time_to_peak = _rel(t_peak) if np.isfinite(t_peak) else np.nan
    time_to_safe = _rel(t_safe) if recovered else np.nan
    time_peak_to_safe = (
        float(t_safe - t_peak) if recovered and np.isfinite(t_safe) and np.isfinite(t_peak) else np.nan
    )
    time_danger_duration = (
        float(t_safe - t_first)
        if recovered and entered and np.isfinite(t_safe) and np.isfinite(t_first)
        else np.nan
    )

    return {
        "variable": spec.variable,
        "label": spec.label,
        "direction": spec.direction,
        "danger_threshold": danger_thr,
        "safe_threshold": row_safe_thr,
        "t_origin": float(t_origin),
        "t_evolution_origin": float(t_origin),
        "t_first_danger": t_first if entered else np.nan,
        "t_peak": t_peak if np.isfinite(t_peak) else np.nan,
        "t_safe": t_safe if recovered else np.nan,
        "time_to_first_danger_s": time_to_first,
        "time_to_peak_s": time_to_peak,
        "time_to_safe_s": time_to_safe,
        "time_peak_to_safe_s": time_peak_to_safe if recovered else np.nan,
        "time_danger_duration_s": time_danger_duration,
        "entered_danger": entered,
        "recovered": recovered,
        "invalid_event_order": invalid,
        "experiment_id": str(experiment_id),
        "evolution_role": str(evolution_role),
    }


def extract_experiment_events(
    csv_path: Path,
    *,
    experiment_id: str,
    evolution_role: str,
    event_time_sec: float | None,
    cfg: Config | None = None,
) -> list[dict[str, Any]]:
    cfg = cfg or build_cfg()
    df = pd.read_csv(csv_path)
    if "time" not in df.columns:
        raise ValueError(f"Missing time column: {csv_path}")

    time = df["time"].astype(float).values
    # Align evolution origin with warning hazard onset time (4-indicator OR, fraction-window),
    # i.e., the start time of the hazard process as defined by hazard(t).
    t_origin = hazard_onset_time_from_unified(df, cfg)
    if not np.isfinite(t_origin):
        t_origin = float(time[0])

    rows: list[dict[str, Any]] = []
    for spec in VARIABLE_SPECS:
        values = _series_for_variable(df, spec.variable)
        rows.append(
            extract_variable_events(
                time,
                values,
                spec,
                cfg,
                t_origin=t_origin,
                evolution_role=evolution_role,
                experiment_id=experiment_id,
            )
        )
    return rows

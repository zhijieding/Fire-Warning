"""
Step 2  – Label generation
  2.1  hazard(t)  – binary danger state per timestep
  2.2  warn_Δ(t)  – whether danger happens within next Δ seconds
  2.3  event_time & lead-time utilities
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from config import Config


def _hazard_window_params(cfg: Config) -> tuple[int, int]:
    """Return (window_seconds, min_hits) for fraction-based hazard."""
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
    above: bool = True,
) -> np.ndarray:
    """
    hazard(t)=1 when the last ``window`` seconds contain at least ``need``
    samples (>=80% of window) meeting the threshold.
    """
    T = len(values)
    mask = np.zeros(T, dtype=np.int32)
    if window <= 0 or T < window:
        return mask

    for i in range(window - 1, T):
        if float(time[i]) < min_start:
            continue
        seg = values[i - window + 1 : i + 1]
        if above:
            hits = seg >= threshold
        else:
            hits = seg <= threshold
        if int(np.sum(hits)) >= need:
            mask[i] = 1
    return mask


def _first_active_time(
    mask: np.ndarray,
    time: np.ndarray,
    min_start: int,
) -> float:
    """First timestep where mask is active and t >= min_start."""
    for i in range(len(mask)):
        if mask[i] == 1 and float(time[i]) >= min_start:
            return float(time[i])
    return np.nan


def _first_sustained_above(
    time: np.ndarray,
    values: np.ndarray,
    threshold: float,
    min_start: int,
    sustain: int,
    *,
    strict: bool = False,
) -> float:
    """
    Return the time of the first instant where `values` stays
    > (strict) or >= `threshold` for at least `sustain` consecutive seconds,
    beginning no earlier than `min_start` seconds.
    """
    count = 0
    start_idx = -1
    for i in range(len(time)):
        t, v = float(time[i]), float(values[i])
        ok = (v > threshold) if strict else (v >= threshold)
        if t >= min_start and ok:
            if count == 0:
                start_idx = i
            count += 1
            if count >= sustain:
                return float(time[start_idx])
        else:
            count = 0
            start_idx = -1
    return np.nan


def _first_sustained_below(
    time: np.ndarray,
    values: np.ndarray,
    threshold: float,
    min_start: int,
    sustain: int,
) -> float:
    """First sustained crossing where values stay <= threshold."""
    count = 0
    start_idx = -1
    for i in range(len(time)):
        t, v = float(time[i]), float(values[i])
        if t >= min_start and v <= threshold:
            if count == 0:
                start_idx = i
            count += 1
            if count >= sustain:
                return float(time[start_idx])
        else:
            count = 0
            start_idx = -1
    return np.nan


def _layer_temp_series(layer_domain: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Max layer-mean temperature across layers at each timestep."""
    if layer_domain is None:
        return None
    return layer_domain[:, :, 0].max(axis=1).astype(np.float64)


def _heat_max_series(smoke: np.ndarray, col_map: dict, cfg: Config) -> Optional[np.ndarray]:
    heat_vals = []
    for hc in cfg.heat_col_candidates:
        if hc in col_map:
            heat_vals.append(smoke[:, col_map[hc]])
    if not heat_vals:
        return None
    return np.stack(heat_vals, axis=0).max(axis=0).astype(np.float64)


def compute_hazard(
    time: np.ndarray,
    smoke: np.ndarray,
    smoke_col_names: list,
    cfg: Config,
    layer_domain: Optional[np.ndarray] = None,
    *,
    temp_field: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Return a per-timestep binary hazard vector (T,) int32.

    并联判据：层域温度、热流、CO、Trans 任一指标在长度 sustain_seconds 的
    滚动窗口内，至少 hazard_window_fraction（默认 80%）采样点达阈，则 hazard=1。
      层域温度 max(layer_mean) >= temp_threshold
      热流 max(q) >= heat_threshold
      CO >= co_threshold
      Trans <= trans_threshold
    """
    T = len(time)
    hazard = np.zeros(T, dtype=np.int32)
    col_map = {name: i for i, name in enumerate(smoke_col_names)}
    min_s = cfg.min_start_time
    window, need = _hazard_window_params(cfg)

    if "CO" in col_map:
        hazard |= _metric_hazard_mask_fraction(
            time, smoke[:, col_map["CO"]], cfg.co_threshold, min_s, window, need,
            above=True,
        )
    if "Trans" in col_map:
        hazard |= _metric_hazard_mask_fraction(
            time, smoke[:, col_map["Trans"]], cfg.trans_threshold, min_s, window, need,
            above=False,
        )

    layer_t = _layer_temp_series(layer_domain)
    if layer_t is None and temp_field is not None:
        layer_t = temp_field.max(axis=1).astype(np.float64)
    if layer_t is not None:
        hazard |= _metric_hazard_mask_fraction(
            time, layer_t, cfg.temp_threshold, min_s, window, need, above=True,
        )

    if cfg.heat_threshold is not None:
        heat_max = _heat_max_series(smoke, col_map, cfg)
        if heat_max is not None:
            hazard |= _metric_hazard_mask_fraction(
                time, heat_max, cfg.heat_threshold, min_s, window, need, above=True,
            )

    return hazard


def _metric_event_candidates(
    time: np.ndarray,
    smoke: np.ndarray,
    smoke_col_names: list,
    cfg: Config,
    layer_domain: Optional[np.ndarray],
    metrics: list[str],
    *,
    temp_field: Optional[np.ndarray] = None,
) -> list[float]:
    col_map = {name: i for i, name in enumerate(smoke_col_names)}
    min_s = cfg.min_start_time
    window, need = _hazard_window_params(cfg)
    candidates = []

    if "CO" in metrics and "CO" in col_map:
        m = _metric_hazard_mask_fraction(
            time, smoke[:, col_map["CO"]], cfg.co_threshold, min_s, window, need,
            above=True,
        )
        candidates.append(_first_active_time(m, time, min_s))

    if "Trans" in metrics and "Trans" in col_map:
        m = _metric_hazard_mask_fraction(
            time, smoke[:, col_map["Trans"]], cfg.trans_threshold, min_s, window, need,
            above=False,
        )
        candidates.append(_first_active_time(m, time, min_s))

    if "T" in metrics:
        layer_t = _layer_temp_series(layer_domain)
        if layer_t is None and temp_field is not None:
            layer_t = temp_field.max(axis=1).astype(np.float64)
        if layer_t is not None:
            m = _metric_hazard_mask_fraction(
                time, layer_t, cfg.temp_threshold, min_s, window, need, above=True,
            )
            candidates.append(_first_active_time(m, time, min_s))

    if "Heat" in metrics and cfg.heat_threshold is not None:
        heat_max = _heat_max_series(smoke, col_map, cfg)
        if heat_max is not None:
            m = _metric_hazard_mask_fraction(
                time, heat_max, cfg.heat_threshold, min_s, window, need, above=True,
            )
            candidates.append(_first_active_time(m, time, min_s))

    return candidates


def find_event_time(
    time: np.ndarray,
    smoke: np.ndarray,
    smoke_col_names: list,
    cfg: Config,
    layer_domain: Optional[np.ndarray] = None,
    *,
    temp_field: Optional[np.ndarray] = None,
    metrics: Optional[list[str]] = None,
) -> float:
    """
    Return event_time = earliest fraction-window hazard onset among selected metrics.

    Default uses ``cfg.event_time_metrics`` (CO+Trans) for warning-anchor timing.
    """
    use = metrics if metrics is not None else list(cfg.event_time_metrics)
    candidates = _metric_event_candidates(
        time, smoke, smoke_col_names, cfg, layer_domain, use, temp_field=temp_field,
    )
    valid = [c for c in candidates if not np.isnan(c)]
    return min(valid) if valid else np.nan


def find_hazard_onset_time(
    time: np.ndarray,
    smoke: np.ndarray,
    smoke_col_names: list,
    cfg: Config,
    layer_domain: Optional[np.ndarray] = None,
    *,
    temp_field: Optional[np.ndarray] = None,
) -> float:
    """Earliest fraction-window hazard onset across all four criteria."""
    return find_event_time(
        time, smoke, smoke_col_names, cfg, layer_domain,
        temp_field=temp_field,
        metrics=["CO", "Trans", "T", "Heat"],
    )


def compute_warning_label(
    time: np.ndarray,
    event_time: float,
    delta: int,
) -> np.ndarray:
    """
    warn_Δ(t) = 1  if  0 < (event_time - t) <= Δ  (danger coming within Δ s,
    but hasn't arrived yet).
    """
    warn = np.zeros(len(time), dtype=np.int32)
    if np.isnan(event_time):
        return warn
    tte = event_time - time  # time-to-event
    warn[(tte > 0) & (tte <= delta)] = 1
    return warn


# ───────────────── convenience: label one experiment ─────────────────

def label_one_experiment(exp: Dict, cfg: Config) -> Dict:
    """
    Add hazard / event_time / warn fields to an experiment dict (in-place).
    """
    time = exp["time"]
    smoke = exp["smoke"]
    layer_domain = exp.get("layer_domain")
    temp_field = exp.get("temp_field")
    names = (
        exp["smoke_col_names"]
        if isinstance(exp.get("smoke_col_names"), list)
        else cfg.smoke_col_names
    )

    exp["hazard"] = compute_hazard(
        time, smoke, names, cfg, layer_domain, temp_field=temp_field,
    )
    exp["hazard_onset_time"] = find_hazard_onset_time(
        time, smoke, names, cfg, layer_domain, temp_field=temp_field,
    )
    exp["event_time"] = find_event_time(
        time, smoke, names, cfg, layer_domain, temp_field=temp_field,
    )
    exp["warn"] = compute_warning_label(time, exp["event_time"], cfg.warning_delta)

    return exp


# ───────────────── lead-time helpers ─────────────────

def compute_lead_time(
    first_alarm_time: float,
    event_time: float,
) -> float:
    if np.isnan(first_alarm_time) or np.isnan(event_time):
        return np.nan
    return event_time - first_alarm_time


# backward-compat alias used by analysis scripts (>= threshold)
_first_sustained_cross_time = _first_sustained_above


def lead_time_statistics(lead_times: list) -> Dict:
    lt = np.array([x for x in lead_times if not np.isnan(x)])
    if len(lt) == 0:
        return dict(mean=np.nan, median=np.nan, ge30=np.nan, ge60=np.nan, n=0)
    return dict(
        mean=float(lt.mean()),
        median=float(np.median(lt)),
        ge30=float((lt >= 30).mean()),
        ge60=float((lt >= 60).mean()),
        n=len(lt),
    )

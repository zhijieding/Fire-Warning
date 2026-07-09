"""
Evaluation script for TriModalFireModel.

5.1  Prediction metrics:  MAE / RMSE / peak error / phase error
5.2  Warning metrics:     AUC / F1 / MCC / Recall / Lead time
5.3  Temperature scaling: post-hoc probability calibration
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_auc_score, average_precision_score,
)

from analysis.kfold_export import export_one_fold_timesteps
from analysis.warn_prediction_io import (
    fold_index_from_dir_name,
    prediction_model_subdir,
    predictions_root,
    save_fold_warn_predictions_from_meta,
)
from checkpoint_io import load_torch_checkpoint
from config import Config
from data_pipeline.dataset import build_dataloaders, FirePredictionDataset
from models import build_model
from models.model import TriModalFireModel


def _json_sanitize(obj: Any) -> Any:
    """Make eval_summary JSON-compliant (NaN/Inf → null, numpy scalars → Python)."""
    if isinstance(obj, np.generic):
        if isinstance(obj, np.floating):
            v = float(obj)
            return None if (math.isnan(v) or math.isinf(v)) else v
        return obj.item()
    if isinstance(obj, dict):
        return {k: _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


# ═══════════════════ 5.3  Temperature Scaling ═══════════════════

class _TemperatureScaling(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        # Clamp to keep the optimization stable (avoid division by ~0 producing NaNs/Inf).
        temp = self.temperature.clamp(min=1e-3, max=100.0)
        return logits / temp


def calibrate_temperature(
    model: TriModalFireModel,
    val_loader: DataLoader,
    device: str,
) -> float:
    """
    Learn a single temperature parameter on the validation set so that
    sigmoid(logits / T) is well-calibrated.  Returns the optimal T.
    Only rows with warn_valid=1 are used (same as warning-head training).
    """
    model.eval()
    all_logits, all_labels = [], []

    with torch.no_grad():
        for batch in val_loader:
            temp = batch["temp_field"].to(device)
            smoke = batch["smoke"].to(device)
            ld = batch["layer_domain"].to(device)
            msk_t = batch["mask_temp"].to(device)
            msk_s = batch["mask_smoke"].to(device)
            outputs = model(temp, ld, smoke, msk_t, msk_s)
            wl = outputs["warn_logits"].cpu()
            y = batch["warn"].cpu()
            wv = batch.get("warn_valid")
            if wv is not None:
                m = wv.cpu().bool()
                all_logits.append(wl[m])
                all_labels.append(y[m])
            else:
                all_logits.append(wl)
                all_labels.append(y)

    if len(all_logits) == 0 or torch.cat(all_logits).numel() == 0:
        print("  WARNING: no valid warning labels in val; using T=1.0")
        return 1.0

    logits_cat = torch.cat(all_logits)
    labels_cat = torch.cat(all_labels)
    finite_mask = torch.isfinite(logits_cat) & torch.isfinite(labels_cat)
    if finite_mask.sum() == 0:
        print("  WARNING: non-finite logits/labels in val for calibration; using T=1.0")
        return 1.0
    logits_cat = logits_cat[finite_mask]
    labels_cat = labels_cat[finite_mask]

    ts = _TemperatureScaling()
    optimizer = torch.optim.LBFGS([ts.temperature], lr=0.01, max_iter=100)

    def closure():
        optimizer.zero_grad()
        scaled = ts(logits_cat)
        if not torch.isfinite(scaled).all():
            # Returning inf discourages this region and prevents NaNs from propagating.
            return scaled.new_tensor(float("inf"))
        loss = F.binary_cross_entropy_with_logits(scaled, labels_cat)
        if not torch.isfinite(loss):
            return loss.new_tensor(float("inf"))
        loss.backward()
        return loss

    optimizer.step(closure)
    t_val = ts.temperature.item()
    if not math.isfinite(t_val) or t_val <= 0:
        print("  WARNING: calibration produced non-finite/invalid T; using T=1.0")
        t_val = 1.0
    print(f"  Calibrated temperature T = {t_val:.4f}")
    return float(t_val)


# ═══════════════════ 5.1  Prediction metrics ═══════════════════

def prediction_metrics(
    mu_all: np.ndarray,
    target_all: np.ndarray,
    col_names: List[str],
) -> pd.DataFrame:
    """
    mu_all, target_all:  (N, H, D)
    Returns DataFrame with per-variable MAE / RMSE / peak-error / peak-phase.
    """
    N, H, D = mu_all.shape
    records = []
    for d in range(D):
        pred_d = mu_all[:, :, d]      # (N, H)
        true_d = target_all[:, :, d]

        mae = float(np.abs(pred_d - true_d).mean())
        rmse = float(np.sqrt(((pred_d - true_d) ** 2).mean()))

        peak_pred = pred_d.max(axis=1)
        peak_true = true_d.max(axis=1)
        peak_err = float(np.abs(peak_pred - peak_true).mean())

        phase_pred = pred_d.argmax(axis=1)
        phase_true = true_d.argmax(axis=1)
        phase_err = float(np.abs(phase_pred - phase_true).mean())

        records.append(dict(
            variable=col_names[d] if d < len(col_names) else f"var_{d}",
            MAE=mae, RMSE=rmse,
            peak_error=peak_err, phase_error_steps=phase_err,
        ))

    return pd.DataFrame(records)


# ═══════════════════ 5.2  Warning metrics ═══════════════════

def _mcc_binary(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Matthews correlation; NaN if empty or only one class in y_true."""
    yt = y_true.astype(int)
    yp = y_pred.astype(int)
    if len(yt) == 0 or len(np.unique(yt)) < 2:
        return float("nan")
    try:
        return float(matthews_corrcoef(yt, yp))
    except ValueError:
        return float("nan")


def warning_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    yt = y_true.astype(int)
    neg_n = int((yt == 0).sum())
    fp = int(((yt == 0) & (y_pred == 1)).sum())
    fpr = float(fp / neg_n) if neg_n > 0 else 0.0
    rec = float(recall_score(yt, y_pred, zero_division=0))
    m = dict(
        accuracy=float(accuracy_score(yt, y_pred)),
        balanced_accuracy=float(balanced_accuracy_score(yt, y_pred)),
        precision=float(precision_score(yt, y_pred, zero_division=0)),
        recall=rec,
        f1=float(f1_score(yt, y_pred, zero_division=0)),
        mcc=_mcc_binary(yt, y_pred),
        fpr=fpr,
        youden_j=float(rec - fpr),
    )
    try:
        m["auc"] = float(roc_auc_score(yt, y_prob))
    except ValueError:
        m["auc"] = float("nan")
    try:
        m["pr_auc"] = float(average_precision_score(yt, y_prob))
    except ValueError:
        m["pr_auc"] = float("nan")
    return m


def experiment_warning_metrics(
    meta_df: pd.DataFrame,
    threshold: float,
) -> Dict[str, float]:
    """
    Per-experiment macro-averaged warning metrics.
    Each experiment contributes equally regardless of its window count.
    """
    metric_keys = [
        "accuracy", "balanced_accuracy", "precision", "recall",
        "f1", "mcc", "fpr", "youden_j", "auc", "pr_auc",
    ]
    per_exp: List[Dict[str, float]] = []
    for _fname, g in meta_df.groupby("file"):
        if "warn_valid" in g.columns:
            g = g[g["warn_valid"] == 1]
        if len(g) == 0:
            continue
        per_exp.append(warning_metrics(g["y_true"].values, g["y_prob"].values, threshold))

    if not per_exp:
        return {k: float("nan") for k in metric_keys}

    result: Dict[str, float] = {}
    for k in metric_keys:
        vals = [m[k] for m in per_exp if k in m and math.isfinite(m.get(k, float("nan")))]
        result[k] = float(np.mean(vals)) if vals else float("nan")
    return result


def save_fold_warn_predictions(
    meta_df: pd.DataFrame,
    out_dir: Path,
    cfg: Config,
    *,
    fold_id: int | None = None,
    operating_threshold: float = 0.5,
) -> Path | None:
    """
    Export per-window test warning predictions for PR / threshold analysis.

    Writes to ``<kfold_root>/outputs/predictions/{Model}/fold_{id}_warn_predictions.csv``.
    """
    out_dir = Path(out_dir)
    fi = fold_id if fold_id is not None else fold_index_from_dir_name(out_dir.name)
    if fi is None:
        fi = 0

    tag = getattr(cfg, "prediction_export_tag", None)
    subdir = prediction_model_subdir(getattr(cfg, "model_type", None), tag)
    kfold_root = out_dir.parent if fold_index_from_dir_name(out_dir.name) is not None else out_dir
    dest = predictions_root(kfold_root, subdir)

    return save_fold_warn_predictions_from_meta(
        meta_df,
        dest,
        fold_id=fi,
        operating_threshold=operating_threshold,
    )


def _build_split_meta_df(
    ds: "FirePredictionDataset",
    warn_prob: np.ndarray,
    warn_true: np.ndarray,
    warn_valid: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    """Build per-window metadata DataFrame for any data split."""
    rows: List[Dict[str, Any]] = []
    for i, (ei, t) in enumerate(ds.indices):
        exp = ds.exps[ei]
        rows.append(dict(
            file=exp.get("file", f"exp_{ei}"),
            window_end_time=float(exp["time"][t]),
            event_time=float(exp.get("event_time", np.nan)),
            y_prob=float(warn_prob[i]),
            y_pred=int(warn_prob[i] >= threshold),
            y_true=int(warn_true[i]),
            warn_valid=int(warn_valid[i]),
        ))
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def experiment_threshold_scan(
    meta_df: pd.DataFrame,
    grid: np.ndarray | None = None,
) -> pd.DataFrame:
    """Experiment-level macro warning metrics × threshold grid."""
    if meta_df.empty:
        return pd.DataFrame()
    src = meta_df
    if "warn_valid" in src.columns:
        src = src[src["warn_valid"] == 1]
    if src.empty:
        return pd.DataFrame()
    if grid is None:
        grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    rows = []
    for th in grid:
        m = experiment_warning_metrics(src, float(th))
        m["threshold"] = float(th)
        rows.append(m)
    return pd.DataFrame(rows)


def threshold_scan(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    grid: np.ndarray | None = None,
) -> pd.DataFrame:
    if grid is None:
        grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    rows = []
    for th in grid:
        m = warning_metrics(y_true, y_prob, threshold=th)
        m["threshold"] = th
        rows.append(m)
    return pd.DataFrame(rows)


def collect_warning_predictions(
    model: TriModalFireModel,
    loader: DataLoader,
    device: str,
    temperature: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calibrated warning probabilities and labels, restricted to warn_valid=1.
    """
    probs: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            temp = batch["temp_field"].to(device)
            smoke = batch["smoke"].to(device)
            ld = batch["layer_domain"].to(device)
            msk_t = batch["mask_temp"].to(device)
            msk_s = batch["mask_smoke"].to(device)
            outputs = model(temp, ld, smoke, msk_t, msk_s)
            wl = outputs["warn_logits"].cpu().numpy() / temperature
            wp = 1.0 / (1.0 + np.exp(-wl)).reshape(-1)
            yt = batch["warn"].cpu().numpy().reshape(-1)
            wv = batch.get("warn_valid")
            if wv is not None:
                m = wv.cpu().numpy().astype(bool).reshape(-1)
                probs.append(wp[m])
                labels.append(yt[m])
            else:
                probs.append(wp)
                labels.append(yt)
    if not probs:
        return np.array([]), np.array([])
    return np.concatenate(probs), np.concatenate(labels)


def _collect_split_predictions(
    model: TriModalFireModel,
    loader: DataLoader,
    device: str,
    temperature: float,
    cfg: Config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Full split: mu (N,H,D), target (N,H,D), warn prob/label/valid per window (N,).
    """
    mus: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    wps: List[np.ndarray] = []
    wts: List[np.ndarray] = []
    wvs: List[np.ndarray] = []
    model.eval()
    debug_nan_checks = bool(getattr(cfg, "debug_nan_checks", False))
    with torch.no_grad():
        for batch_i, batch in enumerate(loader):
            temp = batch["temp_field"].to(device)
            smoke = batch["smoke"].to(device)
            ld = batch["layer_domain"].to(device)
            msk_t = batch["mask_temp"].to(device)
            msk_s = batch["mask_smoke"].to(device)
            outputs = model(temp, ld, smoke, msk_t, msk_s)
            if debug_nan_checks:
                wl = outputs.get("warn_logits")
                mu = outputs.get("mu")
                logvar = outputs.get("logvar")
                if wl is None or mu is None or logvar is None:
                    raise RuntimeError("debug_nan_checks: missing expected output keys")
                if (not torch.isfinite(wl).all()) or (not torch.isfinite(mu).all()) or (not torch.isfinite(logvar).all()):
                    wl_bad = (~torch.isfinite(wl)).sum().item()
                    mu_bad = (~torch.isfinite(mu)).sum().item()
                    lv_bad = (~torch.isfinite(logvar)).sum().item()
                    raise RuntimeError(
                        f"NaN/Inf detected in model outputs during eval at batch_i={batch_i}: "
                        f"warn_logits_bad={wl_bad}, mu_bad={mu_bad}, logvar_bad={lv_bad}"
                    )
            mu = outputs["mu"].cpu().numpy()
            wl = outputs["warn_logits"].cpu().numpy() / temperature
            wp = (1.0 / (1.0 + np.exp(-wl))).reshape(-1)
            if debug_nan_checks and (not np.isfinite(wp).all()):
                raise RuntimeError(f"NaN/Inf detected in warning probabilities during eval at batch_i={batch_i}")
            wt = batch["warn"].cpu().numpy().reshape(-1)
            wv = batch.get("warn_valid", torch.ones_like(batch["warn"]))
            wv = wv.cpu().numpy().astype(np.float32).reshape(-1)
            mus.append(mu)
            targets.append(batch["future"].numpy())
            wps.append(wp)
            wts.append(wt)
            wvs.append(wv)
    if not mus:
        h, d = cfg.prediction_horizon, cfg.n_pred_targets
        z3 = np.zeros((0, h, d), dtype=np.float32)
        z1 = np.array([], dtype=np.float32)
        return z3, z3, z1, z1, np.array([], dtype=bool)
    return (
        np.concatenate(mus, axis=0),
        np.concatenate(targets, axis=0),
        np.concatenate(wps),
        np.concatenate(wts),
        np.concatenate(wvs).astype(bool),
    )


def _regression_split_summary(
    mu: np.ndarray,
    target: np.ndarray,
    col_names: List[str],
) -> Dict[str, float]:
    if mu.size == 0 or target.size == 0:
        return {"mae_mean": float("nan"), "rmse_mean": float("nan")}
    df = prediction_metrics(mu, target, col_names)
    return {
        "mae_mean": float(df["MAE"].mean()),
        "rmse_mean": float(df["RMSE"].mean()),
    }


def threshold_from_val_neg_fpr(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    fpr_cap: float,
) -> float | None:
    """
    Threshold such that ~fpr_cap of validation negatives score at or above it
    (using the (1 - fpr_cap) quantile of negative-class probabilities).
    """
    if fpr_cap <= 0 or fpr_cap >= 1:
        return None
    neg = y_true.astype(int) == 0
    if neg.sum() == 0:
        return None
    return float(np.quantile(y_prob[neg], 1.0 - fpr_cap))


def _clip_operating_threshold(th: float, lo: float = 0.01, hi: float = 0.95) -> float:
    return float(np.clip(th, lo, hi))


def _constrained_threshold_from_scan(
    scan_val: pd.DataFrame,
    min_recall: float,
    secondary: str,
    fallback: float,
) -> float:
    """Pick threshold on validation: recall >= floor, then maximize secondary metric."""
    col = secondary if secondary in scan_val.columns else "mcc"
    for rmin in [min_recall, 0.9, 0.8]:
        sub = scan_val[scan_val["recall"] >= rmin]
        if len(sub) > 0:
            row = sub.sort_values(col, ascending=False).iloc[0]
            return _clip_operating_threshold(float(row["threshold"]))
    return _clip_operating_threshold(float(fallback))


def _high_recall_threshold_from_scan(
    scan_val: pd.DataFrame,
    best_th: float,
    min_recall: float,
    secondary: str = "precision",
) -> float:
    """Pick threshold on validation: among rows with recall >= target, best secondary metric."""
    return _constrained_threshold_from_scan(scan_val, min_recall, secondary, best_th)


def resolve_eval_threshold_strategy(cfg: Config) -> str:
    explicit = getattr(cfg, "eval_threshold_strategy", None)
    if explicit:
        return str(explicit)
    if bool(getattr(cfg, "eval_prefer_high_recall", False)):
        return "high_recall"
    return "max_f1"


def select_operating_threshold(
    cfg: Config,
    scan_val: pd.DataFrame,
    th_pack: dict,
    hr_th: float,
) -> tuple[float, str]:
    """Return (operating_threshold, strategy_name) from validation scan only."""
    strategy = resolve_eval_threshold_strategy(cfg)
    best_th = float(th_pack["best_f1_threshold"])
    merge_op = float(th_pack["operating_threshold"])
    min_rec = float(getattr(cfg, "eval_high_recall_min_recall", 0.95))
    hr_sec = str(getattr(cfg, "eval_high_recall_secondary", "precision"))

    if strategy == "max_f1":
        return _clip_operating_threshold(merge_op), strategy
    if strategy == "high_recall":
        return _clip_operating_threshold(hr_th), strategy
    if strategy == "max_mcc":
        row = scan_val.sort_values("mcc", ascending=False).iloc[0]
        return _clip_operating_threshold(float(row["threshold"])), strategy
    if strategy.startswith("constrained_"):
        sec = strategy.replace("constrained_", "", 1)
        th = _constrained_threshold_from_scan(scan_val, min_rec, sec, best_th)
        return th, strategy
    if strategy == "constrained":
        th = _constrained_threshold_from_scan(scan_val, min_rec, hr_sec, best_th)
        return th, f"constrained_{hr_sec}"
    return _clip_operating_threshold(merge_op), "max_f1"


def operating_threshold_from_validation(
    val_true: np.ndarray,
    val_prob: np.ndarray,
    scan_val: pd.DataFrame,
    fpr_cap: float,
    min_operating: float = 0.0,
) -> dict:
    """
    Select operating threshold via Precision-Recall: the threshold that
    maximises F1 (harmonic mean of P and R) on the validation set.

    Other criteria (balanced_accuracy, Youden J, FPR cap) are still computed
    and returned for diagnostic comparison, but they do NOT override the
    PR-based choice.
    """
    best_f1_row = scan_val.sort_values("f1", ascending=False).iloc[0]
    best_f1_th = float(best_f1_row["threshold"])

    th_fpr = threshold_from_val_neg_fpr(val_true, val_prob, fpr_cap)

    if "balanced_accuracy" in scan_val.columns:
        bal_row = scan_val.sort_values("balanced_accuracy", ascending=False).iloc[0]
        th_bal = float(bal_row["threshold"])
    else:
        th_bal = best_f1_th

    if "youden_j" in scan_val.columns:
        y_row = scan_val.sort_values("youden_j", ascending=False).iloc[0]
        th_youden = float(y_row["threshold"])
    else:
        th_youden = best_f1_th

    operating = float(np.clip(best_f1_th, 0.01, 0.95))
    return dict(
        best_f1_threshold=best_f1_th,
        fpr_cap_threshold=th_fpr,
        balanced_acc_threshold=th_bal,
        youden_threshold=th_youden,
        operating_threshold=operating,
    )


# ═══════════════════ lead-time (experiment level) ═══════════════════

def _leadtime_eligible_meta(meta_df: pd.DataFrame) -> pd.DataFrame:
    """Lead time 仅对具有有效 event_time 的 full_process 实验有意义（排除 post-fire）。"""
    if meta_df.empty or "event_time" not in meta_df.columns:
        return meta_df
    et = pd.to_numeric(meta_df["event_time"], errors="coerce")
    return meta_df.loc[et.notna()].copy()


def experiment_leadtime(
    meta_df: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    """
    meta_df must contain: file, window_end_time, event_time, y_prob.
    """
    results = []
    for fname, g in meta_df.groupby("file"):
        g = g.sort_values("window_end_time")
        event_time = g["event_time"].dropna().iloc[0] if g["event_time"].notna().any() else np.nan

        alarms = g[(g["y_pred"] == 1) & (g["window_end_time"] < event_time)]
        if len(alarms) > 0:
            first_alarm = float(alarms["window_end_time"].iloc[0])
            lead = float(event_time - first_alarm)
            success = 1
        else:
            first_alarm = np.nan
            lead = np.nan
            success = 0

        results.append(dict(
            file=fname, event_time=event_time,
            first_alarm=first_alarm, lead_time=lead,
            success=success, threshold=threshold,
            total_windows=len(g),
        ))

    return pd.DataFrame(results)


def _leadtime_summary(lead_df: pd.DataFrame) -> Dict[str, float]:
    """Aggregate lead-time stats for one split (NaN if no successful alarms)."""
    if lead_df.empty or lead_df["success"].sum() == 0:
        return dict(
            success_rate=float("nan"),
            mean_lead_time=float("nan"),
            median_lead_time=float("nan"),
            lead_ge_30s=float("nan"),
            lead_ge_60s=float("nan"),
        )
    succ = lead_df[lead_df["success"] == 1]
    return dict(
        success_rate=float(lead_df["success"].mean()),
        mean_lead_time=float(succ["lead_time"].mean()),
        median_lead_time=float(succ["lead_time"].median()),
        lead_ge_30s=float((succ["lead_time"] >= 30).mean()),
        lead_ge_60s=float((succ["lead_time"] >= 60).mean()),
    )


# 预警论文主指标（实验级 macro）
WARNING_PRIMARY_METRIC_KEYS = (
    "recall", "mean_lead_time", "f1", "mcc", "precision", "pr_auc",
)


def _primary_warning_row(
    exp_metrics: Dict[str, float],
    lead_summary: Dict[str, float] | None = None,
) -> Dict[str, float]:
    lead = lead_summary or {}
    return {
        "recall": exp_metrics.get("recall", float("nan")),
        "mean_lead_time": lead.get("mean_lead_time", float("nan")),
        "f1": exp_metrics.get("f1", float("nan")),
        "mcc": exp_metrics.get("mcc", float("nan")),
        "precision": exp_metrics.get("precision", float("nan")),
        "pr_auc": exp_metrics.get("pr_auc", float("nan")),
    }


def _write_warning_primary_outputs(
    eval_dir: Path,
    *,
    operating_threshold: float,
    me_train: Dict[str, float],
    me_val: Dict[str, float],
    me_test: Dict[str, float],
    lead_train_sum: Dict[str, float],
    lead_val_sum: Dict[str, float],
    lead_test_sum: Dict[str, float],
) -> None:
    """Save warning primary metrics for train / val / test (experiment-level)."""
    splits = {
        "train": _primary_warning_row(me_train, lead_train_sum),
        "val": _primary_warning_row(me_val, lead_val_sum),
        "test": _primary_warning_row(me_test, lead_test_sum),
    }
    payload = {
        "operating_threshold": operating_threshold,
        "level": "experiment",
        "metrics": list(WARNING_PRIMARY_METRIC_KEYS),
        "splits": splits,
    }
    with open(eval_dir / "warning_primary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            _json_sanitize(payload),
            f,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    pd.DataFrame([{"split": sp, **m} for sp, m in splits.items()]).to_csv(
        eval_dir / "warning_primary_metrics.csv", index=False,
    )


def _print_leadtime_block(split_name: str, lead_df: pd.DataFrame, threshold: float) -> None:
    if lead_df.empty or lead_df["success"].sum() == 0:
        print(f"\n── Lead Time [{split_name}] (th={threshold:.2f}) ──")
        print("  (no successful pre-hazard alarms)")
        return
    succ = lead_df[lead_df["success"] == 1]
    print(f"\n── Lead Time [{split_name}] (th={threshold:.2f}) ──")
    print(f"  success rate  = {lead_df['success'].mean():.2%}")
    print(f"  mean lead     = {succ['lead_time'].mean():.1f} s")
    print(f"  median lead   = {succ['lead_time'].median():.1f} s")
    print(f"  lead ≥ 30s    = {(succ['lead_time'] >= 30).mean():.2%}")
    print(f"  lead ≥ 60s    = {(succ['lead_time'] >= 60).mean():.2%}")


def _print_train_test_comparison(
    train_m: Dict[str, float],
    test_m: Dict[str, float],
    *,
    title: str,
    keys: List[str] | None = None,
) -> None:
    """Side-by-side train vs test metrics in the console."""
    if keys is None:
        keys = ["accuracy", "precision", "recall", "f1", "mcc", "auc", "pr_auc", "fpr"]
    print(f"\n── {title} ──")
    print(f"  {'metric':<14s} {'train':>10s} {'test':>10s}")
    print(f"  {'-' * 14} {'-' * 10} {'-' * 10}")
    for k in keys:
        tv = train_m.get(k, float("nan"))
        tev = test_m.get(k, float("nan"))
        ts = f"{tv:.4f}" if isinstance(tv, (int, float)) and math.isfinite(float(tv)) else "n/a"
        tes = f"{tev:.4f}" if isinstance(tev, (int, float)) and math.isfinite(float(tev)) else "n/a"
        print(f"  {k:<14s} {ts:>10s} {tes:>10s}")


def _warning_metrics_table(
    splits: List[tuple[str, Dict[str, float], Dict[str, float]]],
) -> pd.DataFrame:
    """Build long-form warning metrics table: split × level (pooled / experiment)."""
    rows: List[Dict[str, Any]] = []
    for split_name, pooled, experiment in splits:
        for level, metrics in [("pooled", pooled), ("experiment", experiment)]:
            row = {"split": split_name, "level": level}
            row.update(metrics)
            rows.append(row)
    return pd.DataFrame(rows)


# ═══════════════════ full evaluation ═══════════════════

def evaluate(cfg: Config | None = None, model_path: str | None = None):
    if cfg is None:
        cfg = Config()

    device = cfg.resolve_device()
    out_dir = Path(cfg.output_dir)
    eval_dir = out_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # ── load data ──
    _train_loader, train_eval_loader, val_loader, test_loader, scaler = build_dataloaders(cfg)

    n_test_win = len(test_loader.dataset)
    if 0 < n_test_win < 50:
        print(
            f"\nWARNING: test set has only {n_test_win} windows — "
            f"metrics (especially AUC) are unreliable for reporting."
        )

    if len(test_loader.dataset) == 0:
        print("WARNING: test set has 0 windows, nothing to evaluate.")
        return

    if len(val_loader.dataset) == 0:
        print("WARNING: val set has 0 windows; thresholds will default to 0.5.")

    # ── load model ──
    if model_path is None:
        model_path = out_dir / "best_model.pt"
    ckpt = load_torch_checkpoint(model_path, map_location=device)
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # ── temperature scaling on validation set ──
    print("\n── Calibrating probability (Temperature Scaling) ──")
    temperature = calibrate_temperature(model, val_loader, device)

    pred_col_names: List[str] = []
    if cfg.predict_smoke:
        pred_col_names += cfg.smoke_col_names
    if cfg.predict_layer_means:
        pred_col_names += [f"layer{i+1}_mean" for i in range(cfg.n_layers)]

    print("\n── Forward pass: train / val / test (for metrics) ──")
    train_mu, train_tgt, train_wp, train_wt, train_wv = _collect_split_predictions(
        model, train_eval_loader, device, temperature, cfg,
    )
    val_mu, val_tgt, val_wp, val_wt, val_wv = _collect_split_predictions(
        model, val_loader, device, temperature, cfg,
    )
    mu_all, target_all, warn_prob, warn_true, warn_valid = _collect_split_predictions(
        model, test_loader, device, temperature, cfg,
    )

    val_prob = val_wp[val_wv]
    val_true = val_wt[val_wv]

    warn_prob_eval = warn_prob[warn_valid]
    warn_true_eval = warn_true[warn_valid]
    n_val_pos = int(val_true.sum())
    n_val_neg = int(len(val_true) - n_val_pos)
    n_te_pos = int(warn_true_eval.sum())
    n_te_neg = int(len(warn_true_eval) - n_te_pos)
    print("\n── Class balance (warn_valid=1 windows only) ──")
    print(f"  val:  pos={n_val_pos}  neg={n_val_neg}  "
          f"(neg fraction {n_val_neg / max(len(val_true), 1):.3f})")
    print(f"  test: pos={n_te_pos}  neg={n_te_neg}  "
          f"(neg fraction {n_te_neg / max(len(warn_true_eval), 1):.3f})")

    # ── thresholds from validation (avoid tuning on test) ──
    fpr_cap = float(getattr(cfg, "eval_val_fpr_cap", 0.05))
    min_op = float(getattr(cfg, "eval_operating_min_threshold", 0.5))
    threshold_strategy = resolve_eval_threshold_strategy(cfg)
    threshold_level = str(getattr(cfg, "eval_threshold_level", "experiment"))
    if val_prob.size > 0 and val_true.size > 0:
        if threshold_level == "experiment":
            val_ds_for_scan: FirePredictionDataset = val_loader.dataset
            val_meta_for_scan = _build_split_meta_df(
                val_ds_for_scan, val_wp, val_wt, val_wv, threshold=0.0,
            )
            scan_val = experiment_threshold_scan(val_meta_for_scan)
            scan_val.to_csv(eval_dir / "threshold_scan_val_experiment.csv", index=False)
        else:
            scan_val = threshold_scan(val_true, val_prob)
        scan_val.to_csv(eval_dir / "threshold_scan_val.csv", index=False)
        scan_val.to_csv(eval_dir / "threshold_scan.csv", index=False)
        th_pack = operating_threshold_from_validation(
            val_true, val_prob, scan_val, fpr_cap, min_operating=min_op,
        )
        best_th = th_pack["best_f1_threshold"]
        merge_op = float(th_pack["operating_threshold"])
        min_rec = float(getattr(cfg, "eval_high_recall_min_recall", 0.95))
        hr_sec = str(getattr(cfg, "eval_high_recall_secondary", "precision"))
        hr_th = _high_recall_threshold_from_scan(scan_val, best_th, min_rec, secondary=hr_sec)
        op_th, threshold_strategy = select_operating_threshold(cfg, scan_val, th_pack, hr_th)
        print("\n── Threshold scan (validation only) ──")
        print(f"  scan level                = {threshold_level}")
        print(f"  strategy                  = {threshold_strategy}")
        print(f"  max F1 threshold          = {best_th:.2f}")
        fpr_disp = (
            f"{th_pack['fpr_cap_threshold']:.4f}"
            if th_pack["fpr_cap_threshold"] is not None
            else "n/a"
        )
        print(f"  neg-FPR cap ({fpr_cap:.0%}) threshold  = {fpr_disp}  (diagnostic)")
        print(f"  max balanced-acc threshold = {th_pack['balanced_acc_threshold']:.2f}  (diagnostic)")
        print(f"  max Youden J threshold     = {th_pack['youden_threshold']:.2f}  (diagnostic)")
        print(f"  max MCC threshold          = "
              f"{float(scan_val.sort_values('mcc', ascending=False).iloc[0]['threshold']):.2f}  (diagnostic)")
        print(f"  high-recall val th         = {hr_th:.2f}  (rec≥{min_rec:.0%}→0.9→0.8, best {hr_sec})")
        print(f"  ★ operating threshold      = {op_th:.2f}")
    else:
        best_th, hr_th = 0.5, 0.5
        merge_op = 0.5
        op_th = 0.5
        threshold_strategy = "default"
        th_pack = dict(
            best_f1_threshold=0.5,
            fpr_cap_threshold=None,
            balanced_acc_threshold=0.5,
            youden_threshold=0.5,
            operating_threshold=0.5,
        )
        print("\n── Threshold scan skipped (no val warning samples); using 0.5 ──")

    nan_warn = dict(
        accuracy=float("nan"),
        balanced_accuracy=float("nan"),
        precision=float("nan"),
        recall=float("nan"),
        f1=float("nan"),
        mcc=float("nan"),
        fpr=float("nan"),
        youden_j=float("nan"),
        auc=float("nan"),
        pr_auc=float("nan"),
    )

    # ── 5.1  prediction metrics (train + test CSV; val in summary only) ──
    pred_test_df = prediction_metrics(mu_all, target_all, pred_col_names)
    pred_test_df.insert(0, "split", "test")
    pred_test_df.to_csv(eval_dir / "prediction_metrics.csv", index=False)

    pred_train_df = prediction_metrics(train_mu, train_tgt, pred_col_names)
    pred_train_df.insert(0, "split", "train")
    pred_train_df.to_csv(eval_dir / "prediction_metrics_train.csv", index=False)

    pred_val_df = prediction_metrics(val_mu, val_tgt, pred_col_names)
    pred_val_df.insert(0, "split", "val")
    pd.concat([pred_train_df, pred_val_df, pred_test_df], ignore_index=True).to_csv(
        eval_dir / "prediction_metrics_by_split.csv", index=False,
    )

    print("\n── Prediction Metrics (test) ──")
    print(pred_test_df.drop(columns=["split"]).to_string(index=False))
    print("\n── Prediction Metrics (train) ──")
    print(pred_train_df.drop(columns=["split"]).to_string(index=False))

    # ── warning @ operating threshold (warn_valid=1 rows only) ──
    def _warn_split(wp, wt, wv) -> Dict[str, float]:
        if not np.any(wv):
            return dict(nan_warn)
        return warning_metrics(wt[wv], wp[wv], threshold=op_th)

    m_train = _warn_split(train_wp, train_wt, train_wv)
    m_val = _warn_split(val_wp, val_wt, val_wv)
    m_test = _warn_split(warn_prob, warn_true, warn_valid)

    train_ds: FirePredictionDataset = train_eval_loader.dataset
    val_ds: FirePredictionDataset = val_loader.dataset
    test_ds: FirePredictionDataset = test_loader.dataset
    train_meta_df = _build_split_meta_df(train_ds, train_wp, train_wt, train_wv, op_th)
    val_meta_df = _build_split_meta_df(val_ds, val_wp, val_wt, val_wv, op_th)
    test_meta_df = _build_split_meta_df(test_ds, warn_prob, warn_true, warn_valid, op_th)

    me_train = experiment_warning_metrics(train_meta_df, op_th) if len(train_meta_df) > 0 else dict(nan_warn)
    me_val = experiment_warning_metrics(val_meta_df, op_th) if len(val_meta_df) > 0 else dict(nan_warn)
    me_test = experiment_warning_metrics(test_meta_df, op_th) if len(test_meta_df) > 0 else dict(nan_warn)

    reg_train = _regression_split_summary(train_mu, train_tgt, pred_col_names)
    reg_val = _regression_split_summary(val_mu, val_tgt, pred_col_names)
    reg_test = _regression_split_summary(mu_all, target_all, pred_col_names)

    # ── per-split prediction exports ──
    if len(train_meta_df) > 0:
        train_meta_df.to_csv(eval_dir / "train_predictions.csv", index=False)
    if len(test_meta_df) > 0:
        test_meta_df.to_csv(eval_dir / "test_predictions.csv", index=False)
        meta_df = test_meta_df
    else:
        meta_df = test_meta_df

    _warn_pred_path = save_fold_warn_predictions(
        test_meta_df, out_dir, cfg, operating_threshold=op_th,
    )
    if _warn_pred_path is not None:
        print(f"  Warn predictions export (test): {_warn_pred_path}")

    _ts_path = export_one_fold_timesteps(Path(cfg.output_dir))
    if _ts_path is not None:
        print(f"  逐时刻 test 导出: {_ts_path}")

    # ── lead time: train + test ──
    # Train loader may include post-fire (no event_time); exclude them from train lead-time stats.
    train_lead_meta = _leadtime_eligible_meta(train_meta_df)
    lead_train_df = (
        experiment_leadtime(train_lead_meta, threshold=op_th)
        if len(train_lead_meta) > 0 else pd.DataFrame()
    )
    val_lead_meta = _leadtime_eligible_meta(val_meta_df)
    lead_val_df = (
        experiment_leadtime(val_lead_meta, threshold=op_th)
        if len(val_lead_meta) > 0 else pd.DataFrame()
    )
    lead_test_df = experiment_leadtime(test_meta_df, threshold=op_th) if len(test_meta_df) > 0 else pd.DataFrame()
    if not lead_test_df.empty:
        lead_test_df.to_csv(eval_dir / "leadtime_by_experiment.csv", index=False)
    lead_parts = []
    if not lead_train_df.empty:
        lead_train_df = lead_train_df.copy()
        lead_train_df.insert(0, "split", "train")
        lead_parts.append(lead_train_df)
    if not lead_val_df.empty:
        lead_val_df = lead_val_df.copy()
        lead_val_df.insert(0, "split", "val")
        lead_parts.append(lead_val_df)
    if not lead_test_df.empty:
        lead_test_df = lead_test_df.copy()
        lead_test_df.insert(0, "split", "test")
        lead_parts.append(lead_test_df)
    if lead_parts:
        pd.concat(lead_parts, ignore_index=True).to_csv(
            eval_dir / "leadtime_by_split.csv", index=False,
        )

    _print_leadtime_block("train", lead_train_df, op_th)
    _print_leadtime_block("val", lead_val_df, op_th)
    _print_leadtime_block("test", lead_test_df, op_th)

    lead_train_sum = _leadtime_summary(lead_train_df)
    lead_val_sum = _leadtime_summary(lead_val_df)
    lead_test_sum = _leadtime_summary(lead_test_df)

    _write_warning_primary_outputs(
        eval_dir,
        operating_threshold=op_th,
        me_train=me_train,
        me_val=me_val,
        me_test=me_test,
        lead_train_sum=lead_train_sum,
        lead_val_sum=lead_val_sum,
        lead_test_sum=lead_test_sum,
    )

    from analysis.warning_threshold_sensitivity import write_fold_threshold_sensitivity
    sens_path = write_fold_threshold_sensitivity(
        eval_dir,
        meta_by_split={
            "train": train_meta_df,
            "val": val_meta_df,
            "test": test_meta_df,
        },
        operating_threshold=op_th,
    )
    print(f"  阈值敏感性（主指标 train/val/test）: {sens_path}")

    # ── 5.2  warning metrics on test (threshold sweeps; analysis only) ──
    if warn_prob_eval.size == 0:
        print("\n── Test warning: no warn_valid=1 windows; skipping classification metrics ──")
        scan_test = pd.DataFrame()
    else:
        scan_test = threshold_scan(warn_true_eval, warn_prob_eval)
        scan_test.to_csv(eval_dir / "threshold_scan_test.csv", index=False)

    if warn_prob_eval.size > 0:
        for name, th in [
            ("primary(val-only)", op_th),
            ("pr_f1(val)", merge_op),
            ("max_F1(val)", best_th),
            ("high_recall_pick(val)", hr_th),
            ("default", 0.5),
        ]:
            m = warning_metrics(warn_true_eval, warn_prob_eval, threshold=th)
            print(f"\n── Test warning (warn_valid=1) @ {name} (th={th:.2f}) ──")
            for k, v in m.items():
                print(f"  {k:12s} = {v:.4f}")

    if len(scan_test) > 0 and val_prob.size > 0:
        leak_row = scan_test.sort_values("f1", ascending=False).iloc[0]
        leak_th = float(leak_row["threshold"])
        if abs(leak_th - op_th) > 1e-6:
            print(f"\n  (diagnostic) max-F1 threshold *if chosen on test* would be {leak_th:.2f} "
                  f"(diff vs operating(val-only)={leak_th - op_th:+.2f})")

    print("\n── Warning metrics: window-level pooled (warn_valid=1) ──")
    for split_name, m in [("train", m_train), ("val", m_val), ("test", m_test)]:
        mcc_str = f"  mcc={m['mcc']:.4f}" if "mcc" in m and math.isfinite(m.get("mcc", float("nan"))) else ""
        pr_auc_str = f"  pr_auc={m['pr_auc']:.4f}" if "pr_auc" in m else ""
        print(f"  [{split_name}]  f1={m['f1']:.4f}  auc={m['auc']:.4f}"
              f"{pr_auc_str}{mcc_str}  P={m['precision']:.4f}  R={m['recall']:.4f}")

    print("\n── Warning metrics: experiment-level macro-average (warn_valid=1) ──")
    for split_name, m in [("train", me_train), ("val", me_val), ("test", me_test)]:
        mcc_str = f"  mcc={m['mcc']:.4f}" if "mcc" in m and math.isfinite(m.get("mcc", float("nan"))) else ""
        pr_auc_str = f"  pr_auc={m['pr_auc']:.4f}" if "pr_auc" in m else ""
        print(f"  [{split_name}]  f1={m['f1']:.4f}  auc={m['auc']:.4f}"
              f"{pr_auc_str}{mcc_str}  P={m['precision']:.4f}  R={m['recall']:.4f}")

    _print_train_test_comparison(
        m_train, m_test, title="Train vs Test — pooled warning metrics",
    )
    _print_train_test_comparison(
        me_train, me_test, title="Train vs Test — experiment-level warning metrics",
    )
    print(f"\n── Train vs Test — lead time (th={op_th:.2f}) ──")
    print(f"  {'metric':<18s} {'train':>10s} {'test':>10s}")
    print(f"  {'-' * 18} {'-' * 10} {'-' * 10}")
    for k in ["success_rate", "mean_lead_time", "median_lead_time"]:
        tv = lead_train_sum.get(k, float("nan"))
        tev = lead_test_sum.get(k, float("nan"))
        ts = f"{tv:.4f}" if isinstance(tv, (int, float)) and math.isfinite(float(tv)) else "n/a"
        tes = f"{tev:.4f}" if isinstance(tev, (int, float)) and math.isfinite(float(tev)) else "n/a"
        print(f"  {k:<18s} {ts:>10s} {tes:>10s}")

    warn_table = _warning_metrics_table([
        ("train", m_train, me_train),
        ("val", m_val, me_val),
        ("test", m_test, me_test),
    ])
    warn_table.to_csv(eval_dir / "warning_metrics_by_split.csv", index=False)

    reg_table = pd.DataFrame([
        {"split": "train", **reg_train},
        {"split": "val", **reg_val},
        {"split": "test", **reg_test},
    ])
    reg_table.to_csv(eval_dir / "regression_metrics_by_split.csv", index=False)

    print("\n── Regression summary (train vs test) ──")
    print(f"  {'split':<8s} {'mae_mean':>10s} {'rmse_mean':>10s}")
    for split_name, reg in [("train", reg_train), ("test", reg_test)]:
        print(f"  {split_name:<8s} {reg['mae_mean']:>10.4f} {reg['rmse_mean']:>10.4f}")

    # ── eval_summary.json: threshold provenance + train/val/test only ──
    test_alternatives: Dict[str, Any] = {}
    if warn_prob_eval.size > 0 and val_prob.size > 0:
        test_alternatives["at_max_f1_val_threshold"] = {
            "threshold": float(best_th),
            **warning_metrics(warn_true_eval, warn_prob_eval, threshold=best_th),
        }
        test_alternatives["at_pr_f1_val"] = {
            "threshold": float(merge_op),
            **warning_metrics(warn_true_eval, warn_prob_eval, threshold=merge_op),
        }

    summary = dict(
        operating_threshold=op_th,
        calibration_temperature=temperature,
        thresholds_selected_on="validation",
        threshold_strategy=threshold_strategy,
        threshold_scan_level=threshold_level,
        note=(
            "warning: window-level pooled metrics; "
            "warning_experiment_level: per-experiment macro-averaged metrics "
            "(each experiment weighted equally regardless of window count); "
            "both use warn_valid=1 windows at operating_threshold "
            f"(strategy={threshold_strategy}, scan={threshold_level}); "
            "mcc: Matthews correlation at operating threshold "
            "(experiment-level = macro mean of per-experiment MCC); "
            "regression: mean of per-target MAE/RMSE over all windows."
        ),
        warning=dict(train=m_train, val=m_val, test=m_test),
        warning_experiment_level=dict(train=me_train, val=me_val, test=me_test),
        lead_time=dict(train=lead_train_sum, val=lead_val_sum, test=lead_test_sum),
        test_threshold_sweep=test_alternatives,
        regression=dict(train=reg_train, val=reg_val, test=reg_test),
    )
    with open(eval_dir / "eval_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            _json_sanitize(summary),
            f,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )

    print(f"\nEvaluation outputs saved to {eval_dir}")
    print("  Key files: warning_primary_metrics.json/.csv (主指标 train/val/test),")
    print("             threshold_sensitivity_primary.csv (阈值敏感性),")
    print("             eval_summary.json, warning_metrics_by_split.csv,")
    print("             train_predictions.csv, test_predictions.csv,")
    print("             leadtime_by_split.csv, regression_metrics_by_split.csv")


if __name__ == "__main__":
    from data_pipeline.preprocess import discover_and_configure
    cfg = Config()
    discover_and_configure(cfg)
    evaluate(cfg)

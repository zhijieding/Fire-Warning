"""
Shared I/O for per-fold warning prediction exports (PR / threshold analysis).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLS = ("fold_id", "exp_id", "time_idx", "y_true", "y_prob")

MODEL_PREDICTION_SUBDIRS: dict[str, str] = {
    "lstm": "LSTM",
    "bilstm": "BiLSTM",
    "gru": "GRU",
    "tcn": "TCN",
    "informer": "Informer",
    "transformer_noxa": "Transformer",
    "patchtst": "PatchTST",
    "itransformer": "iTransformer",
    "timesnet": "TimesNet",
    "trimodal": "Ours",
    "layer_bridge": "Ours",
}


def fold_index_from_dir_name(name: str) -> int | None:
    m = re.match(r"fold_(\d+)_", name)
    return int(m.group(1)) if m else None


def prediction_model_subdir(model_type: str | None, export_tag: str | None = None) -> str:
    if export_tag:
        return str(export_tag)
    mt = (model_type or "trimodal").lower()
    return MODEL_PREDICTION_SUBDIRS.get(mt, mt)


def predictions_root(kfold_or_run_root: Path, model_subdir: str | None = None) -> Path:
    base = Path(kfold_or_run_root) / "outputs" / "predictions"
    return base / model_subdir if model_subdir else base


def operating_threshold_for_fold(fold_dir: Path) -> float | None:
    p = fold_dir / "eval" / "eval_summary.json"
    if not p.is_file():
        return None
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        th = data.get("operating_threshold")
        return float(th) if th is not None else None
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def export_df_from_test_predictions(
    raw: pd.DataFrame,
    fold_id: int,
    operating_threshold: float | None = None,
) -> pd.DataFrame | None:
    """Convert eval/test_predictions.csv rows to standard export format."""
    if raw.empty:
        return None
    df = raw.copy()
    if "file" in df.columns and "exp_id" not in df.columns:
        df["exp_id"] = df["file"]
    if "window_end_time" in df.columns and "time_idx" not in df.columns:
        df["time_idx"] = df["window_end_time"]
    if "exp_id" not in df.columns or "y_prob" not in df.columns or "y_true" not in df.columns:
        return None
    if "warn_valid" in df.columns:
        df = df[df["warn_valid"].astype(int) == 1]
    if df.empty:
        return None

    y_prob = df["y_prob"].astype(float)
    th = operating_threshold if operating_threshold is not None else 0.5
    y_pred = (
        df["y_pred"].astype(int)
        if "y_pred" in df.columns
        else (y_prob >= th).astype(int)
    )

    return pd.DataFrame(
        {
            "fold_id": int(fold_id),
            "exp_id": df["exp_id"].astype(str),
            "time_idx": df["time_idx"].astype(float),
            "y_true": df["y_true"].astype(int),
            "y_prob": y_prob,
            "y_pred": y_pred,
            "event_time": (
                df["event_time"].astype(float)
                if "event_time" in df.columns
                else np.nan
            ),
            "operating_threshold": float(th),
        }
    )


def save_fold_warn_predictions_df(
    export: pd.DataFrame,
    dest_dir: Path,
    fold_id: int,
) -> Path:
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_path = dest_dir / f"fold_{fold_id}_warn_predictions.csv"
    export.to_csv(out_path, index=False)
    return out_path


def save_fold_warn_predictions_from_meta(
    meta_df: pd.DataFrame,
    dest_dir: Path,
    *,
    fold_id: int,
    operating_threshold: float,
) -> Path | None:
    if meta_df is None or meta_df.empty:
        return None
    required = ("file", "window_end_time", "y_true", "y_prob")
    missing = [c for c in required if c not in meta_df.columns]
    if missing:
        print(f"  WARNING: cannot export warn predictions; missing columns: {missing}")
        return None

    df = meta_df.copy()
    if "warn_valid" in df.columns:
        df = df[df["warn_valid"].astype(int) == 1]
    if df.empty:
        return None

    export = pd.DataFrame(
        {
            "fold_id": int(fold_id),
            "exp_id": df["file"].astype(str),
            "time_idx": df["window_end_time"].astype(float),
            "y_true": df["y_true"].astype(int),
            "y_prob": df["y_prob"].astype(float),
            "y_pred": (df["y_prob"].astype(float) >= operating_threshold).astype(int),
            "event_time": (
                df["event_time"].astype(float)
                if "event_time" in df.columns
                else np.nan
            ),
            "operating_threshold": float(operating_threshold),
        }
    )
    return save_fold_warn_predictions_df(export, dest_dir, fold_id)


def materialize_predictions_from_eval(
    kfold_root: Path,
    model_subdir: str | None = None,
) -> list[Path]:
    """Build fold_*_warn_predictions.csv from fold_*/eval/test_predictions.csv."""
    kfold_root = Path(kfold_root)
    pred_dir = predictions_root(kfold_root, model_subdir)
    written: list[Path] = []

    for sub in sorted(kfold_root.iterdir()):
        if not sub.is_dir():
            continue
        fi = fold_index_from_dir_name(sub.name)
        if fi is None:
            continue
        src = sub / "eval" / "test_predictions.csv"
        if not src.is_file():
            continue
        raw = pd.read_csv(src)
        op_th = operating_threshold_for_fold(sub)
        export = export_df_from_test_predictions(raw, fi, operating_threshold=op_th)
        if export is None:
            continue
        written.append(save_fold_warn_predictions_df(export, pred_dir, fi))
    return written


def load_fold_prediction_files(predictions_dir: Path) -> list[Path]:
    predictions_dir = Path(predictions_dir)
    if not predictions_dir.is_dir():
        return []
    return sorted(predictions_dir.glob("fold_*_warn_predictions.csv"))


def load_merged_predictions(predictions_dir: Path) -> pd.DataFrame:
    paths = load_fold_prediction_files(predictions_dir)
    if not paths:
        raise FileNotFoundError(f"未找到 {predictions_dir}/fold_*_warn_predictions.csv")

    frames: list[pd.DataFrame] = []
    for p in paths:
        df = pd.read_csv(p)
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(
                f"{p} 缺少列 {missing}；需要 evaluation 阶段保存 y_prob。"
            )
        if df["y_prob"].isna().all():
            raise ValueError(f"{p}: y_prob 全为 NaN，无法绘制 PR 曲线。")
        frames.append(df)

    merged = pd.concat(frames, axis=0, ignore_index=True)
    return merged.sort_values(["fold_id", "exp_id", "time_idx"]).reset_index(drop=True)

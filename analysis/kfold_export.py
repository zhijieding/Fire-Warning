"""
合并各折 test 集上的逐时刻预警预测，导出标准列名 CSV。

标准列（与 evaluate 输出的 test_predictions 对齐）：
  time, true_label, pred_prob, pred_label, 文件名, 折号
其中 time 对应 window_end_time。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# 合并表与下游分析共用的「文件名 / 折号」列名（中文，便于报表）
COL_FILE_ZH = "文件名"
COL_FOLD_ZH = "折号"
# 兼容旧脚本
COL_FILE_EN = "file_name"
COL_FOLD_EN = "fold"


def _fold_index_from_dir(name: str) -> int | None:
    m = re.match(r"fold_(\d+)_", name)
    return int(m.group(1)) if m else None


def experiment_id_to_file_name(exp_id: str) -> str:
    """将数据集 id（如 1、23-1）映射为常见文件名。"""
    if exp_id in ("10", "fire_merged_1hz_reference"):
        return "fire_merged_1hz_reference.csv"
    return f"{exp_id}_clean.csv"


def timestep_export_from_predictions(df: pd.DataFrame, fold_idx: int) -> pd.DataFrame:
    """
    将单折 eval/test_predictions.csv 转为标准 6 列 + 可选 warn_valid / event_time。
    """
    if df.empty:
        return pd.DataFrame()
    exp_col = "file" if "file" in df.columns else None
    if exp_col is None:
        raise ValueError("test_predictions 缺少列 file")
    exp_ids = df[exp_col].astype(str)
    file_names = exp_ids.map(experiment_id_to_file_name)
    out = pd.DataFrame(
        {
            "time": df["window_end_time"].astype(float),
            "true_label": df["y_true"].astype(int),
            "pred_prob": df["y_prob"].astype(float),
            "pred_label": df["y_pred"].astype(int),
            COL_FILE_ZH: file_names.astype(str),
            COL_FOLD_ZH: int(fold_idx),
        }
    )
    if "warn_valid" in df.columns:
        out["warn_valid"] = df["warn_valid"].astype(int)
    if "event_time" in df.columns:
        out["event_time"] = df["event_time"].astype(float)
    return out


def export_one_fold_timesteps(fold_dir: Path) -> Path | None:
    """
    在 fold_dir/eval/ 下写入 test_timesteps_export.csv（标准列）。
    若无 test_predictions.csv 则返回 None。
    """
    pred_path = fold_dir / "eval" / "test_predictions.csv"
    if not pred_path.exists():
        return None
    fi = _fold_index_from_dir(fold_dir.name)
    if fi is None:
        return None
    df = pd.read_csv(pred_path)
    out = timestep_export_from_predictions(df, fi)
    if out.empty:
        return None
    out_path = fold_dir / "eval" / "test_timesteps_export.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return out_path


def export_all_fold_timesteps(kfold_root: str | Path) -> list[Path]:
    """遍历 kfold_root 下各折，分别写入 eval/test_timesteps_export.csv。"""
    kfold_root = Path(kfold_root)
    written: list[Path] = []
    for sub in sorted(kfold_root.iterdir()):
        if not sub.is_dir():
            continue
        p = export_one_fold_timesteps(sub)
        if p is not None:
            written.append(p)
    return written


def merge_kfold_test_predictions(
    kfold_root: str | Path,
    out_csv: str | Path | None = None,
    *,
    add_english_aliases: bool = True,
) -> Path:
    """
    扫描 kfold_root 下 fold_* 子目录，读取 eval/test_predictions.csv，
    合并为一张表，列至少包含：time, true_label, pred_prob, pred_label, 文件名, 折号。

    若 add_english_aliases=True，额外保留 file_name、fold 两列（与 文件名、折号 同值），
    便于旧脚本读取。

    说明：time 对应原表 window_end_time（窗口右端时刻，与 evaluate 一致）。
    """
    kfold_root = Path(kfold_root)
    out_csv = Path(out_csv or kfold_root / "all_folds_test_timesteps.csv")

    rows: list[pd.DataFrame] = []
    for sub in sorted(kfold_root.iterdir()):
        if not sub.is_dir():
            continue
        fi = _fold_index_from_dir(sub.name)
        if fi is None:
            continue
        pred_path = sub / "eval" / "test_predictions.csv"
        if not pred_path.exists():
            continue
        df = pd.read_csv(pred_path)
        if df.empty:
            continue
        exp_col = "file" if "file" in df.columns else None
        if exp_col is None:
            continue
        out = timestep_export_from_predictions(df, fi)
        if add_english_aliases:
            out[COL_FILE_EN] = out[COL_FILE_ZH]
            out[COL_FOLD_EN] = out[COL_FOLD_ZH]
        rows.append(out)

    if not rows:
        raise FileNotFoundError(
            f"未找到任何 {kfold_root}/fold_*/eval/test_predictions.csv"
        )

    merged = pd.concat(rows, axis=0, ignore_index=True)
    sort_cols = [COL_FOLD_ZH, COL_FILE_ZH, "time"]
    merged = merged.sort_values(sort_cols).reset_index(drop=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False)
    return out_csv


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="各折 test 逐时刻预测导出与合并")
    ap.add_argument(
        "kfold_root",
        nargs="?",
        default="./trimodal_output_w/o_cross",
        help="含 fold_* 子目录的根路径",
    )
    ap.add_argument(
        "-o", "--output",
        default=None,
        help="合并输出 CSV 路径（默认 <kfold_root>/all_folds_test_timesteps.csv）",
    )
    ap.add_argument(
        "--per-fold",
        action="store_true",
        help="在每折 eval/ 下写入 test_timesteps_export.csv",
    )
    ap.add_argument(
        "--no-en-alias",
        action="store_true",
        help="合并表不附加 file_name / fold 英文别名列",
    )
    ap.add_argument(
        "--key-moments",
        action="store_true",
        help="合并后基于同一张表生成完整过程关键时刻 CSV（需 warning_lead_analysis）",
    )
    ap.add_argument(
        "--key-moments-output",
        default=None,
        help="关键时刻表输出路径（默认 <kfold_root>/complete_fire_key_moments.csv）",
    )
    args = ap.parse_args()
    if args.per_fold:
        paths = export_all_fold_timesteps(args.kfold_root)
        print(f"已写入各折 test_timesteps_export.csv，共 {len(paths)} 个文件")
    p = merge_kfold_test_predictions(
        args.kfold_root,
        args.output,
        add_english_aliases=not args.no_en_alias,
    )
    print(f"已写入合并表: {p}  行数={len(pd.read_csv(p))}")
    if args.key_moments:
        from analysis.warning_lead_analysis import build_complete_fire_key_moments

        km_out = args.key_moments_output or str(
            Path(args.kfold_root) / "complete_fire_key_moments.csv"
        )
        _, km_path = build_complete_fire_key_moments(p, out_csv=km_out)
        print(f"已写入关键时刻表: {km_path}")

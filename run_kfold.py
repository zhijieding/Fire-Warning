#!/usr/bin/env python
"""
K-Fold Cross-Validation across experiments.

Splits full-process experiments into K groups (default K=5) via round-robin
so that each fold gets a mix of experiment IDs.  Each fold holds out one
group for testing, uses the *next* group for validation, and trains on the
remaining groups.

With 10 experiments and K=5:  2 test / 2 val / 6 train per fold.

Contrastive pre-training (``--pretrain``) uses only ``post_fire_ids`` CSVs
(disjoint from CV test experiments) — one shared backbone, no leakage.

Scope:
    Uses ``cfg.full_process_ids`` only (auto-discovered: the ~10 experiments with
    enough pre-fire labels for supervised warning).  Evolution / stage tasks on
    all ~80 unified CSVs are separate — see ``analysis/workflow_post_train.py``.

Usage:
    python run_kfold.py                      # 5-fold from scratch
    python run_kfold.py --kfold 5            # explicit 5-fold
    python run_kfold.py --pretrain           # with contrastive backbone
    python run_kfold.py --max-folds 2        # debug: run first 2 folds only
    python run_kfold.py --fusion-mode direct_smoke_temp  #w/o Bridge
    python run_kfold.py --fusion-mode late_concat # w/o Cross-Attn
    python run_kfold.py --lambda-consist 0 # w/o physical consistency loss
    python run_kfold.py --lambda-trend 0     # w/o slope (trend) loss
    python run_kfold.py --lambda-pred 0 --lambda-trend 0 --lambda-consist 0 \\
        --output-dir ./ablate_wo_pred        # RQ3: warn-only (no regression supervision)
    python run_kfold.py --output-dir ./trimodal_ablate_late --fusion-mode late_concat
    

"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import Config


# ───────────────── fold paths / resume ─────────────────

def _fold_dir_path(base_output: str | Path, fold_idx: int, test_ids: list[str]) -> Path:
    fold_tag = "_".join(test_ids)
    return Path(base_output) / f"fold_{fold_idx}_test{fold_tag}"


def _fold_is_complete(fold_dir: Path) -> bool:
    return (fold_dir / "eval" / "eval_summary.json").is_file()


_EXP_METRICS = (
    "accuracy", "precision", "recall", "f1", "mcc", "auc", "pr_auc",
    "mean_lead_time", "success_rate",
)
_POOLED_CLASS_METRICS = (
    "accuracy", "precision", "recall", "f1", "mcc", "auc", "pr_auc",
)


def _split_warning_metrics(
    eval_summary: dict,
    split: str,
) -> tuple[dict[str, float], dict[str, float]]:
    """Experiment-level and window-pooled warning metrics for one split."""
    exp = (eval_summary.get("warning_experiment_level") or {}).get(split) or {}
    pooled = (eval_summary.get("warning") or {}).get(split) or {}
    lead = (eval_summary.get("lead_time") or {}).get(split) or {}
    exp_out = {m: float(exp.get(m, np.nan)) for m in _POOLED_CLASS_METRICS}
    exp_out["mean_lead_time"] = float(lead.get("mean_lead_time", np.nan))
    exp_out["success_rate"] = float(lead.get("success_rate", np.nan))
    pooled_out = {m: float(pooled.get(m, np.nan)) for m in _POOLED_CLASS_METRICS}
    return exp_out, pooled_out


def _prefixed_metric_row(
    exp: dict[str, float],
    pooled: dict[str, float],
    *,
    prefix: str = "",
) -> dict[str, float]:
    """Flatten experiment-level + pooled metrics with optional column prefix."""
    p = f"{prefix}_" if prefix else ""
    row: dict[str, float] = {}
    for m in _EXP_METRICS:
        row[f"{p}{m}"] = exp.get(m, np.nan)
    for m in _POOLED_CLASS_METRICS:
        row[f"{p}pooled_{m}"] = pooled.get(m, np.nan)
    return row


def _mean_std_block(df: pd.DataFrame, columns: list[str]) -> dict[str, dict[str, float]]:
    block: dict[str, dict[str, float]] = {}
    for col in columns:
        if col not in df.columns:
            continue
        vals = df[col].dropna()
        if len(vals) == 0:
            continue
        block[col] = {
            "mean": float(vals.mean()),
            "std": float(vals.std()) if len(vals) > 1 else 0.0,
        }
    return block


def _summary_section_from_df(df: pd.DataFrame, *, prefix: str = "") -> dict[str, dict[str, float]]:
    """Build {metric: {mean, std}} for experiment-level columns with optional prefix."""
    p = f"{prefix}_" if prefix else ""
    cols = [f"{p}{m}" for m in _EXP_METRICS if f"{p}{m}" in df.columns]
    raw = _mean_std_block(df, cols)
    return {k[len(p):] if p and k.startswith(p) else k: v for k, v in raw.items()}


def _pooled_summary_from_df(df: pd.DataFrame, *, prefix: str = "") -> dict[str, dict[str, float]]:
    p = f"{prefix}_" if prefix else ""
    cols = [f"{p}pooled_{m}" for m in _POOLED_CLASS_METRICS if f"{p}pooled_{m}" in df.columns]
    raw = _mean_std_block(df, cols)
    return {k[len(f"{p}pooled_"):]: v for k, v in raw.items()}


def _fold_metrics_from_eval(
    eval_summary: dict,
    *,
    fold_idx: int,
    test_ids: list[str],
    val_ids: list[str],
    test_loss: float = np.nan,
) -> dict:
    """Build per-fold result row (train + test) from eval_summary.json."""
    test_exp, test_pooled = _split_warning_metrics(eval_summary, "test")
    train_exp, train_pooled = _split_warning_metrics(eval_summary, "train")
    row = dict(
        fold=fold_idx,
        test_ids=",".join(test_ids),
        val_ids=",".join(val_ids),
        best_threshold=eval_summary.get("operating_threshold", np.nan),
        test_loss=test_loss,
    )
    row.update(_prefixed_metric_row(test_exp, test_pooled, prefix=""))
    row.update(_prefixed_metric_row(train_exp, train_pooled, prefix="train"))
    return row


def _result_from_completed_fold(
    fold_idx: int,
    test_ids: list[str],
    val_ids: list[str],
    fold_dir: Path,
) -> dict:
    """Rebuild fold metrics from an existing eval/ directory (for --resume)."""
    eval_summary_path = fold_dir / "eval" / "eval_summary.json"
    with open(eval_summary_path, encoding="utf-8") as f:
        eval_summary = json.load(f)

    test_loss = np.nan
    run_cfg_path = fold_dir / "run_config.json"
    if run_cfg_path.exists():
        with open(run_cfg_path, encoding="utf-8") as f:
            run_cfg = json.load(f)
        test_loss = (run_cfg.get("test_metrics") or {}).get("loss", np.nan)

    return _fold_metrics_from_eval(
        eval_summary,
        fold_idx=fold_idx,
        test_ids=test_ids,
        val_ids=val_ids,
        test_loss=test_loss,
    )


# ───────────────── fold runner ─────────────────

def _run_one_fold(
    fold_idx: int,
    test_ids: list[str],
    val_ids: list[str],
    train_ids: list[str],
    base_cfg: Config,
    pretrained: bool,
) -> dict:
    """Train + evaluate a single CV fold, return metrics dict."""
    cfg = copy.deepcopy(base_cfg)
    cfg.test_ids = list(test_ids)
    cfg.val_ids = list(val_ids)
    cfg.train_ids = list(train_ids)

    fold_dir = _fold_dir_path(cfg.output_dir, fold_idx, test_ids)
    fold_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_dir = str(fold_dir)

    print(f"\n{'=' * 60}")
    print(f"  FOLD {fold_idx}: test={test_ids}  val={val_ids}  "
          f"train={train_ids}")
    print(f"{'=' * 60}")

    from data_pipeline.preprocess import preprocess_all, cache_is_complete

    if not cache_is_complete(cfg):
        preprocess_all(cfg)

    if pretrained:
        pretrain_backbone = Path(base_cfg.output_dir) / "pretrained_backbone.pt"
        fold_backbone = fold_dir / "pretrained_backbone.pt"
        if pretrain_backbone.exists() and not fold_backbone.exists():
            import shutil
            shutil.copy2(pretrain_backbone, fold_backbone)

    from train import train
    model, test_m = train(cfg, pretrained=pretrained)

    from evaluate import evaluate
    evaluate(cfg)

    eval_summary_path = fold_dir / "eval" / "eval_summary.json"
    if eval_summary_path.exists():
        with open(eval_summary_path) as f:
            eval_summary = json.load(f)
    else:
        eval_summary = {}

    lead_path = fold_dir / "eval" / "leadtime_by_experiment.csv"
    if lead_path.exists():
        lead_df = pd.read_csv(lead_path)
        mean_lead = lead_df["lead_time"].dropna().mean()
        success_rate = lead_df["success"].mean()
    else:
        mean_lead = np.nan
        success_rate = np.nan

    warn_m = (eval_summary.get("warning_experiment_level") or {}).get("test") or {}
    warn_pooled = (eval_summary.get("warning") or {}).get("test") or {}
    return dict(
        fold=fold_idx,
        test_ids=",".join(test_ids),
        val_ids=",".join(val_ids),
        accuracy=warn_m.get("accuracy", np.nan),
        precision=warn_m.get("precision", np.nan),
        recall=warn_m.get("recall", np.nan),
        f1=warn_m.get("f1", np.nan),
        mcc=warn_m.get("mcc", np.nan),
        auc=warn_m.get("auc", np.nan),
        pr_auc=warn_m.get("pr_auc", np.nan),
        pooled_accuracy=warn_pooled.get("accuracy", np.nan),
        pooled_precision=warn_pooled.get("precision", np.nan),
        pooled_recall=warn_pooled.get("recall", np.nan),
        pooled_f1=warn_pooled.get("f1", np.nan),
        pooled_mcc=warn_pooled.get("mcc", np.nan),
        pooled_auc=warn_pooled.get("auc", np.nan),
        pooled_pr_auc=warn_pooled.get("pr_auc", np.nan),
        best_threshold=eval_summary.get("operating_threshold", np.nan),
        mean_lead_time=mean_lead,
        success_rate=success_rate,
        test_loss=test_m.get("loss", np.nan),
    )


# ───────────────── fold splitting ─────────────────

def _make_folds(
    all_ids: list[str],
    k: int,
    event_times: dict[str, float] | None = None,
) -> list[list[str]]:
    """Distribute experiments into K groups for balanced folds.

    When *event_times* is provided, experiments are sorted by event_time
    and interleaved (earliest, latest, 2nd-earliest, 2nd-latest, …) before
    round-robin assignment.  This ensures each fold gets a mix of early-
    and late-event experiments, preventing two extremes (e.g. exp 5 at 58 s
    and exp 10 at 304 s) from landing in the same test fold.
    """
    if event_times is not None:
        sorted_ids = sorted(
            all_ids,
            key=lambda eid: float(event_times.get(eid, float("inf"))),
        )
        interleaved: list[str] = []
        lo, hi = 0, len(sorted_ids) - 1
        while lo <= hi:
            interleaved.append(sorted_ids[lo])
            lo += 1
            if lo <= hi:
                interleaved.append(sorted_ids[hi])
                hi -= 1
        ordered = interleaved
    else:
        ordered = list(all_ids)

    folds: list[list[str]] = [[] for _ in range(k)]
    for i, eid in enumerate(ordered):
        folds[i % k].append(eid)
    return folds


# ───────────────── main ─────────────────

def main(base_cfg: Config | None = None):
    """Run K-fold CV.

    *base_cfg* — optional pre-built :class:`Config` (e.g. from
    ``run_kfold_tuned.py`` + ``my_tuned_cfg.json``).  CLI flags still
    override individual fields when explicitly passed.
    """
    _fusion_choices = ("layer_bridge", "direct_smoke_temp", "late_concat")
    parser = argparse.ArgumentParser(description="K-Fold CV fire prediction")
    parser.add_argument("--kfold", type=int, default=5,
                        help="Number of folds (default: 5)")
    parser.add_argument("--pretrain", action="store_true",
                        help="Use contrastive pre-trained backbone")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-folds", type=int, default=None,
                        help="For debugging: stop after N folds.")
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip folds that already have eval/eval_summary.json; re-aggregate all folds.",
    )
    parser.add_argument(
        "--start-fold", type=int, default=None,
        help="Only run folds with index >= this value (use with --resume).",
    )
    parser.add_argument(
        "--fusion-mode", type=str, default=None, choices=_fusion_choices,
        help="Fusion ablation (same as train.py)",
    )
    parser.add_argument(
        "--lambda-trend", type=float, default=None,
        help="Trend loss weight; 0 disables slope loss term",
    )
    parser.add_argument(
        "--lambda-consist", type=float, default=None,
        help="Physical consistency loss weight",
    )
    parser.add_argument(
        "--lambda-pred", type=float, default=None,
        help="Prediction loss weight; 0 with trend/consist=0 for warn-only (RQ3)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Base output directory for all folds and summary CSV/JSON",
    )
    args = parser.parse_args()

    k = args.kfold

    cfg = copy.deepcopy(base_cfg) if base_cfg is not None else Config()
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.device is not None:
        cfg.device = args.device
    if args.fusion_mode is not None:
        cfg.fusion_mode = args.fusion_mode
    if args.lambda_trend is not None:
        cfg.lambda_trend = args.lambda_trend
    if args.lambda_consist is not None:
        cfg.lambda_consist = args.lambda_consist
    if args.lambda_pred is not None:
        cfg.lambda_pred = args.lambda_pred
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir

    if base_cfg is not None:
        print(
            "\n[kfold] Using injected Config — "
            f"lr={cfg.lr:g}  train_oversample_factor={cfg.train_oversample_factor:g}  "
            f"lambda_pred={cfg.lambda_pred:g}  lambda_warn={cfg.lambda_warn:g}  "
            f"lambda_trend={cfg.lambda_trend:g}  lambda_consist={cfg.lambda_consist:g}  "
            f"epochs={cfg.epochs}  "
            f"patience={cfg.early_stopping_patience}"
        )

    base_output = cfg.output_dir
    cfg.make_dirs()

    from data_pipeline.preprocess import discover_and_configure
    discover_and_configure(cfg)

    all_ids = list(cfg.full_process_ids)
    n = len(all_ids)
    if k > n:
        raise ValueError(f"kfold={k} exceeds number of experiments ({n})")
    if k < 2:
        raise ValueError(f"kfold={k} must be >= 2")

    from data_pipeline.preprocess import _get_event_times
    event_times = _get_event_times(cfg, all_ids)
    folds = _make_folds(all_ids, k, event_times=event_times)

    print(f"\n[{k}-Fold CV] {n} experiments → {k} folds")
    for i, grp in enumerate(folds):
        print(f"  fold {i}: {grp}")

    if args.pretrain:
        print("\n╔══════════════════════════════════════╗")
        print("║   Contrastive pre-train (post-fire)  ║")
        print("╚══════════════════════════════════════╝")
        from pretrain import pretrain
        pretrain(cfg)

    print(f"\n╔══════════════════════════════════════╗")
    print(f"║        {k}-Fold Cross-Validation        ║")
    print(f"╚══════════════════════════════════════╝")

    results_by_fold: dict[int, dict] = {}
    for fold_idx in range(k):
        if args.max_folds is not None and fold_idx >= args.max_folds:
            break

        test_ids = folds[fold_idx]
        val_ids = folds[(fold_idx + 1) % k]
        train_ids = [
            eid
            for i, group in enumerate(folds)
            for eid in group
            if i != fold_idx and i != (fold_idx + 1) % k
        ]

        cfg.output_dir = base_output
        fold_dir = _fold_dir_path(base_output, fold_idx, test_ids)

        if args.start_fold is not None and fold_idx < args.start_fold:
            if args.resume and _fold_is_complete(fold_dir):
                print(f"\n[resume] fold {fold_idx}: complete (before --start-fold), keeping")
                results_by_fold[fold_idx] = _result_from_completed_fold(
                    fold_idx, test_ids, val_ids, fold_dir,
                )
            continue

        if args.resume and _fold_is_complete(fold_dir):
            print(f"\n[resume] fold {fold_idx}: eval/eval_summary.json exists — skipping train/eval")
            results_by_fold[fold_idx] = _result_from_completed_fold(
                fold_idx, test_ids, val_ids, fold_dir,
            )
            continue

        fold_result = _run_one_fold(
            fold_idx, test_ids, val_ids, train_ids,
            cfg, pretrained=args.pretrain,
        )
        results_by_fold[fold_idx] = fold_result

    if not results_by_fold:
        raise RuntimeError("No fold results collected (check --max-folds / --start-fold).")

    # ── aggregate ──
    results_df = pd.DataFrame([results_by_fold[i] for i in sorted(results_by_fold)])
    results_path = Path(base_output) / f"{k}fold_results.csv"
    results_df.to_csv(results_path, index=False)

    print("\n" + "=" * 60)
    print(f"  {k}-FOLD CV SUMMARY")
    print("=" * 60)
    print(results_df.to_string(index=False))

    exp_metrics = ["accuracy", "precision", "recall", "f1", "mcc", "auc", "pr_auc",
                   "mean_lead_time", "success_rate"]
    pooled_metrics = ["pooled_accuracy", "pooled_precision", "pooled_recall",
                      "pooled_f1", "pooled_mcc", "pooled_auc", "pooled_pr_auc"]

    print("\n── Experiment-level macro-average (mean ± std) ──")
    for m in exp_metrics:
        vals = results_df[m].dropna()
        if len(vals) > 0:
            print(f"  {m:20s} = {vals.mean():.4f} ± {vals.std():.4f}")

    print("\n── Window-level pooled (mean ± std) ──")
    for m in pooled_metrics:
        vals = results_df[m].dropna()
        if len(vals) > 0:
            label = m.replace("pooled_", "")
            print(f"  {label:20s} = {vals.mean():.4f} ± {vals.std():.4f}")

    summary = {
        m: {"mean": float(results_df[m].dropna().mean()),
            "std": float(results_df[m].dropna().std())}
        for m in exp_metrics if results_df[m].notna().any()
    }
    summary["pooled"] = {
        m.replace("pooled_", ""): {
            "mean": float(results_df[m].dropna().mean()),
            "std": float(results_df[m].dropna().std()),
        }
        for m in pooled_metrics if results_df[m].notna().any()
    }
    summary["n_folds"] = k
    summary["n_experiments"] = n
    summary["training_config"] = {
        "fusion_mode": getattr(cfg, "fusion_mode", "layer_bridge"),
        "lr": cfg.lr,
        "train_oversample_factor": cfg.train_oversample_factor,
        "lambda_pred": cfg.lambda_pred,
        "lambda_warn": cfg.lambda_warn,
        "lambda_trend": cfg.lambda_trend,
        "lambda_consist": cfg.lambda_consist,
        "epochs": cfg.epochs,
        "early_stopping_patience": cfg.early_stopping_patience,
        "pretrained": bool(args.pretrain),
        "output_dir": base_output,
    }
    summary["metric_note"] = (
        "Primary metrics (accuracy, precision, recall, f1, mcc, etc.) use "
        "experiment-level macro-averaging; 'pooled' section contains "
        "window-level pooled metrics for reference."
    )
    summary["fold_assignments"] = [
        {
            "fold": i,
            "test": folds[i],
            "val": folds[(i + 1) % k],
            "train": [
                eid for j, g in enumerate(folds) for eid in g
                if j != i and j != (i + 1) % k
            ],
        }
        for i in range(k)
    ]
    summary_path = Path(base_output) / f"{k}fold_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nResults saved to {results_path}")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()

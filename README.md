# Fire — Early Fire Warning & Evolution Analysis

This repository contains the **full code and data needed to reproduce** the regularized main pipeline from scratch. The `summary/` folder is generated after warning-track training completes.

The dual-track design separates **warning** from **evolution**, avoiding mixed supervision: the warning track does not use post-fire labels, while the evolution track systematically measures time-to-peak and time-to-safe concentrations. **Run all commands from the `Fire/` directory.**

---

## Task Overview

| Track | Data | Objective | Output Directory |
|------|------|------|----------|
| **Warning** | 10 `complete_full` / `reference_long` runs (ids 1–9 + experiment 10 long-duration reference) | F1 / MCC / Recall / lead time | `dual_track_analysis/outputs/warning/` |
| **Evolution** | 70 post-fire + 10 full (80 total, `usable_for_evolution=true`) | Ignition→peak, peak→safe concentration duration | `dual_track_analysis/outputs/evolution*` |

Experiment lists come from `dataset_ready/unified_1hz/meta.csv` (consistent with `dual_track_analysis/common/registry.py`).

---

## Directory Structure

```
Fire/
├── README.md
├── requirements.txt
├── config.py                    # Global Config
├── train.py / evaluate.py       # Single-fold training & evaluation
├── run_kfold.py / run_kfold_tuned.py
├── checkpoint_io.py
├── analysis/                    # Eval export & threshold sensitivity (4 scripts)
├── data_pipeline/               # Preprocessing, labels, Dataset
├── models/                      # LBCA (Ours) + 9 baselines
├── dataset_ready/               # unified_1hz + usable_csv
├── processed_data/              # Preprocessing cache (safe to delete; rebuilt automatically)
└── dual_track_analysis/
    ├── run_all.py               # One-shot: registry + evolution [+ optional training]
    ├── config.json              # Evolution-track params (smooth_window, etc.)
    ├── common/                  # Experiment registry, CSV locator
    ├── warning/                 # Regularized 5-fold, baselines, ablations, export
    ├── evolution/               # key_moments, run_extract
    ├── analysis/                # fire_evolution_timescale_analysis
    └── outputs/                 # Run artifacts
```

---

## Environment & Prerequisites

```bash
cd Fire
pip install -r requirements.txt
```

Requires Python 3.10+ and PyTorch (CUDA recommended).

| Path | Description |
|------|------|
| `dataset_ready/unified_1hz/` | Unified 1 Hz CSVs + `meta.csv` |
| `dataset_ready/usable_csv/` | Raw usable CSVs |
| `processed_data/` | Training preprocessing cache |

**Main config** `config.py` (regularized warning-track defaults):

- `post_fire_as_supervised_neg=false` — post-fire data excluded from warning supervision
- `dropout=0.35`, `head_dropout=0.50`, `weight_decay=0.003`
- `lambda_trend=0.0`, `eval_threshold_strategy=constrained_f1` (experiment-level threshold selection)
- `epochs=50`, `device=cuda`

Each fold writes `run_config.json` for reproducibility. Ablations can override via `--config overrides.json` or `--set KEY=VALUE`.

Ablation definitions: `dual_track_analysis/warning/ablation_study_regularized.json`.

---

## Full Retrain from Scratch (Recommended Order)

### 1. Evolution analysis (lightweight, no GPU)

```bash
python -m dual_track_analysis.run_all
```

Outputs (all under `dual_track_analysis/outputs/`):
- `registry/experiment_registry.csv`
- `evolution/key_moments_long.csv`
- `evolution_analysis/evolution_event_table.csv`
- `evolution_analysis/final/` — summary tables + `figures/fig_timescale_statistics.*`

Evolution only:

```bash
python -m dual_track_analysis.evolution.run_extract
python -m dual_track_analysis.analysis.fire_evolution_timescale_analysis
```

### 2. Main model LBCA 5-fold

```bash
python -m dual_track_analysis.warning.run_kfold
```

Explicit output directory:

```bash
python -m dual_track_analysis.warning.run_kfold \
  --output-dir dual_track_analysis/outputs/warning/kfold_regularized
```

- Automatically calls `export_results` → `outputs/warning/summary/` after training
- Resume: add `--resume` (skips folds that already have `eval/eval_summary.json`)
- Single-fold debug: `--set epochs=5 --max-folds 1`

### 3. Baseline 5-fold (9 models)

```bash
python -m dual_track_analysis.warning.run_baselines
```

Resume: `--resume --skip-existing`. Rebuild comparison tables only:

```bash
python -m dual_track_analysis.warning.export_baseline_comparison
```

### 4. LBCA ablations

```bash
python -m dual_track_analysis.warning.run_ablations --dry-run

python -m dual_track_analysis.warning.run_ablations \
  --only wo_bridge,wo_cross_attn,wo_consist,wo_pred
```

Rebuild comparison tables only: `python -m dual_track_analysis.warning.export_ablation_comparison`

### 5. Aggregation & sensitivity

```bash
python -m dual_track_analysis.warning.export_results \
  --kfold-root dual_track_analysis/outputs/warning/kfold_regularized

python -m dual_track_analysis.warning.export_baseline_comparison
python -m dual_track_analysis.warning.export_ablation_comparison
python -m dual_track_analysis.warning.export_sensitivity --threshold
```

### 6. Re-evaluate only (change threshold strategy, no retraining)

```bash
python -m dual_track_analysis.warning.reeval_kfold
python -m dual_track_analysis.warning.reeval_kfold \
  --set eval_threshold_strategy=constrained_f1
```

---

## One-Shot Entry Points

```bash
python -m dual_track_analysis.run_all                                    # evolution only
python -m dual_track_analysis.run_all --with-warning-kfold               # + main model
python -m dual_track_analysis.run_all --with-warning-kfold --with-baselines
```

---

## Clean Old Results Before Retraining

```powershell
Remove-Item -Recurse -Force `
  dual_track_analysis\outputs\warning\kfold_regularized, `
  dual_track_analysis\outputs\warning\baseline_regularized, `
  dual_track_analysis\outputs\warning\ablation_regularized, `
  dual_track_analysis\outputs\warning\summary `
  -ErrorAction SilentlyContinue
```

When retraining from scratch, **do not** use `--resume` / `--skip-existing`.

---

## Output Layout

```
dual_track_analysis/outputs/
  registry/
  evolution/key_moments_long.csv
  evolution_analysis/
    evolution_event_table.csv
    final/                           # summary tables + fig_timescale_statistics
  warning/
    kfold_regularized/               # Ours 5-fold
    baseline_regularized/            # 9 baselines
    ablation_regularized/            # ablations
    summary/                         # primary metrics & sensitivity (generated after training)
    window_sweep/  tune/             # optional: generated after sensitivity sweeps
```

| Content | Path |
|------|------|
| Main model 5-fold | `.../warning/kfold_regularized/` |
| Primary paper metrics | `.../warning/summary/warning_primary_summary.*` |
| Threshold sensitivity | `.../warning/summary/sensitivity/threshold/` |
| Evolution composite figure | `.../evolution_analysis/final/figures/fig_timescale_statistics.*` |

Key files per fold: `best_model.pt`, `run_config.json`, `eval/warning_primary_metrics.json`, `eval/threshold_sensitivity_*.csv`.

---

## Evolution Metric Definitions (`dual_track_analysis/evolution/key_moments.py`)

For each experiment and each variable (CO, Trans, Heat_max, T_max):

1. **t_origin** — `event_time` for `complete_full`; `time[0]` for `post_fire_only`
2. **t_first_danger** — first sustained crossing of hazard threshold per variable (CO ≥ 200 ppm; Trans ≤ 0.55; temperature ≥ 60 °C; heat flux ≥ 2.5 kW/m²)
3. **t_peak** — global extremum time on the smoothed curve for `t >= t_origin`
4. **t_safe** — after the peak, sustained `sustain_seconds` below safe threshold (CO < 50 ppm; Trans < 0.3; temperature variables fall to peak × 0.35)
5. **Exported durations** — `time_to_peak_s` = t_peak − t_origin; `time_peak_to_safe_s` = t_safe − t_peak (NaN = not recovered)

Parameters: `dual_track_analysis/config.json`. Summary statistics and the 2×2 composite figure are produced by `dual_track_analysis/analysis/fire_evolution_timescale_analysis.py`. Recovery time depends on `t_safe`; unrecovered samples are recorded as NaN.

---

## Training Pipeline

```
dual_track_analysis.warning.run_kfold
  → run_kfold_tuned.py → run_kfold.py → train.py / evaluate.py
```

The warning track enforces `post_fire_as_supervised_neg=false` and `include_post_fire_regression=false`.

---

## Notes

- The default workflow does not use contrastive pre-training (`--pretrain`); training is from scratch.
- In the repository, `outputs/` contains only `.gitkeep` placeholders; running the pipeline writes CSVs, figures, and `best_model.pt`.
- `window_sweep/` and `tune/` are optional sensitivity-sweep outputs and are not part of the main training path.
- If `processed_data/` is deleted, the preprocessing cache is rebuilt automatically on the first training run.

# Fire — 火灾早期预警与演化分析

本目录为 **从零复现** 所需的完整代码与数据（regularized 主流程）；预警训练完成后会生成 `summary/` 汇总。

双轨设计将 **预警** 与 **演化** 从训练数据混用中拆开：预警不用 post-fire 监督，演化系统化统计达峰/恢复时间。**所有命令在 `Fire/` 目录下执行。**

---

## 任务划分

| 轨道 | 数据 | 目标 | 输出目录 |
|------|------|------|----------|
| **预警 Warning** | 10 份 `complete_full` / `reference_long`（id 1–9 + 实验 10 长时程参考） | F1 / MCC / Recall / lead time | `dual_track_analysis/outputs/warning/` |
| **演化 Evolution** | 70 post-fire + 10 full（共 80，`usable_for_evolution=true`） | 着火→峰值、峰值→安全浓度时长 | `dual_track_analysis/outputs/evolution*` |

实验清单来自 `dataset_ready/unified_1hz/meta.csv`（与 `dual_track_analysis/common/registry.py` 一致）。

---

## 目录结构

```
Fire/
├── README.md
├── requirements.txt
├── config.py                    # 全局 Config
├── train.py / evaluate.py       # 单折训练与评估
├── run_kfold.py / run_kfold_tuned.py
├── checkpoint_io.py
├── analysis/                    # 评估导出与阈值敏感性（4 个脚本）
├── data_pipeline/               # 预处理、标签、Dataset
├── models/                      # LBCA（Ours）+ 9 基线
├── dataset_ready/               # unified_1hz + usable_csv
├── processed_data/              # 预处理缓存（可删后自动重建）
└── dual_track_analysis/
    ├── run_all.py               # 一键：registry + 演化 [+ 可选训练]
    ├── config.json              # 演化轨参数（smooth_window 等）
    ├── common/                  # 实验注册、CSV 定位
    ├── warning/                 # regularized 5-fold、基线、消融、导出
    ├── evolution/               # key_moments、run_extract
    ├── analysis/                # fire_evolution_timescale_analysis
    └── outputs/                 # 运行结果
```

---

## 环境与前置条件

```bash
cd Fire
pip install -r requirements.txt
```

需要 Python 3.10+、PyTorch（建议 CUDA）。

| 路径 | 说明 |
|------|------|
| `dataset_ready/unified_1hz/` | 统一 1 Hz CSV + `meta.csv` |
| `dataset_ready/usable_csv/` | 原始可用 CSV |
| `processed_data/` | 训练预处理缓存 |

**主配置** `config.py`（regularized 预警轨默认值）：

- `post_fire_as_supervised_neg=false` — post-fire 不参与预警监督
- `dropout=0.35`, `head_dropout=0.50`, `weight_decay=0.003`
- `lambda_trend=0.0`，`eval_threshold_strategy=constrained_f1`（实验级选阈）
- `epochs=50`，`device=cuda`

每折训练后写入 `run_config.json` 供复现；消融可用 `--config overrides.json` 或 `--set KEY=VALUE` 覆盖。

消融定义见 `dual_track_analysis/warning/ablation_study_regularized.json`。

---

## 从零完整重训（推荐顺序）

### 1. 演化分析（轻量，无需 GPU）

```bash
python -m dual_track_analysis.run_all
```

产出（均在 `dual_track_analysis/outputs/` 下）：
- `registry/experiment_registry.csv`
- `evolution/key_moments_long.csv`
- `evolution_analysis/evolution_event_table.csv`
- `evolution_analysis/final/` — 统计表 + `figures/fig_timescale_statistics.*`

仅重跑演化：

```bash
python -m dual_track_analysis.evolution.run_extract
python -m dual_track_analysis.analysis.fire_evolution_timescale_analysis
```

### 2. 主模型 LBCA 5-fold

```bash
python -m dual_track_analysis.warning.run_kfold
```

等价显式指定输出目录：

```bash
python -m dual_track_analysis.warning.run_kfold \
  --output-dir dual_track_analysis/outputs/warning/kfold_regularized
```

- 训练完自动调用 `export_results` → `outputs/warning/summary/`
- 断点续跑：加 `--resume`（跳过已有 `eval/eval_summary.json` 的 fold）
- 单折调试：`--set epochs=5 --max-folds 1`

### 3. 基线 5-fold（9 个模型）

```bash
python -m dual_track_analysis.warning.run_baselines
```

断点续跑：`--resume --skip-existing`。仅重建对比表：

```bash
python -m dual_track_analysis.warning.export_baseline_comparison
```

### 4. LBCA 消融

```bash
python -m dual_track_analysis.warning.run_ablations --dry-run

python -m dual_track_analysis.warning.run_ablations \
  --only wo_bridge,wo_cross_attn,wo_consist,wo_pred
```

仅重建对比表：`python -m dual_track_analysis.warning.export_ablation_comparison`

### 5. 汇总与敏感性

```bash
python -m dual_track_analysis.warning.export_results \
  --kfold-root dual_track_analysis/outputs/warning/kfold_regularized

python -m dual_track_analysis.warning.export_baseline_comparison
python -m dual_track_analysis.warning.export_ablation_comparison
python -m dual_track_analysis.warning.export_sensitivity --threshold
```

### 6. 仅重跑 evaluate（改阈值策略，不重训）

```bash
python -m dual_track_analysis.warning.reeval_kfold
python -m dual_track_analysis.warning.reeval_kfold \
  --set eval_threshold_strategy=constrained_f1
```

---

## 一键入口

```bash
python -m dual_track_analysis.run_all                                    # 仅演化
python -m dual_track_analysis.run_all --with-warning-kfold               # + 主模型
python -m dual_track_analysis.run_all --with-warning-kfold --with-baselines
```

---

## 清空旧结果后重训

```powershell
Remove-Item -Recurse -Force `
  dual_track_analysis\outputs\warning\kfold_regularized, `
  dual_track_analysis\outputs\warning\baseline_regularized, `
  dual_track_analysis\outputs\warning\ablation_regularized, `
  dual_track_analysis\outputs\warning\summary `
  -ErrorAction SilentlyContinue
```

从零重训时 **不要加** `--resume` / `--skip-existing`。

---

## 输出目录

```
dual_track_analysis/outputs/
  registry/
  evolution/key_moments_long.csv
  evolution_analysis/
    evolution_event_table.csv
    final/                           # 统计表 + fig_timescale_statistics
  warning/
    kfold_regularized/               # Ours 5-fold
    baseline_regularized/            # 9 基线
    ablation_regularized/            # 消融
    summary/                         # 论文主指标、敏感性（训练后生成）
    window_sweep/  tune/             # 可选：敏感性 sweep 后生成
```

| 内容 | 路径 |
|------|------|
| 主模型 5-fold | `.../warning/kfold_regularized/` |
| 论文主指标 | `.../warning/summary/warning_primary_summary.*` |
| 阈值敏感性 | `.../warning/summary/sensitivity/threshold/` |
| 演化合图 | `.../evolution_analysis/final/figures/fig_timescale_statistics.*` |

每个 fold 内关键文件：`best_model.pt`、`run_config.json`、`eval/warning_primary_metrics.json`、`eval/threshold_sensitivity_*.csv`。

---

## 演化指标定义（`dual_track_analysis/evolution/key_moments.py`）

对每个实验、每个变量（CO、Trans、Heat_max、T_max）：

1. **t_origin** — `complete_full` 用 `event_time`；`post_fire_only` 用 `time[0]`
2. **t_first_danger** — 各变量独立首次持续越危险阈（CO ≥ 200 ppm；Trans ≤ 0.55；温度 ≥ 60 °C；热流 ≥ 2.5 kW/m²）
3. **t_peak** — `t >= t_origin` 上平滑曲线全局极值时刻
4. **t_safe** — 峰后持续 `sustain_seconds` 秒低于安全阈（CO < 50 ppm；Trans < 0.3；温度类回落至峰值 × 0.35）
5. **导出时长** — `time_to_peak_s` = t_peak − t_origin；`time_peak_to_safe_s` = t_safe − t_peak（NaN = 未恢复）

参数见 `dual_track_analysis/config.json`。统计与 2×2 合图由 `dual_track_analysis/analysis/fire_evolution_timescale_analysis.py` 生成；恢复时间依赖 `t_safe`，未恢复样本记 NaN。

---

## 训练链路

```
dual_track_analysis.warning.run_kfold
  → run_kfold_tuned.py → run_kfold.py → train.py / evaluate.py
```

预警轨强制 `post_fire_as_supervised_neg=false`、`include_post_fire_regression=false`。

---

## 说明

- 默认流程不使用对比预训练（`--pretrain`）；主流程为从零训练。
- 仓库内 `outputs/` 仅含 `.gitkeep` 占位，运行后会写入 CSV、图表与 `best_model.pt`。
- `window_sweep/`、`tune/` 为可选敏感性 sweep 输出，不参与主训练。
- 删除 `processed_data/` 后，首次训练会自动重建预处理缓存。

"""
Central configuration for the fire prediction & early-warning system.

All regularized warning-track defaults live here (formerly
``warning_regularized_cfg.json``).  Per-fold ``run_config.json`` is written
by ``train.py`` for reproducibility; override at runtime via CLI ``--set``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def parse_config_value(raw: str):
    """Parse ``KEY=VALUE`` strings from CLI ``--set`` overrides."""
    s = raw.strip()
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    try:
        if "." in s or "e" in s.lower():
            return float(s)
        return int(s)
    except ValueError:
        return s


@dataclass
class Config:
    # ──────────────── Paths ────────────────
    raw_csv_dir: str = "./dataset_ready/usable_csv"
    processed_dir: str = "./processed_data"
    output_dir: str = "./trimodal_output"

    # ──────────────── Auto-discovery ────────────────
    auto_discover: bool = True

    # ──────────────── Experiment selection ────────────────
    # Populated by auto-discovery, or set manually.
    # Auto-discovery scans all CSVs and finds those with enough pre-fire
    # data for supervised warning-label training.
    full_process_ids: List[str] = field(default_factory=list)
    train_ids: List[str] = field(default_factory=list)
    val_ids:   List[str] = field(default_factory=list)
    test_ids:  List[str] = field(default_factory=list)

    # Train / val / test split by *experiment id* (must sum to 1.0).
    # Uses Hamilton rounding so counts add up to n_full. For small pools (e.g. 10
    # experiments) this yields more val+test than 15%/15% (often 3+3+4).
    split_train_ratio: float = 0.50
    split_val_ratio: float = 0.25
    split_test_ratio: float = 0.25

    # Maps experiment IDs to CSV paths when the file is not named "{id}_clean.csv".
    # fire_merged_1hz_reference: stem from usable_csv glob (no _clean suffix).
    extra_csv_map: Dict = field(default_factory=lambda: {
        "10": "./dataset_ready/usable_csv/fire_merged_1hz_reference.csv",
        "fire_merged_1hz_reference": "./dataset_ready/usable_csv/fire_merged_1hz_reference.csv",
    })

    # Post-fire-start experiments (auto-populated; used only for pretraining)
    post_fire_ids: List[str] = field(default_factory=list)

    # ──────────────── Sensor layout ────────────────
    n_temp_channels: int = 160
    n_layers: int = 4
    channels_per_layer: int = 40
    smoke_col_names: List[str] = field(default_factory=lambda: [
        "CO", "Trans", "1D", "0.5D", "1.5D",
    ])

    # ──────────────── Optical physics ────────────────
    optical_path_length: float = 1.0
    transmittance_is_percent: bool = False
    eps: float = 1e-6

    # ──────────────── Hazard thresholds ────────────────
    # hazard(t)：层域温度、热流、CO、Trans 并联；滚动窗口 sustain_seconds 内
    # 至少 hazard_window_fraction 的采样点达阈则当前时刻危险。
    #   层域温度 max(layer_mean) >= temp_threshold (°C)
    #   热流 max(q) >= heat_threshold (kW/m²)
    #   CO >= co_threshold (ppm)
    #   Trans <= trans_threshold（透光率）
    # event_time / warn 锚点默认仅用 CO+Trans（同 80%/20s 规则）。
    co_threshold: float = 200.0
    trans_threshold: float = 0.55
    temp_threshold: float = 60.0
    heat_threshold: float = 2.5
    heat_col_candidates: List[str] = field(default_factory=lambda: [
        "1D", "0.5D", "1.5D",
    ])
    event_time_metrics: List[str] = field(default_factory=lambda: ["CO", "Trans"])
    min_start_time: int = 20
    sustain_seconds: int = 20
    hazard_window_fraction: float = 0.8

    # ──────────────── Windows ────────────────
    # w: used by FirePredictionDataset, contrastive windows, and model input length (T=w).
    # w=50: more pre-hazard windows when event_time is early (vs w=60).
    # warning_delta: label only (0 < event_time−t <= Δ); may differ from w (e.g. Δ=60, w=50).
    history_window: int = 50
    prediction_horizon: int = 15    # H  (future steps to predict)
    warning_delta: int = 60         # Δ  (seconds ahead for binary warning)
    # full_process discovery: require event_time >= this (seconds). None → history_window - 2.
    # Set e.g. 58 to keep the same full/post split as under w=60 while training with w=50.
    discover_min_pre_fire: Optional[int] = None

    # ──────────────── Normalisation ────────────────
    log_transform_cols: List[str] = field(default_factory=lambda: [
        "CO", "1D", "0.5D", "1.5D",
    ])

    # ──────────────── Model architecture ────────────────
    d_model: int = 64
    n_heads: int = 4
    d_ff: int = 128
    n_spatial_layers: int = 2       # spatial self-attn in TempFieldEncoder
    n_temporal_layers: int = 2      # temporal self-attn shared across encoders
    n_cross_attn_stages: int = 2    # repetitions of the 3-step cross-attn
    # "layer_bridge" — full 3-step bridge + gate (default).
    # "direct_smoke_temp" — ablation: only smoke→temp cross-attn, no bridge chain.
    # "late_concat" — ablation: no cross-attn; concat pooled modalities + MLP.
    fusion_mode: str = "layer_bridge"
    dropout: float = 0.35           # encoder / fusion Transformer layers
    head_dropout: float = 0.50      # prediction & warning head MLP (anti-overfit)
    pooling: str = "last"           # "last" | "mean"

    # ──────────────── Baseline models (for comparison vs TriModalFireModel) ────────────────
    # "trimodal"         — default TriModalFireModel (cross-attention fusion).
    # "lstm" / "bilstm" / "gru" — concat features → RNN backbone.
    # "tcn"              — concat features → dilated causal Conv1D.
    # "informer"         — concat features → ProbSparse self-attention encoder.
    # "transformer_noxa" — concat features → plain Transformer encoder
    #                      (self-attention only, no cross-attention).
    # "patchtst"         — patch + Transformer encoder (Nie et al. 2023).
    # "itransformer"     — inverted Transformer on variate tokens (Liu et al. 2024).
    # "timesnet"         — FFT multi-period 2D inception blocks (Wu et al. 2023).
    model_type: str = "trimodal"
    baseline_dropout: float = 0.3
    baseline_rnn_layers: int = 2            # LSTM / GRU depth
    baseline_bidirectional: bool = False    # bi-directional RNN (fair early-warning = False)
    baseline_tcn_levels: int = 4            # number of dilated-conv residual blocks
    baseline_tcn_kernel: int = 3
    baseline_transformer_layers: int = 2    # Transformer / PatchTST / iTransformer / TimesNet depth
    baseline_informer_factor: int = 5       # Informer ProbSparse sampling factor c
    baseline_patch_len: int = 8             # PatchTST patch length (T=history_window)
    baseline_patch_stride: int = 4          # PatchTST stride
    baseline_timesnet_top_k: int = 3        # TimesNet top-k periods from FFT

    # ──────────────── Prediction targets ────────────────
    predict_smoke: bool = True      # CO, k, heat cols
    predict_layer_means: bool = True  # 4 layer means

    # ──────────────── Training ────────────────
    # NOTE: batch size heavily affects GPU memory usage.
    # Lower it temporarily when debugging to avoid CUDA OOM.
    use_amp: bool = True              # mixed-precision (fp16) training
    batch_size: int = 32
    epochs: int = 50
    lr: float = 0.001
    weight_decay: float = 0.003     # L2 on weights (higher → less regression overfit)
    early_stopping_patience: int = 8
    warmup_epochs: int = 5
    seed: int = 42
    device: str = "cuda"            # "auto" | "cuda" | "cpu"
    # Train DataLoader uses WeightedRandomSampler(replacement=True) to balance
    # warn pos/neg.  num_samples = int(len(train_ds) * train_oversample_factor).
    # >1.0 → more draws per epoch (minority windows seen more often); does not
    # create new independent data—use modest values (e.g. 1.5–2) to limit overfit.
    # Closer to 1.0 = less repeated minority draws → less memorization (was 1.5).
    train_oversample_factor: float = 1.0
    # Cap sampler weight ratio w_pos/w_neg to limit extreme oversampling of warn=1
    # (reduces overfitting → better precision / F1 at similar recall).
    sampler_max_class_weight_ratio: float = 4.0
    # Include post-hazard windows (hazard=1) as warn=0 negatives in supervised
    # training.  Conceptually correct: once danger has already arrived, the
    # system no longer needs to issue a *warning*; these are valid negative
    # examples and roughly double the usable windows per experiment.
    use_post_hazard_as_negative: bool = True

    # Per-experiment balanced sampling: weight each experiment equally so that
    # large experiments (e.g. exp 10 with 1609 windows) don't dominate training.
    # Within each experiment, pos/neg classes are further balanced.
    experiment_balanced_sampling: bool = True
    # Use post-fire experiments as supervised negatives (warn=0, warn_valid=1)
    # rather than prediction_only.  Adds ~60k diverse negative windows from 69
    # experiments, teaching the model "active fire ≠ warning needed".
    post_fire_as_supervised_neg: bool = False
    # Add post-fire windows to supervised train loader (prediction_only or neg above).
    # False → train on full_process only (faster warning CV; regression head sees less data).
    include_post_fire_regression: bool = False

    # If True, train loader applies light jitter/mask/dropout (same idea as pretrain).
    supervised_train_augment: bool = True

    # DataLoader parallelism: >0 enables multi-process data loading.
    num_workers: int = 4
    pin_memory: bool = True

    # ──────────────── Debugging ────────────────
    # If True, training/evaluation will abort when NaN/Inf appears in
    # model outputs or loss. Useful for diagnosing bad folds.
    debug_nan_checks: bool = False

    # ──────────────── Loss weights ────────────────
    lambda_pred: float = 0.3
    lambda_warn: float = 1.5        # balance precision vs recall on imbalanced windows
    lambda_trend: float = 0.0       # disabled in regularized warning track
    lambda_consist: float = 0.05
    logvar_reg: float = 0.15
    label_smoothing: float = 0.05
    # Focal BCE alpha: lower → less up-weight of positives (helps precision/F1).
    warn_focal_alpha: float = 0.55
    warn_focal_gamma_neg: float = 4.0
    # Cap BCE pos_weight = n_neg/n_pos in CombinedLoss (train.py) when imbalance is extreme.
    warn_pos_weight_cap: float = 25.0

    # ──────────────── Evaluation (thresholds use validation only) ────────────────
    # When merging thresholds, require at most this false-positive rate on val negatives
    # (prob threshold = (1 - cap) quantile of calibrated scores where warn==0).
    # Lower cap → higher merged threshold → fewer false alarms (may lower recall).
    eval_val_fpr_cap: float = 0.05
    # Legacy floor (unused in PR-based selection; kept for backward compat).
    # The operating threshold is now max-F1 on validation (Precision-Recall
    # criterion), so this floor is no longer applied.
    eval_operating_min_threshold: float = 0.0
    # Threshold selection on validation (test never used for tuning):
    #   max_f1            — val max-F1 (balanced P/R at window or experiment level)
    #   high_recall       — recall >= eval_high_recall_min_recall, then best secondary
    #   max_mcc           — val max MCC
    #   constrained_mcc   — recall floor then max MCC (recommended for high-R + high-P)
    #   constrained_f1    — recall floor then max F1
    #   constrained_precision — recall floor then max precision
    # Legacy: eval_prefer_high_recall=true maps to strategy "high_recall" when unset.
    eval_threshold_strategy: str = "constrained_f1"
    # "window" (pooled windows) or "experiment" (macro per experiment; matches primary metrics)
    eval_threshold_level: str = "experiment"
    eval_prefer_high_recall: bool = False
    eval_high_recall_min_recall: float = 1.0
    eval_high_recall_secondary: str = "f1"  # "precision" | "mcc" | "f1" | "accuracy"

    # ──────────────── Contrastive pre-training ────────────────
    # Data source (no leakage for supervised / LOOCV):
    #   "post_fire_only" — only cfg.post_fire_ids (disjoint from full_process_ids
    #       after discover). Single pretrain run; safe for all evaluation folds.
    #   "all_csv" — every CSV under raw_csv_dir (LEAKY: includes future test exps).
    contrastive_data_mode: str = "post_fire_only"
    pretrain_epochs: int = 50
    pretrain_lr: float = 5e-4
    pretrain_batch_size: int = 64
    pretrain_patience: int = 10
    contrastive_temp: float = 0.07       # NT-Xent temperature τ
    contrastive_proj_dim: int = 64       # projection head output dim
    lambda_instance: float = 1.0         # instance-level NT-Xent weight
    lambda_cross_modal: float = 0.5      # cross-modal alignment weight
    # augmentation probabilities
    aug_jitter_std: float = 0.07
    aug_channel_mask_ratio: float = 0.25  # fraction of temp channels to mask
    aug_temporal_mask_ratio: float = 0.2  # fraction of time steps to mask
    aug_smoke_dropout_prob: float = 0.3  # prob of zeroing one smoke channel
    # fine-tuning after pre-training
    finetune_lr_backbone: float = 1e-4   # lower lr for pre-trained layers
    finetune_lr_head: float = 1e-3       # higher lr for new heads

    # ──────────────── Derived (read-only) ────────────────
    @property
    def n_smoke_channels(self) -> int:
        return len(self.smoke_col_names)

    @property
    def n_layer_features(self) -> int:
        """mean, max, std, slope per layer."""
        return 4

    @property
    def n_pred_targets(self) -> int:
        n = 0
        if self.predict_smoke:
            n += self.n_smoke_channels
        if self.predict_layer_means:
            n += self.n_layers
        return n

    def resolve_device(self) -> str:
        if self.device == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device

    def _split_sizes(self, n: int) -> tuple:
        """Return (n_train, n_val, n_test) with sum == n."""
        if n <= 0:
            return 0, 0, 0
        if n == 1:
            return 1, 0, 0
        if n == 2:
            return 1, 1, 0

        r_tr = float(self.split_train_ratio)
        r_va = float(self.split_val_ratio)
        r_te = float(self.split_test_ratio)
        s = r_tr + r_va + r_te
        if abs(s - 1.0) > 1e-5:
            raise ValueError(
                f"split_train/val/test_ratio must sum to 1.0, got {s}"
            )
        r_tr, r_va, r_te = r_tr / s, r_va / s, r_te / s

        items = [
            ("tr", n * r_tr),
            ("va", n * r_va),
            ("te", n * r_te),
        ]
        floors = {k: int(v) for k, v in items}
        rem = n - sum(floors.values())
        order = sorted(items, key=lambda kv: kv[1] - int(kv[1]), reverse=True)
        for i in range(rem):
            floors[order[i][0]] += 1

        n_tr, n_va, n_te = floors["tr"], floors["va"], floors["te"]

        # Need val+test for threshold tuning and evaluation when n >= 3
        if n >= 3:
            while n_te < 1 and n_tr > 1:
                n_tr -= 1
                n_te += 1
            while n_va < 1 and n_tr > 1:
                n_tr -= 1
                n_va += 1
            while n_tr < 1 and (n_va > 1 or n_te > 1):
                if n_va >= n_te and n_va > 1:
                    n_va -= 1
                elif n_te > 1:
                    n_te -= 1
                else:
                    break
                n_tr += 1

        return n_tr, n_va, n_te

    def setup_splits(
        self,
        full_ids: List[str],
        event_times: Optional[Dict[str, float]] = None,
        seed: Optional[int] = None,
    ):
        """Split experiments into train / val / test.

        When *event_times* is provided, experiments are ordered by event_time
        and merged as [earliest, latest, 2nd-earliest, 2nd-latest, ...] so
        train / val / test each see a mix of early- and late-event runs.
        Assigning only late events to val/test would leave train with almost
        no warn=0 windows (hazard masks everything after event).
        """
        import random as _rnd
        rng = _rnd.Random(seed if seed is not None else self.seed)

        n = len(full_ids)
        n_tr, n_va, n_te = self._split_sizes(n)

        if event_times is not None and n >= 3:
            sv = sorted(
                full_ids,
                key=lambda eid: float(event_times.get(eid, float("nan"))),
            )
            merged: List[str] = []
            i, j = 0, len(sv) - 1
            while i <= j:
                merged.append(sv[i])
                i += 1
                if i <= j:
                    merged.append(sv[j])
                    j -= 1
            self.test_ids = merged[:n_te]
            self.val_ids = merged[n_te : n_te + n_va]
            self.train_ids = merged[n_te + n_va :]
        else:
            ids = list(full_ids)
            rng.shuffle(ids)
            self.test_ids = ids[:n_te]
            self.val_ids = ids[n_te : n_te + n_va]
            self.train_ids = ids[n_te + n_va :]

    def make_dirs(self):
        for d in [self.processed_dir, self.output_dir]:
            Path(d).mkdir(parents=True, exist_ok=True)

    def apply_overrides(self, overrides: dict) -> "Config":
        """Apply ``{field: value}`` overrides; unknown keys raise."""
        for key, value in overrides.items():
            if not hasattr(self, key):
                raise AttributeError(f"unknown Config field '{key}'")
            setattr(self, key, value)
        return self

    def apply_sets(self, items: Iterable[str]) -> "Config":
        """Apply CLI-style ``KEY=VALUE`` strings."""
        for item in items:
            if "=" not in item:
                raise ValueError(f"expected KEY=VALUE, got {item!r}")
            key, raw = item.split("=", 1)
            key = key.strip()
            if not hasattr(self, key):
                raise AttributeError(f"unknown Config field '{key}'")
            setattr(self, key, parse_config_value(raw))
        return self

    @classmethod
    def from_json(cls, path: str | Path) -> Tuple["Config", dict]:
        """Load optional JSON overrides on top of defaults (for ablations)."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        cfg = cls()
        cfg.apply_overrides(data)
        return cfg, data

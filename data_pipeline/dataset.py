"""
Step 3 – Windowed PyTorch Dataset & DataLoader construction.

Each sample is a sliding window of length *w* with associated
prediction targets (future *H* steps) and a binary warning label.

Key design choices vs the previous version:
  - Windows are kept for ALL non-danger timesteps (warn=0 AND warn=1)
    so the model sees both "safe" and "warning-zone" samples.
  - Samples where danger has *already* arrived (hazard==1) are still
    excluded because there is nothing left to warn about.
  - `file` and `event_time` are preserved through the pipeline so that
    evaluate.py can compute per-experiment lead-time.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from config import Config
from data_pipeline.preprocess import (
    TriModalScaler,
    load_processed,
    preprocess_all,
    preprocess_post_fire,
    cache_is_complete,
)
from data_pipeline.labels import compute_hazard, label_one_experiment
from data_pipeline.contrastive_dataset import FireAugmentation


# ─────────────────────── Dataset ───────────────────────

class FirePredictionDataset(Dataset):
    """
    Each item yields:
        temp_field   (w, 160)       float32
        smoke        (w, D_smoke)   float32
        layer_domain (w, 4, 4)      float32
        mask_temp    (w, 160)       float32
        mask_smoke   (w, D_smoke)   float32
        future       (H, D_target)  float32   prediction targets
        warn         ()             float32   binary warning label
    """

    def __init__(
        self,
        experiments: List[Dict],
        cfg: Config,
        augment: bool = False,
    ):
        super().__init__()
        self.cfg = cfg
        self.w = cfg.history_window
        self.H = cfg.prediction_horizon
        self.augment = augment
        self._aug = FireAugmentation(cfg) if augment else None

        self.exps = experiments
        self.indices: List[Tuple[int, int]] = []
        self._build_index()

    def _build_index(self):
        use_post_hz = bool(
            getattr(self.cfg, "use_post_hazard_as_negative", False)
        )
        for ei, exp in enumerate(self.exps):
            T = len(exp["time"])
            hazard = exp.get("hazard")
            pred_only = exp.get("prediction_only", False)
            event_time = exp.get("event_time", np.nan)
            for t in range(self.w - 1, T - self.H):
                if pred_only:
                    self.indices.append((ei, t))
                    continue
                if not np.isnan(event_time) and float(exp["time"][t]) >= float(event_time):
                    if use_post_hz:
                        self.indices.append((ei, t))
                    continue
                if hazard is not None and hazard[t] == 1:
                    if not use_post_hz:
                        continue
                self.indices.append((ei, t))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        ei, t = self.indices[idx]
        exp = self.exps[ei]
        w, H = self.w, self.H
        s, e = t - w + 1, t + 1

        temp_field = exp["temp_field"][s:e]
        smoke      = exp["smoke"][s:e]
        ld         = exp["layer_domain"][s:e]
        mask_t     = exp["mask_temp"][s:e]
        mask_s     = exp["mask_smoke"][s:e]

        if self._aug is not None:
            temp_field, smoke, ld, mask_t, mask_s = self._aug(
                temp_field.copy(), smoke.copy(), ld.copy(), mask_t.copy(), mask_s.copy(),
            )

        future_smoke = exp["smoke"][e:e + H]
        future_lmean = exp["layer_domain"][e:e + H, :, 0]

        parts = []
        if self.cfg.predict_smoke:
            parts.append(future_smoke)
        if self.cfg.predict_layer_means:
            parts.append(future_lmean)
        future = np.concatenate(parts, axis=1).astype(np.float32)

        pred_only = exp.get("prediction_only", False)
        if pred_only:
            warn = np.float32(0)
            warn_valid = np.float32(0)
        else:
            event_time = exp.get("event_time", np.nan)
            cur_t = float(exp["time"][t])
            if not np.isnan(event_time) and cur_t >= float(event_time):
                warn = np.float32(0)
            else:
                warn = np.float32(exp["warn"][t]) if "warn" in exp else np.float32(0)
            warn_valid = np.float32(1)

        return dict(
            temp_field=torch.from_numpy(temp_field),
            smoke=torch.from_numpy(smoke),
            layer_domain=torch.from_numpy(ld),
            mask_temp=torch.from_numpy(mask_t),
            mask_smoke=torch.from_numpy(mask_s),
            future=torch.from_numpy(future),
            warn=torch.tensor(warn),
            warn_valid=torch.tensor(warn_valid),
        )

    def _effective_warn(self, ei: int, t: int) -> float:
        """Return the warning label used at training time (post-hazard → 0)."""
        exp = self.exps[ei]
        if exp.get("prediction_only", False):
            return 0.5
        event_time = exp.get("event_time", np.nan)
        cur_t = float(exp["time"][t])
        if not np.isnan(event_time) and cur_t >= float(event_time):
            return 0.0
        return float(exp["warn"][t]) if "warn" in exp else 0.0

    def label_counts(self):
        pos, neg, pred_only = 0, 0, 0
        for ei, t in self.indices:
            w = self._effective_warn(ei, t)
            if w == 0.5:
                pred_only += 1
            elif w == 1:
                pos += 1
            else:
                neg += 1
        return dict(total=len(self.indices), positive=pos, negative=neg,
                    prediction_only=pred_only)

    def get_sample_weights(self) -> np.ndarray:
        """Return per-sample weights for WeightedRandomSampler.

        When ``experiment_balanced_sampling`` is enabled (default), weighting
        has two layers:
          1. **Experiment-level**: every experiment gets equal total weight,
             preventing large experiments (e.g. exp 10 with 1609 windows)
             from dominating training.
          2. **Class-level** (within each experiment): pos and neg windows are
             balanced so both classes are seen equally often.

        When disabled, falls back to the original global pos/neg balancing.
        """
        use_exp_balance = bool(
            getattr(self.cfg, "experiment_balanced_sampling", False)
        )
        if use_exp_balance:
            return self._get_experiment_balanced_weights()
        return self._get_global_balanced_weights()

    def _get_global_balanced_weights(self) -> np.ndarray:
        """Original global pos/neg balancing (legacy)."""
        labels = []
        for ei, t in self.indices:
            labels.append(self._effective_warn(ei, t))
        labels = np.array(labels, dtype=np.float32)

        warn_mask = np.array([
            not self.exps[ei].get("prediction_only", False)
            for ei, _ in self.indices
        ], dtype=bool)

        warn_labels = labels[warn_mask]
        n_pos = warn_labels.sum()
        n_neg = len(warn_labels) - n_pos

        weights = np.ones(len(labels), dtype=np.float32)
        if n_pos > 0 and n_neg > 0:
            w_pos = len(warn_labels) / (2.0 * n_pos)
            w_neg = len(warn_labels) / (2.0 * n_neg)
            max_ratio = float(
                getattr(self.cfg, "sampler_max_class_weight_ratio", 4.0)
            )
            if max_ratio > 0 and w_neg > 0 and (w_pos / w_neg) > max_ratio:
                w_pos = w_neg * max_ratio
            for i, (ei, t) in enumerate(self.indices):
                if self.exps[ei].get("prediction_only", False):
                    weights[i] = 0.5
                elif labels[i] == 1:
                    weights[i] = w_pos
                else:
                    weights[i] = w_neg
        return weights

    def _get_experiment_balanced_weights(self) -> np.ndarray:
        """Per-experiment balanced sampling with group-level budget control.

        Experiments are split into two groups:
          - **full-process** (have warn=1 positive windows): get 60% of
            the total sampling budget, shared equally among experiments.
            Within each, pos and neg are further balanced 50/50.
          - **neg-only** (post-fire or all-negative): get 40% of budget,
            shared equally.  These provide diverse negative patterns so
            the model learns "active fire ≠ warning".

        This prevents the large number of post-fire experiments (69) from
        drowning out the few full-process experiments (5-10) that carry
        the actual positive-class signal.
        """
        from collections import defaultdict

        n_samples = len(self.indices)
        weights = np.zeros(n_samples, dtype=np.float32)

        exp_pos: dict[int, list[int]] = defaultdict(list)
        exp_neg: dict[int, list[int]] = defaultdict(list)
        exp_pred: dict[int, list[int]] = defaultdict(list)

        for i, (ei, t) in enumerate(self.indices):
            w = self._effective_warn(ei, t)
            if w == 0.5:
                exp_pred[ei].append(i)
            elif w == 1.0:
                exp_pos[ei].append(i)
            else:
                exp_neg[ei].append(i)

        all_exp_ids = set(exp_pos) | set(exp_neg) | set(exp_pred)
        if not all_exp_ids:
            return np.ones(n_samples, dtype=np.float32)

        full_exps = [ei for ei in all_exp_ids if len(exp_pos.get(ei, [])) > 0]
        neg_only_exps = [ei for ei in all_exp_ids if ei not in full_exps]

        FULL_BUDGET = 0.60
        NEG_BUDGET = 0.40

        if not full_exps:
            FULL_BUDGET, NEG_BUDGET = 0.0, 1.0
        if not neg_only_exps:
            FULL_BUDGET, NEG_BUDGET = 1.0, 0.0

        max_ratio = float(
            getattr(self.cfg, "sampler_max_class_weight_ratio", 4.0)
        )

        if full_exps:
            budget_per_full = FULL_BUDGET / len(full_exps)
            for ei in full_exps:
                pos_idx = exp_pos.get(ei, [])
                neg_idx = exp_neg.get(ei, [])
                pred_idx = exp_pred.get(ei, [])
                n_p, n_n, n_pr = len(pos_idx), len(neg_idx), len(pred_idx)

                if n_p > 0 and n_n > 0:
                    w_p = (budget_per_full * 0.5) / n_p
                    w_n = (budget_per_full * 0.5) / n_n
                    if max_ratio > 0 and w_n > 0 and (w_p / w_n) > max_ratio:
                        w_p = w_n * max_ratio
                    for idx in pos_idx:
                        weights[idx] = w_p
                    for idx in neg_idx:
                        weights[idx] = w_n
                elif n_p > 0:
                    for idx in pos_idx:
                        weights[idx] = budget_per_full / n_p
                elif n_n > 0:
                    for idx in neg_idx:
                        weights[idx] = budget_per_full / n_n

                if n_pr > 0:
                    w_pr = (budget_per_full * 0.2) / n_pr
                    for idx in pred_idx:
                        weights[idx] = w_pr

        if neg_only_exps:
            budget_per_neg = NEG_BUDGET / len(neg_only_exps)
            for ei in neg_only_exps:
                neg_idx = exp_neg.get(ei, [])
                pred_idx = exp_pred.get(ei, [])
                n_n, n_pr = len(neg_idx), len(pred_idx)

                if n_n > 0:
                    for idx in neg_idx:
                        weights[idx] = budget_per_neg / n_n
                if n_pr > 0:
                    w_pr = (budget_per_neg * 0.3) / n_pr
                    for idx in pred_idx:
                        weights[idx] = w_pr

        min_w = weights[weights > 0].min() if (weights > 0).any() else 1.0
        weights[weights == 0] = min_w * 0.1

        return weights


# ─────────────────────── collate ───────────────────────

def _collate(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch]) for k in keys}


# ─────────────────────── builder ───────────────────────

def build_dataloaders(
    cfg: Config,
    force_preprocess: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader, TriModalScaler]:
    """
    End-to-end: preprocess → label → normalise → split → DataLoader.

    Returns (train_loader, train_eval_loader, val_loader, test_loader, scaler).

    train_eval_loader: same windows as train, **no augmentation**, sequential
    (no WeightedRandomSampler). Use this for honest train-set metrics; train_loader
    is for optimisation only (sampler + optional augment).
    """
    cfg.make_dirs()

    # ── 1. preprocess (or load cached) ──
    proc_dir = Path(cfg.processed_dir)
    missing_npz = [
        eid for eid in cfg.full_process_ids
        if not (proc_dir / f"{eid}_processed.npz").exists()
    ]
    if missing_npz and not force_preprocess:
        print(f"[dataset] NOTE: no npz on disk yet for ids={missing_npz} (will preprocess if cache incomplete).")

    if force_preprocess or not cache_is_complete(cfg):
        print("[dataset] Running preprocessing …")
        all_exps = preprocess_all(cfg)
    else:
        print("[dataset] Loading cached preprocessed data …")
        all_exps = load_processed(cfg)

    if len(all_exps) == 0:
        raise RuntimeError(
            f"No experiments found. Check raw_csv_dir={cfg.raw_csv_dir} "
            f"and full_process_ids={cfg.full_process_ids}"
        )

    # ── 2. add labels ──
    for eid, exp in all_exps.items():
        exp["smoke_col_names"] = cfg.smoke_col_names
        label_one_experiment(exp, cfg)
        print(f"  {eid}: T={len(exp['time'])}  event_time={exp['event_time']:.0f}s  "
              f"warn_pos={exp['warn'].sum()}")

    # ── 3. split by experiment id ──
    def _select(ids):
        return {k: v for k, v in all_exps.items() if k in ids}

    train_exps = _select(cfg.train_ids)
    val_exps   = _select(cfg.val_ids)
    test_exps  = _select(cfg.test_ids)

    print(f"[dataset] train={list(train_exps.keys())}  "
          f"val={list(val_exps.keys())}  test={list(test_exps.keys())}")

    # ── 4. fit scaler on train ──
    scaler = TriModalScaler(cfg)
    scaler.fit(train_exps)

    # ── 5. transform (preserve metadata fields) ──
    def _apply(exps):
        out = []
        for e in exps.values():
            t = scaler.transform_experiment(e)
            for key in ("file", "event_time", "hazard_onset_time", "hazard", "warn", "time",
                        "mask_temp", "mask_smoke", "smoke_col_names",
                        "prediction_only"):
                if key in e and key not in t:
                    t[key] = e[key]
            out.append(t)
        return out

    train_list = _apply(train_exps)
    val_list   = _apply(val_exps)
    test_list  = _apply(test_exps)

    # ── 5b. add post-fire experiments to training ──
    use_post_as_neg = bool(
        getattr(cfg, "post_fire_as_supervised_neg", False)
    )
    include_post_fire = bool(
        getattr(cfg, "include_post_fire_regression", True)
    )
    if include_post_fire and cfg.post_fire_ids:
        mode_str = "supervised negatives" if use_post_as_neg else "prediction-only"
        print(f"[dataset] Loading {len(cfg.post_fire_ids)} post-fire experiments "
              f"as {mode_str} …")
        post_exps = preprocess_post_fire(cfg)
        for eid, exp in post_exps.items():
            exp["smoke_col_names"] = cfg.smoke_col_names
            exp["warn"] = np.zeros(len(exp["time"]), dtype=np.int32)
            exp["event_time"] = np.nan
            exp["hazard"] = compute_hazard(
                exp["time"], exp["smoke"], cfg.smoke_col_names, cfg,
                exp.get("layer_domain"), temp_field=exp.get("temp_field"),
            )
            if use_post_as_neg:
                exp["prediction_only"] = False
            else:
                exp["prediction_only"] = True
        post_list = _apply(post_exps)
        for pe in post_list:
            if not use_post_as_neg:
                pe["prediction_only"] = True
        train_list.extend(post_list)

    # ── 6. create datasets ──
    use_aug = bool(getattr(cfg, "supervised_train_augment", False))
    train_ds = FirePredictionDataset(train_list, cfg, augment=use_aug)
    val_ds   = FirePredictionDataset(val_list, cfg, augment=False)
    test_ds  = FirePredictionDataset(test_list, cfg, augment=False)

    for name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        lc = ds.label_counts()
        pred_str = f", pred_only={lc['prediction_only']}" if lc['prediction_only'] else ""
        n_exps = len(set(ei for ei, _ in ds.indices))
        print(f"  {name}: {lc['total']} windows from {n_exps} experiments  "
              f"(pos={lc['positive']}, neg={lc['negative']}{pred_str})")
        if name == "test" and lc["total"] < 50:
            print(
                f"  ⚠ test windows < 50 — classification metrics / AUC may be unstable; "
                f"consider LOOCV, different split, or smaller history_window."
            )

    # ── 7. DataLoaders (balanced sampling for train) ──
    train_weights = train_ds.get_sample_weights()
    factor = max(0.1, float(getattr(cfg, "train_oversample_factor", 1.0)))
    n_draw = max(1, int(len(train_ds) * factor))
    if abs(factor - 1.0) > 1e-6:
        print(f"[dataset] train oversample: factor={factor:g}  "
              f"num_samples={n_draw} (dataset size={len(train_ds)})")
    train_sampler = WeightedRandomSampler(
        weights=torch.from_numpy(train_weights),
        num_samples=n_draw,
        replacement=True,
    )
    nw = getattr(cfg, "num_workers", 0)
    pm = getattr(cfg, "pin_memory", False) and nw > 0
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, sampler=train_sampler,
        collate_fn=_collate, drop_last=False,
        num_workers=nw, pin_memory=pm, persistent_workers=nw > 0,
    )
    train_eval_ds = FirePredictionDataset(train_list, cfg, augment=False)
    train_eval_loader = DataLoader(
        train_eval_ds, batch_size=cfg.batch_size, shuffle=False,
        collate_fn=_collate, drop_last=False,
        num_workers=nw, pin_memory=pm, persistent_workers=nw > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        collate_fn=_collate, drop_last=False,
        num_workers=nw, pin_memory=pm, persistent_workers=nw > 0,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        collate_fn=_collate, drop_last=False,
        num_workers=nw, pin_memory=pm, persistent_workers=nw > 0,
    )

    # ── 8. save scaler ──
    scaler_path = Path(cfg.output_dir) / "scaler.json"
    with open(scaler_path, "w", encoding="utf-8") as f:
        json.dump(scaler.state_dict(), f, ensure_ascii=False)

    return train_loader, train_eval_loader, val_loader, test_loader, scaler

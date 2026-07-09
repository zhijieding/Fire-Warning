"""
Contrastive pre-training dataset.

By default loads only *post_fire_ids* (disjoint from supervised full_process
experiments) so pre-training does not leak hold-out / LOOCV test data.
Optional mode ``all_csv`` restores legacy behaviour (leaky; not for clean eval).

Augmentations (fire-domain motivated):
  1. Gaussian jitter      – sensor noise
  2. Channel masking      – simulated sensor failure in temp field
  3. Smoke channel dropout – force cross-channel learning
  4. Temporal segment mask – data transmission gaps
"""
from __future__ import annotations

import random
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from config import Config
from data_pipeline.preprocess import (
    preprocess_one_experiment,
    resolve_experiment_csv_path,
    TriModalScaler,
)


# ─────────────────────── augmentations ───────────────────────

class FireAugmentation:
    """Random augmentation pipeline for one view."""

    def __init__(self, cfg: Config):
        self.jitter_std = cfg.aug_jitter_std
        self.ch_mask_ratio = cfg.aug_channel_mask_ratio
        self.temp_mask_ratio = cfg.aug_temporal_mask_ratio
        self.smoke_drop_prob = cfg.aug_smoke_dropout_prob

    def __call__(
        self,
        temp: np.ndarray,     # (w, 160)
        smoke: np.ndarray,    # (w, D)
        ld: np.ndarray,       # (w, 4, 4)
        mt: np.ndarray,       # (w, 160)
        ms: np.ndarray,       # (w, D)
    ) -> Tuple[np.ndarray, ...]:
        temp, smoke, ld = temp.copy(), smoke.copy(), ld.copy()
        mt, ms = mt.copy(), ms.copy()
        w = temp.shape[0]

        # 1. Gaussian jitter (p=0.8)
        if random.random() < 0.8:
            temp += np.random.randn(*temp.shape).astype(np.float32) * self.jitter_std
            smoke += np.random.randn(*smoke.shape).astype(np.float32) * self.jitter_std
            ld += np.random.randn(*ld.shape).astype(np.float32) * self.jitter_std * 0.5

        # 2. Random channel masking on temp field (p=0.5)
        if random.random() < 0.5:
            n_mask = max(1, int(temp.shape[1] * self.ch_mask_ratio))
            idx = np.random.choice(temp.shape[1], n_mask, replace=False)
            temp[:, idx] = 0.0
            mt[:, idx] = 0.0

        # 3. Smoke channel dropout – zero one entire channel (p=smoke_drop_prob)
        if random.random() < self.smoke_drop_prob and smoke.shape[1] > 1:
            ch = np.random.randint(smoke.shape[1])
            smoke[:, ch] = 0.0
            ms[:, ch] = 0.0

        # 4. Temporal segment masking (p=0.5)
        if random.random() < 0.5:
            n_mask = max(1, int(w * self.temp_mask_ratio))
            n_segs = random.randint(1, 3)
            for _ in range(n_segs):
                seg_len = max(1, n_mask // n_segs)
                start = random.randint(0, w - seg_len)
                temp[start:start + seg_len] = 0.0
                smoke[start:start + seg_len] = 0.0
                ld[start:start + seg_len] = 0.0
                mt[start:start + seg_len] = 0.0
                ms[start:start + seg_len] = 0.0

        return temp, smoke, ld, mt, ms


# ─────────────────────── dataset ───────────────────────

class ContrastiveFireDataset(Dataset):
    """
    Each item returns two augmented views of the same sliding window.
    No labels are used — only raw sensor time-series.
    """

    def __init__(self, experiments: List[Dict], cfg: Config):
        super().__init__()
        self.exps = experiments
        self.w = cfg.history_window
        self.aug = FireAugmentation(cfg)

        self.indices: List[Tuple[int, int]] = []
        for ei, exp in enumerate(self.exps):
            T = len(exp["time"])
            for t in range(self.w - 1, T):
                self.indices.append((ei, t))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        ei, t = self.indices[idx]
        exp = self.exps[ei]
        s, e = t - self.w + 1, t + 1

        temp = exp["temp_field"][s:e]
        smoke = exp["smoke"][s:e]
        ld = exp["layer_domain"][s:e]
        mt = exp["mask_temp"][s:e]
        ms = exp["mask_smoke"][s:e]

        t1, s1, ld1, mt1, ms1 = self.aug(temp, smoke, ld, mt, ms)
        t2, s2, ld2, mt2, ms2 = self.aug(temp, smoke, ld, mt, ms)

        def _to_tensor(x):
            return torch.from_numpy(np.ascontiguousarray(x))

        return {
            "temp_field_1": _to_tensor(t1), "smoke_1": _to_tensor(s1),
            "layer_domain_1": _to_tensor(ld1),
            "mask_temp_1": _to_tensor(mt1), "mask_smoke_1": _to_tensor(ms1),
            "temp_field_2": _to_tensor(t2), "smoke_2": _to_tensor(s2),
            "layer_domain_2": _to_tensor(ld2),
            "mask_temp_2": _to_tensor(mt2), "mask_smoke_2": _to_tensor(ms2),
        }


# ─────────────────────── loader builder ───────────────────────

def _cl_collate(batch):
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch]) for k in keys}


def _collect_csv_paths_all_csv(cfg: Config) -> List[Path]:
    """Glob raw_csv_dir plus extra_csv_map paths, deduped by resolved path."""
    input_dir = Path(cfg.raw_csv_dir)
    by_resolved: dict[str, Path] = {}
    for fp in sorted(input_dir.glob("*.csv")):
        by_resolved[str(fp.resolve())] = fp
    for _eid, path_str in getattr(cfg, "extra_csv_map", {}).items():
        p = Path(path_str)
        if p.exists():
            by_resolved[str(p.resolve())] = p
    return list(by_resolved.values())


def collect_contrastive_csv_paths(cfg: Config) -> List[Path]:
    """
    CSV paths for contrastive pre-training.

    *post_fire_only* (default): only experiments in ``cfg.post_fire_ids`` —
    disjoint from ``full_process_ids`` after discovery → no leakage for
    supervised test / LOOCV.

    *all_csv*: legacy glob of all CSVs (includes supervised experiments → leaky).
    """
    mode = getattr(cfg, "contrastive_data_mode", "post_fire_only")
    if mode == "all_csv":
        warnings.warn(
            "contrastive_data_mode='all_csv' includes supervised experiments — "
            "information leakage for hold-out test or LOOCV. "
            "Use 'post_fire_only' for unbiased evaluation.",
            UserWarning,
            stacklevel=2,
        )
        return _collect_csv_paths_all_csv(cfg)

    if mode != "post_fire_only":
        raise ValueError(
            f"Unknown contrastive_data_mode={mode!r}; "
            "use 'post_fire_only' or 'all_csv'"
        )

    ids = list(getattr(cfg, "post_fire_ids", None) or [])
    by_resolved: dict[str, Path] = {}
    for exp_id in ids:
        p = resolve_experiment_csv_path(cfg, exp_id)
        if p is not None:
            by_resolved[str(p.resolve())] = p
    return list(by_resolved.values())


def build_contrastive_loader(cfg: Config) -> Tuple[DataLoader, TriModalScaler]:
    """
    Preprocess contrastive experiments, fit scaler on those experiments only,
    and return a DataLoader. Default data = post_fire_ids (no supervised leakage).
    """
    cfg.make_dirs()
    csv_files = collect_contrastive_csv_paths(cfg)
    if not csv_files:
        mode = getattr(cfg, "contrastive_data_mode", "post_fire_only")
        if mode == "post_fire_only":
            raise RuntimeError(
                "No CSV files for contrastive pre-training in post_fire_only mode. "
                "Need at least one experiment in post_fire_ids (run discovery: "
                "experiments with event_time too early for full-process warning). "
                "Or set config.contrastive_data_mode='all_csv' only for debugging "
                "(leaky for evaluation)."
            )
        raise RuntimeError(
            f"No CSV files found in {cfg.raw_csv_dir} or extra_csv_map paths"
        )

    mode = getattr(cfg, "contrastive_data_mode", "post_fire_only")
    print(
        f"[contrastive] mode={mode!r}  Preprocessing {len(csv_files)} experiments …"
    )
    all_exps: List[Dict] = []
    for fp in csv_files:
        try:
            exp = preprocess_one_experiment(fp, cfg)
            all_exps.append(exp)
        except Exception as e:
            print(f"  [skip] {fp.name}: {e}")

    print(f"[contrastive] Loaded {len(all_exps)} experiments, "
          f"total rows = {sum(len(e['time']) for e in all_exps):,}")

    # fit scaler on contrastive pool only (same experiments as windows)
    scaler = TriModalScaler(cfg)
    exp_dict = {str(i): e for i, e in enumerate(all_exps)}
    scaler.fit(exp_dict)

    # transform
    scaled_exps = []
    for exp in all_exps:
        t = scaler.transform_experiment(exp)
        for key in ("time", "mask_temp", "mask_smoke"):
            if key in exp and key not in t:
                t[key] = exp[key]
        scaled_exps.append(t)

    ds = ContrastiveFireDataset(scaled_exps, cfg)
    print(f"[contrastive] Dataset: {len(ds)} windows")

    loader = DataLoader(
        ds,
        batch_size=cfg.pretrain_batch_size,
        shuffle=True,
        collate_fn=_cl_collate,
        drop_last=True,
        num_workers=0,
    )
    return loader, scaler

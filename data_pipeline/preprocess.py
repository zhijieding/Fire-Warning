"""
Step 1  – Data pre-processing
  1.1 Align to 1 Hz, build missing masks, interpolate
  1.2 Construct three modalities (temp-field / smoke-optical / layer-domain)
  1.3 Per-channel log-transform + z-score normalisation
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from config import Config


# ───────────────────── helpers ─────────────────────

def _is_temp_col_old(col: str) -> bool:
    """Old format: '1-1', '2-40', etc."""
    return re.match(r"^\d+-\d+$", str(col)) is not None


def _is_temp_col_new(col: str) -> bool:
    """New format: 'T_m1_01', 'T_m4_40', etc."""
    return re.match(r"^T_m\d+_\d+$", str(col)) is not None


def _temp_col_key_old(col: str):
    a, b = col.split("-")
    return int(a), int(b)


def _temp_col_key_new(col: str):
    m = re.match(r"^T_m(\d+)_(\d+)$", col)
    return int(m.group(1)), int(m.group(2))


def _detect_format(df: pd.DataFrame):
    """Auto-detect old ('1-1') vs new ('T_m1_01') column naming."""
    old_cols = [c for c in df.columns if _is_temp_col_old(c)]
    new_cols = [c for c in df.columns if _is_temp_col_new(c)]
    if len(new_cols) >= len(old_cols) and len(new_cols) > 0:
        return "new"
    return "old"


def _get_temp_cols(df: pd.DataFrame, fmt: str):
    if fmt == "new":
        cols = sorted(
            [c for c in df.columns if _is_temp_col_new(c)],
            key=_temp_col_key_new,
        )
    else:
        cols = sorted(
            [c for c in df.columns if _is_temp_col_old(c)],
            key=_temp_col_key_old,
        )
    return cols


def _get_layer_prefix(fmt: str, layer_num: int) -> str:
    if fmt == "new":
        return f"T_m{layer_num}_"
    return f"{layer_num}-"


def _fill_series(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    s = s.interpolate(method="linear", limit_direction="both")
    s = s.ffill().bfill()
    return s


# ───────────────────── single experiment ─────────────────────

def preprocess_one_experiment(file_path: Path, cfg: Config) -> Dict[str, np.ndarray]:
    """
    Read one cleaned CSV → return dict of numpy arrays for the 3 modalities
    plus missing masks.  Supports both old (时间 / 1-1) and new (datetime /
    T_m1_01) column formats automatically.
    """
    df = pd.read_csv(file_path, encoding="utf-8-sig")
    fmt = _detect_format(df)

    # ── time column ──
    if "时间" in df.columns:
        df["时间"] = pd.to_numeric(df["时间"], errors="coerce")
        df = df.sort_values("时间").reset_index(drop=True)
        time = df["时间"].values.astype(np.float32)
    else:
        first_col = df.columns[0]
        try:
            df[first_col] = pd.to_datetime(df[first_col])
            df = df.sort_values(first_col).reset_index(drop=True)
            t0 = df[first_col].iloc[0]
            time = ((df[first_col] - t0).dt.total_seconds()).values.astype(np.float32)
        except Exception:
            raise ValueError(
                f"{file_path.name}: no '时间' column and first column "
                f"'{first_col}' is not parseable as datetime"
            )

    # ── identify temperature columns ──
    temp_cols = _get_temp_cols(df, fmt)
    if len(temp_cols) != cfg.n_temp_channels:
        raise ValueError(
            f"{file_path.name}: expected {cfg.n_temp_channels} temp cols, "
            f"got {len(temp_cols)}"
        )

    # ── identify available smoke / heat columns (format-aware) ──
    raw_smoke_cols: List[str] = []
    if fmt == "new":
        for c in ["CO", "Trans", "k", "Heat"]:
            if c in df.columns:
                raw_smoke_cols.append(c)
    else:
        for c in ["CO", "透光", "1D", "0.5D", "1.5D"]:
            if c in df.columns:
                raw_smoke_cols.append(c)

    # ── handle pre-computed mask columns (new format) ──
    has_precomputed_masks = any(c.endswith("_mask") for c in df.columns)

    # ── to numeric ──
    for c in temp_cols + raw_smoke_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # ── 1.1  masks (before any filling) ──
    if has_precomputed_masks:
        mask_cols = [c + "_mask" for c in temp_cols
                     if c + "_mask" in df.columns]
        if len(mask_cols) == len(temp_cols):
            mask_temp = (1 - df[mask_cols].values).astype(np.float32)
        else:
            mask_temp = (~df[temp_cols].isna()).values.astype(np.float32)
    else:
        mask_temp = (~df[temp_cols].isna()).values.astype(np.float32)

    raw_masks: Dict[str, np.ndarray] = {}
    for c in raw_smoke_cols:
        mask_col = c + "_mask"
        if has_precomputed_masks and mask_col in df.columns:
            raw_masks[c] = (1 - pd.to_numeric(
                df[mask_col], errors="coerce"
            ).fillna(1)).values.astype(np.float32)
        else:
            raw_masks[c] = (~df[c].isna()).values.astype(np.float32)

    # ── 1.1  fill missing values ──
    temp_filled = df[temp_cols].copy()
    for c in temp_cols:
        temp_filled[c] = _fill_series(temp_filled[c])

    for layer_num in range(1, cfg.n_layers + 1):
        prefix = _get_layer_prefix(fmt, layer_num)
        lcols = [c for c in temp_cols if c.startswith(prefix)]
        layer_mean = temp_filled[lcols].mean(axis=1, skipna=True)
        for c in lcols:
            if temp_filled[c].isna().any():
                temp_filled[c] = temp_filled[c].fillna(layer_mean)
    temp_filled = temp_filled.fillna(0.0)

    smoke_filled: Dict[str, np.ndarray] = {}
    for c in raw_smoke_cols:
        smoke_filled[c] = _fill_series(df[c]).fillna(0.0).values.astype(np.float32)

    # ── 1.2a  modality-1: temperature field (T, 160) ──
    temp_field = temp_filled.values.astype(np.float32)

    # ── 1.2b  modality-2: smoke / optical / heat (T, D_smoke) ──
    T = len(df)
    if fmt == "new":
        trans_vals = smoke_filled.get("Trans", np.zeros(T, dtype=np.float32))
        heat_val = smoke_filled.get("Heat", np.zeros(T, dtype=np.float32))
        smoke_channels: Dict[str, np.ndarray] = {
            "CO": smoke_filled.get("CO", np.zeros(T, dtype=np.float32)),
            "Trans": trans_vals,
            "1D": heat_val,
            "0.5D": np.zeros(T, dtype=np.float32),
            "1.5D": np.zeros(T, dtype=np.float32),
        }
        mask_remap = {
            "CO": raw_masks.get("CO", np.ones(T, dtype=np.float32)),
            "Trans": raw_masks.get("Trans", raw_masks.get("k",
                               np.ones(T, dtype=np.float32))),
            "1D": raw_masks.get("Heat", np.ones(T, dtype=np.float32)),
            "0.5D": np.zeros(T, dtype=np.float32),
            "1.5D": np.zeros(T, dtype=np.float32),
        }
    else:
        trans_vals = (
            smoke_filled["透光"]
            if "透光" in smoke_filled
            else np.zeros(T, dtype=np.float32)
        )
        smoke_channels = {
            "CO": smoke_filled.get("CO", np.zeros(T, dtype=np.float32)),
            "Trans": trans_vals,
        }
        for hc in cfg.heat_col_candidates:
            if hc in smoke_filled:
                smoke_channels[hc] = smoke_filled[hc]
        mask_remap = None

    smoke_list, smoke_names = [], []
    for name in cfg.smoke_col_names:
        if name in smoke_channels:
            smoke_list.append(smoke_channels[name])
            smoke_names.append(name)
    smoke_array = np.stack(smoke_list, axis=1).astype(np.float32)  # (T, D_smoke)

    mask_smoke_list = []
    for name in smoke_names:
        if mask_remap is not None:
            mk = mask_remap.get(name, np.ones(T, dtype=np.float32))
        elif name == "Trans":
            mk = raw_masks.get("透光", raw_masks.get("Trans",
                               np.ones(T, dtype=np.float32)))
        elif name in raw_masks:
            mk = raw_masks[name]
        else:
            mk = np.ones(T, dtype=np.float32)
        mask_smoke_list.append(mk)
    mask_smoke = np.stack(mask_smoke_list, axis=1).astype(np.float32)

    # ── 1.2c  modality-3: layer-domain (T, 4, 4) ──
    layer_domain = np.zeros(
        (T, cfg.n_layers, cfg.n_layer_features), dtype=np.float32
    )
    for li, ln in enumerate(range(1, cfg.n_layers + 1)):
        prefix = _get_layer_prefix(fmt, ln)
        lcols = [c for c in temp_cols if c.startswith(prefix)]
        ldata = temp_filled[lcols].values.astype(np.float32)
        lmean = np.nanmean(ldata, axis=1)
        layer_domain[:, li, 0] = lmean
        layer_domain[:, li, 1] = np.nanmax(ldata, axis=1)
        layer_domain[:, li, 2] = np.nanstd(ldata, axis=1)
        slope = np.zeros_like(lmean)
        slope[5:] = (lmean[5:] - lmean[:-5]) / 5.0
        layer_domain[:, li, 3] = slope

    exp_id = file_path.stem.replace("_clean", "")

    return dict(
        file=exp_id,
        time=time,
        temp_field=temp_field,
        smoke=smoke_array,
        layer_domain=layer_domain,
        mask_temp=mask_temp,
        mask_smoke=mask_smoke,
        smoke_col_names=smoke_names,
    )


# ───────────────────── batch processing ─────────────────────

def preprocess_all(cfg: Config) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Preprocess every *full-process* experiment, save .npz, return dict keyed
    by experiment id.  Supports both old-format (*_clean.csv) and new-format
    CSVs via extra_csv_map in config.
    """
    out_dir = Path(cfg.processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    input_dir = Path(cfg.raw_csv_dir)

    results: Dict[str, Dict[str, np.ndarray]] = {}

    for exp_id in cfg.full_process_ids:
        # Check if there is a custom path for this experiment
        extra = getattr(cfg, "extra_csv_map", {})
        if exp_id in extra:
            fpath = Path(extra[exp_id])
        else:
            fname = f"{exp_id}_clean.csv"
            fpath = input_dir / fname

        if not fpath.exists():
            print(f"  [skip] {fpath.name} not found")
            continue

        print(f"  Preprocessing {fpath.name} …")
        data = preprocess_one_experiment(fpath, cfg)
        data["file"] = exp_id

        save_path = out_dir / f"{exp_id}_processed.npz"
        np.savez_compressed(
            save_path,
            time=data["time"],
            temp_field=data["temp_field"],
            smoke=data["smoke"],
            layer_domain=data["layer_domain"],
            mask_temp=data["mask_temp"],
            mask_smoke=data["mask_smoke"],
        )
        results[exp_id] = data
        print(f"    → T={len(data['time'])}  saved to {save_path}")

    missing = [eid for eid in cfg.full_process_ids if eid not in results]
    if missing:
        print(f"\n[preprocess] WARNING: no npz written for ids={missing} "
              f"(missing files or preprocess errors); cache will stay incomplete.")

    return results


def load_processed(cfg: Config) -> Dict[str, Dict[str, np.ndarray]]:
    """Load previously saved .npz experiments."""
    proc_dir = Path(cfg.processed_dir)
    results: Dict[str, Dict[str, np.ndarray]] = {}
    for exp_id in cfg.full_process_ids:
        p = proc_dir / f"{exp_id}_processed.npz"
        if not p.exists():
            continue
        d = dict(np.load(p, allow_pickle=True))
        d["file"] = exp_id
        d["smoke_col_names"] = cfg.smoke_col_names
        results[exp_id] = d
    return results


def cache_is_complete(cfg: Config) -> bool:
    """Check whether all full_process experiments have cached .npz files."""
    proc_dir = Path(cfg.processed_dir)
    for exp_id in cfg.full_process_ids:
        if not (proc_dir / f"{exp_id}_processed.npz").exists():
            return False
    return True


# ───────────────────── normalisation ─────────────────────

class ModalityScaler:
    """Per-channel z-score (optionally preceded by log1p)."""

    def __init__(self, log_cols_idx: List[int] | None = None):
        self.log_cols_idx = log_cols_idx or []
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    # ---------- fit on training data ----------
    def fit(self, arrays: List[np.ndarray]):
        """arrays: list of (T_i, D) – one per training experiment."""
        cat = np.concatenate(arrays, axis=0)
        if self.log_cols_idx:
            cat = cat.copy()
            cat[:, self.log_cols_idx] = np.log1p(
                np.clip(cat[:, self.log_cols_idx], 0, None)
            )
        self.mean_ = cat.mean(axis=0).astype(np.float32)
        self.std_ = cat.std(axis=0).astype(np.float32)
        self.std_[self.std_ < 1e-8] = 1.0

    # ---------- transform ----------
    def transform(self, x: np.ndarray) -> np.ndarray:
        out = x.copy()
        if self.log_cols_idx:
            out[:, self.log_cols_idx] = np.log1p(
                np.clip(out[:, self.log_cols_idx], 0, None)
            )
        return ((out - self.mean_) / self.std_).astype(np.float32)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        out = x * self.std_ + self.mean_
        if self.log_cols_idx:
            out[:, self.log_cols_idx] = np.expm1(out[:, self.log_cols_idx])
        return out

    def state_dict(self):
        return dict(
            log_cols_idx=self.log_cols_idx,
            mean=self.mean_.tolist() if self.mean_ is not None else None,
            std=self.std_.tolist() if self.std_ is not None else None,
        )

    def load_state_dict(self, d):
        self.log_cols_idx = d["log_cols_idx"]
        self.mean_ = np.array(d["mean"], dtype=np.float32) if d["mean"] else None
        self.std_ = np.array(d["std"], dtype=np.float32) if d["std"] else None



# ───────────────────── auto-discovery ─────────────────────

def _natural_sort_key(s: str):
    """Sort strings with embedded numbers naturally: '2' < '10' < '23-1'."""
    import re as _re
    return [int(c) if c.isdigit() else c.lower()
            for c in _re.split(r"(\d+)", s)]


def discover_full_process_experiments(
    cfg: Config,
) -> Tuple[List[str], List[str]]:
    """
    Scan ALL CSVs in ``raw_csv_dir`` (plus ``extra_csv_map`` entries) and
    classify each into *full-process* (enough pre-fire data for warning
    labels) vs *post-fire-start* (only usable for contrastive pre-training).

    Returns ``(full_process_ids, post_fire_ids)``.
    """
    from data_pipeline.labels import find_event_time

    input_dir = Path(cfg.raw_csv_dir)

    exp_paths: Dict[str, Path] = {}
    for fp in sorted(input_dir.glob("*.csv")):
        exp_id = fp.stem.replace("_clean", "")
        exp_paths[exp_id] = fp

    for exp_id, path_str in cfg.extra_csv_map.items():
        p = Path(path_str)
        if p.exists():
            exp_paths[exp_id] = p

    override = getattr(cfg, "discover_min_pre_fire", None)
    if override is not None:
        min_pre_fire = int(override)
    else:
        min_pre_fire = cfg.history_window - 2

    full_process: List[str] = []
    post_fire: List[str] = []

    print(f"[discover] Scanning {len(exp_paths)} candidate experiments …")

    for exp_id in sorted(exp_paths, key=_natural_sort_key):
        fp = exp_paths[exp_id]
        try:
            exp = preprocess_one_experiment(fp, cfg)
            event_time = find_event_time(
                exp["time"], exp["smoke"], cfg.smoke_col_names, cfg,
                exp.get("layer_domain"), temp_field=exp.get("temp_field"),
            )
            max_time = float(exp["time"][-1])

            if not np.isnan(event_time) and event_time >= min_pre_fire:
                full_process.append(exp_id)
                print(f"  [full] {exp_id:>8s}  event={event_time:6.0f}s  "
                      f"duration={max_time:.0f}s")
            else:
                post_fire.append(exp_id)
                reason = ("no event" if np.isnan(event_time)
                          else f"event too early ({event_time:.0f}s)")
                print(f"  [skip] {exp_id:>8s}  {reason}")
        except Exception as e:
            print(f"  [err]  {exp_id:>8s}  {e}")

    print(f"\n[discover] Result: {len(full_process)} full-process, "
          f"{len(post_fire)} post-fire / unusable")
    return full_process, post_fire


def resolve_experiment_csv_path(cfg: Config, exp_id: str) -> Path | None:
    """Absolute path to CSV for this experiment id, or None if missing."""
    extra = getattr(cfg, "extra_csv_map", {})
    if exp_id in extra:
        p = Path(extra[exp_id])
    else:
        p = Path(cfg.raw_csv_dir) / f"{exp_id}_clean.csv"
    return p.resolve() if p.exists() else None


def dedupe_full_process_ids_by_csv_path(cfg: Config, full_ids: List[str]) -> List[str]:
    """Remove duplicate IDs that resolve to the same CSV (e.g. 10 vs fire_merged_1hz_reference)."""
    seen: set[str] = set()
    out: List[str] = []
    for eid in full_ids:
        path = resolve_experiment_csv_path(cfg, eid)
        if path is None:
            out.append(eid)
            continue
        key = str(path)
        if key in seen:
            print(f"[dedupe] skipping id={eid!r} (same CSV as an earlier experiment)")
            continue
        seen.add(key)
        out.append(eid)
    return out


def discover_and_configure(cfg: Config) -> None:
    """Run auto-discovery and populate experiment splits if needed."""
    if not cfg.auto_discover or cfg.full_process_ids:
        return

    full_ids, post_ids = discover_full_process_experiments(cfg)
    full_ids = dedupe_full_process_ids_by_csv_path(cfg, full_ids)
    cfg.full_process_ids = full_ids
    cfg.post_fire_ids = post_ids

    if not cfg.train_ids:
        event_times = _get_event_times(cfg, full_ids)
        cfg.setup_splits(full_ids, event_times=event_times)

    print(f"  Train ({len(cfg.train_ids)}): {cfg.train_ids}")
    print(f"  Val   ({len(cfg.val_ids)}):   {cfg.val_ids}")
    print(f"  Test  ({len(cfg.test_ids)}):  {cfg.test_ids}")


def _get_event_times(cfg: Config, exp_ids: list) -> dict:
    """Retrieve event_time for a list of experiment IDs."""
    from data_pipeline.labels import find_event_time

    input_dir = Path(cfg.raw_csv_dir)
    result = {}
    for exp_id in exp_ids:
        extra = getattr(cfg, "extra_csv_map", {})
        fpath = Path(extra[exp_id]) if exp_id in extra else input_dir / f"{exp_id}_clean.csv"
        if not fpath.exists():
            continue
        try:
            exp = preprocess_one_experiment(fpath, cfg)
            et = find_event_time(
                exp["time"], exp["smoke"], cfg.smoke_col_names, cfg,
                exp.get("layer_domain"), temp_field=exp.get("temp_field"),
            )
            result[exp_id] = et
        except Exception:
            pass
    return result


def preprocess_post_fire(cfg: Config) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Preprocess post-fire-start experiments for prediction-only training.
    These experiments lack a pre-fire phase but still contain valuable
    fire dynamics data for the prediction head.
    """
    out_dir = Path(cfg.processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    input_dir = Path(cfg.raw_csv_dir)

    results: Dict[str, Dict[str, np.ndarray]] = {}

    for exp_id in cfg.post_fire_ids:
        cache_path = out_dir / f"{exp_id}_processed.npz"

        if cache_path.exists():
            d = dict(np.load(cache_path, allow_pickle=True))
            d["file"] = exp_id
            d["smoke_col_names"] = cfg.smoke_col_names
            results[exp_id] = d
            continue

        extra = getattr(cfg, "extra_csv_map", {})
        if exp_id in extra:
            fpath = Path(extra[exp_id])
        else:
            fpath = input_dir / f"{exp_id}_clean.csv"

        if not fpath.exists():
            continue

        try:
            data = preprocess_one_experiment(fpath, cfg)
            data["file"] = exp_id
            np.savez_compressed(
                cache_path,
                time=data["time"],
                temp_field=data["temp_field"],
                smoke=data["smoke"],
                layer_domain=data["layer_domain"],
                mask_temp=data["mask_temp"],
                mask_smoke=data["mask_smoke"],
            )
            results[exp_id] = data
        except Exception as e:
            print(f"  [skip post-fire] {exp_id}: {e}")

    print(f"[post-fire] Loaded {len(results)} experiments for prediction-only training")
    return results


class TriModalScaler:
    """Wraps three ModalityScalers for temp / smoke / layer-domain."""

    def __init__(self, cfg: Config):
        smoke_log_idx = [
            i for i, name in enumerate(cfg.smoke_col_names)
            if name in cfg.log_transform_cols
        ]
        self.temp = ModalityScaler()
        self.smoke = ModalityScaler(log_cols_idx=smoke_log_idx)
        self.layer = ModalityScaler()

    def fit(self, experiments: Dict[str, Dict]):
        self.temp.fit([e["temp_field"] for e in experiments.values()])
        self.smoke.fit([e["smoke"] for e in experiments.values()])
        layers = [
            e["layer_domain"].reshape(len(e["time"]), -1) for e in experiments.values()
        ]
        self.layer.fit(layers)

    def transform_experiment(self, exp: Dict) -> Dict:
        exp = dict(exp)
        exp["temp_field"] = self.temp.transform(exp["temp_field"])
        exp["smoke"] = self.smoke.transform(exp["smoke"])
        T, nL, nF = exp["layer_domain"].shape
        flat = exp["layer_domain"].reshape(T, nL * nF)
        exp["layer_domain"] = self.layer.transform(flat).reshape(T, nL, nF)
        return exp

    def state_dict(self):
        return dict(
            temp=self.temp.state_dict(),
            smoke=self.smoke.state_dict(),
            layer=self.layer.state_dict(),
        )

    def load_state_dict(self, d):
        self.temp.load_state_dict(d["temp"])
        self.smoke.load_state_dict(d["smoke"])
        self.layer.load_state_dict(d["layer"])

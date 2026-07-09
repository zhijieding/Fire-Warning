"""
Training loop for TriModalFireModel.

Supports two modes:
  A. Train from scratch   →  python train.py
  B. Fine-tune after contrastive pre-training  →  python train.py --pretrained

In mode B, pre-trained backbone (encoders + fusion) gets a lower learning
rate (finetune_lr_backbone) while new heads get a higher rate (finetune_lr_head).
"""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from checkpoint_io import load_torch_checkpoint
from config import Config
from data_pipeline.dataset import build_dataloaders
from models import build_model, TriModalFireModel
from models.losses import CombinedLoss


# ───────────────── lr schedule ─────────────────

def get_cosine_schedule_with_warmup(
    optimizer, warmup_epochs: int, total_epochs: int, min_lr_ratio: float = 0.01,
) -> LambdaLR:
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return max(epoch / max(warmup_epochs, 1), 0.01)
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return min_lr_ratio + 0.5 * (1 - min_lr_ratio) * (1 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


# ───────────────── reproducibility ─────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ───────────────── one epoch ─────────────────

def run_epoch(
    model: TriModalFireModel,
    loader: DataLoader,
    criterion: CombinedLoss,
    optimizer=None,
    device: str = "cpu",
    debug_nan_checks: bool = False,
    scaler: GradScaler | None = None,
    use_amp: bool = False,
) -> dict:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    loss_parts = {"pred": 0.0, "warn": 0.0, "trend": 0.0, "consist": 0.0}
    n_samples = 0

    all_warn_logits = []
    all_warn_labels = []

    all_warn_valid = []

    amp_dtype = torch.float16 if use_amp else None

    for batch_i, batch in enumerate(loader):
        temp   = batch["temp_field"].to(device, non_blocking=True)
        smoke  = batch["smoke"].to(device, non_blocking=True)
        ld     = batch["layer_domain"].to(device, non_blocking=True)
        msk_t  = batch["mask_temp"].to(device, non_blocking=True)
        msk_s  = batch["mask_smoke"].to(device, non_blocking=True)
        future = batch["future"].to(device, non_blocking=True)
        warn   = batch["warn"].to(device, non_blocking=True)
        warn_valid = batch.get("warn_valid", torch.ones_like(warn)).to(device, non_blocking=True)

        if debug_nan_checks:
            to_check = {
                "temp_field": temp,
                "smoke": smoke,
                "layer_domain": ld,
                "mask_temp": msk_t,
                "mask_smoke": msk_s,
                "future": future,
                "warn": warn,
                "warn_valid": warn_valid,
            }
            bad = {k: (~torch.isfinite(v)).sum().item() for k, v in to_check.items() if v is not None}
            bad_nonzero = {k: v for k, v in bad.items() if v > 0}
            if bad_nonzero:
                raise RuntimeError(
                    f"NaN/Inf in inputs at batch_i={batch_i}: {bad_nonzero}"
                )

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            with autocast("cuda", enabled=use_amp and device != "cpu", dtype=amp_dtype):
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
                            f"NaN/Inf detected in model outputs at batch_i={batch_i}: "
                            f"warn_logits_bad={wl_bad}, mu_bad={mu_bad}, logvar_bad={lv_bad}"
                        )
                losses = criterion(outputs, {
                    "future": future, "warn": warn, "warn_valid": warn_valid,
                })
                loss = losses["total"]
                if debug_nan_checks:
                    if (not torch.isfinite(loss).all()):
                        raise RuntimeError(
                            f"NaN/Inf detected in loss at batch_i={batch_i}: loss={loss.item()}"
                        )

            if is_train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()

        bs = temp.size(0)
        total_loss += loss.item() * bs
        for k in loss_parts:
            loss_parts[k] += losses[k].item() * bs
        n_samples += bs

        all_warn_logits.append(outputs["warn_logits"].detach().cpu())
        all_warn_labels.append(warn.detach().cpu())
        all_warn_valid.append(warn_valid.detach().cpu())

    avg_loss = total_loss / max(n_samples, 1)
    for k in loss_parts:
        loss_parts[k] /= max(n_samples, 1)

    wl = torch.cat(all_warn_logits).numpy()
    wt = torch.cat(all_warn_labels).numpy()
    wv = torch.cat(all_warn_valid).numpy()

    valid_mask = wv > 0.5
    wl_v = wl[valid_mask]
    wt_v = wt[valid_mask]
    warn_probs = 1.0 / (1.0 + np.exp(-np.clip(wl_v, -500, 500)))
    warn_preds = (warn_probs >= 0.5).astype(int)

    if len(wt_v) > 0:
        warn_acc = float((warn_preds == wt_v.astype(int)).mean())
    else:
        warn_acc = float("nan")

    from sklearn.metrics import f1_score, roc_auc_score
    warn_f1 = float(f1_score(wt_v.astype(int), warn_preds, zero_division=0)) if len(wt_v) > 0 else float("nan")
    try:
        warn_auc = float(roc_auc_score(wt_v.astype(int), warn_probs))
    except ValueError:
        warn_auc = float("nan")

    return dict(
        loss=avg_loss,
        **{f"loss_{k}": v for k, v in loss_parts.items()},
        warn_acc=warn_acc,
        warn_f1=warn_f1,
        warn_auc=warn_auc,
        n_samples=n_samples,
    )


# ───────────────── load pre-trained backbone ─────────────────

def _load_pretrained_backbone(model: TriModalFireModel, cfg: Config, device: str):
    """
    Load encoder + fusion weights from contrastive pre-training checkpoint.
    Prediction and warning heads remain randomly initialised.
    """
    ckpt_path = Path(cfg.output_dir) / "pretrained_backbone.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Pre-trained backbone not found at {ckpt_path}.  "
            "Run `python pretrain.py` first."
        )
    ckpt = load_torch_checkpoint(ckpt_path, map_location=device)
    backbone_sd = ckpt["backbone_state_dict"]

    model_sd = model.state_dict()
    loaded, skipped = 0, 0
    for k, v in backbone_sd.items():
        if k in model_sd and model_sd[k].shape == v.shape:
            model_sd[k] = v
            loaded += 1
        else:
            skipped += 1

    model.load_state_dict(model_sd)
    print(f"  Loaded {loaded} pre-trained backbone tensors "
          f"(skipped {skipped}, epoch {ckpt.get('epoch', '?')})")
    return model


def _build_optimizer_with_differential_lr(model: TriModalFireModel, cfg: Config):
    """
    Backbone (encoders + fusion) → lower lr
    Heads (pred_head + warn_head) → higher lr
    """
    backbone_names = {"temp_encoder", "layer_encoder", "smoke_encoder", "fusion"}
    backbone_params, head_params = [], []

    for name, param in model.named_parameters():
        if any(name.startswith(bn) for bn in backbone_names):
            backbone_params.append(param)
        else:
            head_params.append(param)

    param_groups = [
        {"params": backbone_params, "lr": cfg.finetune_lr_backbone},
        {"params": head_params, "lr": cfg.finetune_lr_head},
    ]
    print(f"  Backbone params: {sum(p.numel() for p in backbone_params):,}  "
          f"lr={cfg.finetune_lr_backbone}")
    print(f"  Head params:     {sum(p.numel() for p in head_params):,}  "
          f"lr={cfg.finetune_lr_head}")

    return torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)


# ───────────────── main training ─────────────────

def train(cfg: Config | None = None, pretrained: bool = False):
    if cfg is None:
        cfg = Config()

    set_seed(cfg.seed)
    device = cfg.resolve_device()
    print(f"Device: {device}")

    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    cfg.make_dirs()
    out_dir = Path(cfg.output_dir)

    # ── data ──
    print("\n═══ Building data loaders ═══")
    train_loader, _train_eval_loader, val_loader, test_loader, scaler = build_dataloaders(cfg)

    # ── model ──
    mode_str = "Fine-tuning (pre-trained)" if pretrained else "Training from scratch"
    print(f"\n═══ {mode_str} ═══")
    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model type = {getattr(cfg, 'model_type', 'trimodal')}")
    print(f"  Trainable parameters: {n_params:,}")
    print(f"  fusion_mode = {getattr(cfg, 'fusion_mode', 'layer_bridge')}")
    print(
        f"  lambda_pred = {cfg.lambda_pred}  lambda_warn = {cfg.lambda_warn}  "
        f"lambda_trend = {cfg.lambda_trend}  lambda_consist = {cfg.lambda_consist}"
    )
    print(f"  debug_nan_checks = {bool(getattr(cfg, 'debug_nan_checks', False))}")

    if pretrained:
        _load_pretrained_backbone(model, cfg, device)

    # ── loss ──
    train_lc = train_loader.dataset.label_counts()
    n_neg = max(train_lc["negative"], 1)
    n_pos = max(train_lc["positive"], 1)
    n_pred_only = train_lc.get("prediction_only", 0)
    raw_pw = n_neg / n_pos
    cap = float(getattr(cfg, "warn_pos_weight_cap", 35.0))
    warn_pos_weight = min(raw_pw, cap) if cap > 0 else raw_pw
    if warn_pos_weight < raw_pw:
        print(f"  Warning pos_weight = {warn_pos_weight:.2f} (capped from {raw_pw:.2f})  "
              f"(neg={n_neg}, pos={n_pos}, pred_only={n_pred_only})")
    else:
        print(f"  Warning pos_weight = {warn_pos_weight:.2f}  "
              f"(neg={n_neg}, pos={n_pos}, pred_only={n_pred_only})")
    criterion = CombinedLoss(cfg, pos_weight=warn_pos_weight).to(device)

    # ── optimizer ──
    if pretrained:
        optimizer = _build_optimizer_with_differential_lr(model, cfg)
        peak_lr = cfg.finetune_lr_head
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
        )
        peak_lr = cfg.lr

    warmup = getattr(cfg, "warmup_epochs", 5)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_epochs=warmup, total_epochs=cfg.epochs,
    )

    # ── AMP scaler ──
    use_amp = bool(getattr(cfg, "use_amp", False)) and device == "cuda"
    grad_scaler = GradScaler("cuda") if use_amp else None
    if use_amp:
        print("  Mixed-precision training (AMP fp16) enabled")

    # ── training loop ──
    print(f"\n═══ Training ({cfg.epochs} epochs, warmup={warmup}) ═══")
    history = []
    best_val_score = float("inf")
    best_epoch = -1
    patience_count = 0
    best_model_path = out_dir / "best_model.pt"

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()

        train_m = run_epoch(
            model, train_loader, criterion, optimizer, device,
            debug_nan_checks=bool(getattr(cfg, "debug_nan_checks", False)),
            scaler=grad_scaler, use_amp=use_amp,
        )
        val_m   = run_epoch(
            model, val_loader, criterion, None, device,
            debug_nan_checks=bool(getattr(cfg, "debug_nan_checks", False)),
            use_amp=use_amp,
        )
        scheduler.step()

        elapsed = time.time() - t0

        record = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"]}
        for prefix, m in [("train", train_m), ("val", val_m)]:
            for k, v in m.items():
                record[f"{prefix}_{k}"] = v
        history.append(record)

        # Early stopping metric: use -AUC (lower=better) when available,
        # otherwise fall back to warning loss.  This avoids the instability
        # of total loss dominated by the Gaussian NLL prediction term.
        # NB: NaN < inf is False in Python — a non-finite val_score would
        # prevent *any* checkpoint from being saved (LOOCV then crashes).
        val_auc = val_m["warn_auc"]
        vf1 = val_m.get("warn_f1", 0.0)
        if not isinstance(vf1, (int, float)) or not math.isfinite(float(vf1)):
            vf1 = 0.0
        else:
            vf1 = float(vf1)
        if not np.isnan(val_auc):
            # Primary: val AUC; F1 weighted more heavily so early stopping does not
            # lock in underfit checkpoints when AUC peaks in the first few epochs.
            f1_bonus = 0.25 * vf1 if vf1 > 0 else 0.0
            val_score = -float(val_auc) - f1_bonus
        elif vf1 > 0:
            val_score = -vf1
        else:
            lw = float(val_m["loss_warn"])
            if not math.isfinite(lw):
                lw = float(val_m["loss"])
            if not math.isfinite(lw):
                lw = 0.0
            val_score = lw - 2.0 * vf1
        if not math.isfinite(val_score):
            val_score = float(val_m["loss"])
        if not math.isfinite(val_score):
            val_score = 0.0

        marker = ""
        if val_score < best_val_score:
            marker = "  ★"

        print(
            f"Epoch {epoch:03d}  "
            f"train_loss={train_m['loss']:.4f}  val_loss={val_m['loss']:.4f}  "
            f"val_warn_f1={val_m['warn_f1']:.4f}  "
            f"val_warn_auc={val_m['warn_auc']:.4f}{marker}  "
            f"[{elapsed:.1f}s]"
        )

        if val_score < best_val_score:
            best_val_score = val_score
            best_epoch = epoch
            patience_count = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_score": val_score,
                    "val_loss": val_m["loss"],
                    "pretrained": pretrained,
                    "config": vars(cfg),
                },
                best_model_path,
            )
        else:
            patience_count += 1
            if patience_count >= cfg.early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # ── save history ──
    import pandas as pd
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(out_dir / "train_history.csv", index=False)

    if best_epoch < 0 or not best_model_path.is_file():
        if not history:
            raise RuntimeError(
                "No training epochs ran (cfg.epochs < 1?). Cannot write best_model.pt."
            )
        print(
            "\nWARNING: no best checkpoint was written (e.g. all val scores were NaN). "
            "Saving last-epoch model weights to best_model.pt."
        )
        best_epoch = int(history[-1]["epoch"])
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "epoch": best_epoch,
                "val_score": float("nan"),
                "val_loss": float("nan"),
                "pretrained": pretrained,
                "config": vars(cfg),
            },
            best_model_path,
        )

    # ── load best & evaluate on test ──
    print(f"\n═══ Loading best model (epoch {best_epoch}) ═══")
    ckpt = load_torch_checkpoint(best_model_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    test_m = run_epoch(model, test_loader, criterion, None, device, use_amp=use_amp)

    print("\n═══ TEST RESULTS ═══")
    print(f"  loss       = {test_m['loss']:.4f}")
    print(f"  loss_pred  = {test_m['loss_pred']:.4f}")
    print(f"  loss_warn  = {test_m['loss_warn']:.4f}")
    print(f"  loss_trend = {test_m['loss_trend']:.4f}")
    print(f"  warn_acc   = {test_m['warn_acc']:.4f}")
    print(f"  warn_f1    = {test_m['warn_f1']:.4f}")
    print(f"  warn_auc   = {test_m['warn_auc']:.4f}")
    print("═══════════════════")

    # ── save run config ──
    run_info = {
        "best_epoch": best_epoch,
        "best_val_score": best_val_score,
        "test_metrics": test_m,
        "n_params": n_params,
        "pretrained": pretrained,
        "config": {k: v for k, v in vars(cfg).items() if not k.startswith("_")},
    }
    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_info, f, ensure_ascii=False, indent=2, default=str)

    print(f"\nAll outputs saved to {out_dir}")
    return model, test_m


if __name__ == "__main__":
    import argparse
    _fusion_choices = ("layer_bridge", "direct_smoke_temp", "late_concat")
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", action="store_true",
                        help="Load pre-trained backbone from pretrain.py")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument(
        "--fusion-mode", type=str, default=None, choices=_fusion_choices,
        help="Fusion ablation: layer_bridge | direct_smoke_temp | late_concat",
    )
    parser.add_argument(
        "--lambda-trend", type=float, default=None,
        help="Weight for trend (slope) loss; use 0 for w/o slope ablation",
    )
    parser.add_argument(
        "--lambda-consist", type=float, default=None,
        help="Weight for physical consistency loss",
    )
    parser.add_argument(
        "--lambda-pred", type=float, default=None,
        help="Weight for prediction loss; 0 for warn-only ablation (RQ3)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Override cfg.output_dir for this run",
    )
    args = parser.parse_args()

    cfg = Config()
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.lr is not None:
        cfg.lr = args.lr
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

    from data_pipeline.preprocess import discover_and_configure
    discover_and_configure(cfg)

    train(cfg, pretrained=args.pretrained)

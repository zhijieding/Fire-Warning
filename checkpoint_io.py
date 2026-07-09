"""Safe torch.load: prefer weights_only (PyTorch 2.x), with fallbacks."""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import torch


def load_torch_checkpoint(path: str | Path, map_location: str | torch.device) -> Any:
    path = str(path)
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)
    except Exception as e:
        warnings.warn(
            f"weights_only load failed ({e!r}); falling back to full pickle — "
            "only load checkpoints you trust.",
            stacklevel=2,
        )
        try:
            return torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=map_location)

from .model import TriModalFireModel
from .losses import CombinedLoss
from .baselines import BaselineFireModel


def build_model(cfg):
    """Factory: return TriModalFireModel or BaselineFireModel based on cfg.model_type.

    ``cfg.model_type`` (default ``"trimodal"``) selects the architecture:
        - ``"trimodal"``          → main TriModalFireModel (cross-attention fusion)
        - ``"lstm"`` / ``"bilstm"`` / ``"gru"``  → RNN baseline
        - ``"tcn"``               → TCN baseline
        - ``"informer"``          → Informer baseline (ProbSparse attention)
        - ``"transformer_noxa"``  → Transformer-encoder baseline (no cross-attn)
        - ``"patchtst"``          → PatchTST baseline
        - ``"itransformer"``      → iTransformer baseline
        - ``"timesnet"``          → TimesNet baseline

    All returned modules share the same forward signature / output dict.
    """
    mt = str(getattr(cfg, "model_type", "trimodal")).lower()
    if mt == "trimodal":
        return TriModalFireModel(cfg)
    return BaselineFireModel(cfg)


__all__ = ["TriModalFireModel", "BaselineFireModel", "CombinedLoss", "build_model"]

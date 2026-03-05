from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover - optional dependency
    torch = None
    nn = None

from .opportunity_model import SharedTCNEncoder


class DirectionModel(nn.Module):  # type: ignore[misc]
    """Direction classifier with outputs ordered as [long, short]."""

    def __init__(
        self,
        seq_features: int,
        ctx_dim: int,
        regime_dim: int = 4,
        tcn_channels: tuple[int, ...] = (32, 32),
        hidden: int = 64,
        dropout: float = 0.1,
    ):
        if nn is None:
            raise ImportError("PyTorch is required for DirectionModel.")
        super().__init__()
        self.encoder = SharedTCNEncoder(
            in_features=seq_features,
            channels=tcn_channels,
            kernel_size=3,
            dropout=dropout,
        )
        in_dim = int(self.encoder.embedding_dim + ctx_dim + regime_dim)
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),  # logits for [long, short]
        )

    def forward_logits(self, seq: torch.Tensor, ctx: torch.Tensor, regime: torch.Tensor) -> torch.Tensor:
        z_seq = self.encoder(seq)
        if ctx.ndim != 2 or regime.ndim != 2:
            raise ValueError("ctx and regime must be 2D [B, D].")
        z = torch.cat([z_seq, ctx, regime], dim=1)
        return self.head(z)

    def forward(self, seq: torch.Tensor, ctx: torch.Tensor, regime: torch.Tensor) -> torch.Tensor:
        logits = self.forward_logits(seq, ctx, regime)
        return torch.softmax(logits, dim=1)  # [B,2] -> [long, short]


def train_direction_model(
    model: DirectionModel,
    seq: np.ndarray | torch.Tensor,
    ctx: np.ndarray | torch.Tensor,
    regime: np.ndarray | torch.Tensor,
    y_direction: np.ndarray | torch.Tensor,
    *,
    epochs: int = 20,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Train with masked CE loss.

    y_direction convention:
    - 1 -> long class (index 0)
    - 0 -> optional long class alias (index 0)
    - -1 -> ignore sample
    - 2 -> optional short class alias (index 1)
    """
    if torch is None:
        raise ImportError("PyTorch is required for train_direction_model.")

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(dev)
    model.train()

    x_seq = torch.as_tensor(seq, dtype=torch.float32, device=dev)
    x_ctx = torch.as_tensor(ctx, dtype=torch.float32, device=dev)
    x_reg = torch.as_tensor(regime, dtype=torch.float32, device=dev)
    y_raw = torch.as_tensor(y_direction, dtype=torch.long, device=dev).view(-1)

    if x_seq.shape[0] != x_ctx.shape[0] or x_seq.shape[0] != x_reg.shape[0] or x_seq.shape[0] != y_raw.shape[0]:
        raise ValueError("Batch dimension mismatch among inputs/targets.")

    # Normalize target encoding into {0: long, 1: short, -1: ignore}
    # Supported inputs:
    # - Stage labeler: 1=long, 0=short, -1 ignore
    # - Alternate: 0=long, 1=short, -1 ignore (set metadata/transform upstream if needed)
    # - Optional explicit 2=short alias
    y = torch.full_like(y_raw, -1)
    y = torch.where(y_raw == 1, torch.zeros_like(y), y)  # long -> class 0
    y = torch.where((y_raw == 0) | (y_raw == 2), torch.ones_like(y), y)  # short -> class 1
    y = torch.where(y_raw == -1, torch.full_like(y, -1), y)

    criterion = nn.CrossEntropyLoss(reduction="none")
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    losses = []
    valid_counts = []
    for _ in range(int(epochs)):
        opt.zero_grad()
        logits = model.forward_logits(x_seq, x_ctx, x_reg)
        valid = y != -1
        if not torch.any(valid):
            raise ValueError("No valid direction labels after masking (all y_direction == -1).")
        loss_vec = criterion(logits[valid], y[valid])
        loss = loss_vec.mean()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu().item()))
        valid_counts.append(int(valid.sum().detach().cpu().item()))

    return {
        "loss_history": losses,
        "final_loss": float(losses[-1]) if losses else np.nan,
        "valid_count": int(valid_counts[-1]) if valid_counts else 0,
    }

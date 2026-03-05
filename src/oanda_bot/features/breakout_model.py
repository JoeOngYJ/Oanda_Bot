from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover - optional dependency
    torch = None
    nn = None

from .opportunity_model import SharedTCNEncoder


class BreakoutModel(nn.Module):  # type: ignore[misc]
    """Breakout probability model: TCN encoder + sigmoid head."""

    def __init__(
        self,
        seq_features: int,
        tcn_channels: tuple[int, ...] = (32, 32),
        hidden: int = 64,
        dropout: float = 0.1,
    ):
        if nn is None:
            raise ImportError("PyTorch is required for BreakoutModel.")
        super().__init__()
        self.encoder = SharedTCNEncoder(
            in_features=seq_features,
            channels=tcn_channels,
            kernel_size=3,
            dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(int(self.encoder.embedding_dim), int(hidden)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden), 1),
        )

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        z = self.encoder(seq)
        logits = self.head(z).squeeze(-1)
        return torch.sigmoid(logits)  # [B]


def make_breakout_labels(
    df: pd.DataFrame,
    *,
    horizon_bars: int = 8,
    compression_pct_threshold: float = 0.20,
    net_ret_threshold: float = 0.002,
    bb_width_pct_col: str = "bb_width_pct",
    opportunity_col: str = "y_opportunity",
) -> pd.DataFrame:
    """Create breakout labels.

    Breakout = compression and large move and opportunity.
    Rules:
    - compression: BB width percentile < compression threshold
    - magnitude: abs(net_ret) > net_ret_threshold within horizon
    - opportunity: y_opportunity == 1
    """
    if bb_width_pct_col not in df.columns:
        raise ValueError(f"Missing '{bb_width_pct_col}' column.")
    if opportunity_col not in df.columns:
        raise ValueError(f"Missing '{opportunity_col}' column.")
    if horizon_bars <= 0:
        raise ValueError("horizon_bars must be > 0.")

    out = df.copy()
    if "net_ret" in out.columns:
        net_ret = pd.to_numeric(out["net_ret"], errors="coerce").astype(float)
    else:
        if "close" not in out.columns:
            raise ValueError("Need 'net_ret' or 'close' column to derive breakout magnitude.")
        close = pd.to_numeric(out["close"], errors="coerce").astype(float)
        net_ret = (close.shift(-horizon_bars) / close) - 1.0
        out["net_ret"] = net_ret

    compression = pd.to_numeric(out[bb_width_pct_col], errors="coerce").astype(float) < float(compression_pct_threshold)
    magnitude = net_ret.abs() > float(net_ret_threshold)
    opportunity = pd.to_numeric(out[opportunity_col], errors="coerce").astype(float) == 1.0

    label = (compression & magnitude & opportunity).astype(float)
    if len(out) > 0:
        tail_idx = out.index[-min(horizon_bars, len(out)):]
        label.loc[tail_idx] = np.nan

    out["y_breakout"] = label
    return out


def train_breakout_model(
    model: BreakoutModel,
    seq: np.ndarray | torch.Tensor,
    y_breakout: np.ndarray | torch.Tensor,
    *,
    epochs: int = 20,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Train breakout model with BCE on y_breakout."""
    if torch is None:
        raise ImportError("PyTorch is required for train_breakout_model.")

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(dev)
    model.train()

    x_seq = torch.as_tensor(seq, dtype=torch.float32, device=dev)
    y = torch.as_tensor(y_breakout, dtype=torch.float32, device=dev).view(-1)
    if x_seq.shape[0] != y.shape[0]:
        raise ValueError("Batch dimension mismatch between seq and y_breakout.")

    criterion = nn.BCELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    losses = []
    for _ in range(int(epochs)):
        opt.zero_grad()
        p = model(x_seq)
        loss = criterion(p, y)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu().item()))

    return {"loss_history": losses, "final_loss": float(losses[-1]) if losses else np.nan}

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


class MeanReversionModel(nn.Module):  # type: ignore[misc]
    """Mean reversion probability model: TCN encoder + sigmoid head."""

    def __init__(
        self,
        seq_features: int,
        tcn_channels: tuple[int, ...] = (32, 32),
        hidden: int = 64,
        dropout: float = 0.1,
    ):
        if nn is None:
            raise ImportError("PyTorch is required for MeanReversionModel.")
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


def make_mean_reversion_labels(
    df: pd.DataFrame,
    *,
    horizon_bars: int = 8,
    z_k: float = 1.5,
    net_ret_threshold: float = 0.0015,
    zscore_col: str = "price_zscore",
    close_col: str = "close",
    opportunity_col: str = "y_opportunity",
) -> pd.DataFrame:
    """Create mean reversion labels.

    Label is 1 when:
    - price z-score is beyond +/- k
    - subsequent net return over horizon reverts toward mean
      (zscore>0 => net_ret < -thr, zscore<0 => net_ret > +thr)
    - market is in/near range regime (if regime hints are available)
    """
    if horizon_bars <= 0:
        raise ValueError("horizon_bars must be > 0.")

    out = df.copy()
    if zscore_col in out.columns:
        z = pd.to_numeric(out[zscore_col], errors="coerce").astype(float)
    else:
        if close_col not in out.columns:
            raise ValueError(f"Need '{zscore_col}' or '{close_col}' to compute mean-reversion labels.")
        close = pd.to_numeric(out[close_col], errors="coerce").astype(float)
        mu = close.rolling(20, min_periods=20).mean()
        sd = close.rolling(20, min_periods=20).std(ddof=0)
        z = (close - mu) / (sd + 1e-9)
        out[zscore_col] = z

    if "net_ret" in out.columns:
        net_ret = pd.to_numeric(out["net_ret"], errors="coerce").astype(float)
    else:
        if close_col not in out.columns:
            raise ValueError("Need 'net_ret' or 'close' to derive horizon return.")
        close = pd.to_numeric(out[close_col], errors="coerce").astype(float)
        net_ret = (close.shift(-horizon_bars) / close) - 1.0
        out["net_ret"] = net_ret

    if opportunity_col in out.columns:
        opp = pd.to_numeric(out[opportunity_col], errors="coerce").fillna(0.0) == 1.0
    else:
        opp = pd.Series(True, index=out.index)

    extreme = z.abs() >= float(z_k)
    reverted = ((z > 0.0) & (net_ret < -float(net_ret_threshold))) | ((z < 0.0) & (net_ret > float(net_ret_threshold)))
    range_mask = _range_regime_mask(out)

    label = (extreme & reverted & opp & range_mask).astype(float)
    if len(out) > 0:
        tail_idx = out.index[-min(horizon_bars, len(out)):]
        label.loc[tail_idx] = np.nan

    out["y_mean_reversion"] = label
    return out


def _range_regime_mask(df: pd.DataFrame) -> pd.Series:
    if "regime" in df.columns:
        r = df["regime"]
        if np.issubdtype(r.dtype, np.number):
            # Convention used elsewhere: 1=range in 4-class targets.
            return pd.to_numeric(r, errors="coerce").fillna(-999).astype(int) == 1
        return r.astype(str).str.lower().isin({"range", "ranging", "mean_reversion"})

    for c in ("regime_prob_range", "range_prob"):
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce").fillna(0.0) >= 0.5

    return pd.Series(True, index=df.index)


def train_mean_reversion_model(
    model: MeanReversionModel,
    seq: np.ndarray | torch.Tensor,
    y_mean_reversion: np.ndarray | torch.Tensor,
    *,
    epochs: int = 20,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Train mean reversion model with BCE."""
    if torch is None:
        raise ImportError("PyTorch is required for train_mean_reversion_model.")

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(dev)
    model.train()

    x_seq = torch.as_tensor(seq, dtype=torch.float32, device=dev)
    y = torch.as_tensor(y_mean_reversion, dtype=torch.float32, device=dev).view(-1)
    if x_seq.shape[0] != y.shape[0]:
        raise ValueError("Batch dimension mismatch between seq and y_mean_reversion.")

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

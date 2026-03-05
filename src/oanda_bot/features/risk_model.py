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


class RiskModel(nn.Module):  # type: ignore[misc]
    """Risk regressor MLP: in_dim -> 64 -> 32 -> 1 with dropout 0.1."""

    def __init__(self, in_dim: int = 32, dropout: float = 0.1):
        if nn is None:
            raise ImportError("PyTorch is required for RiskModel.")
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(in_dim), 64),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def make_risk_labels(
    df: pd.DataFrame,
    *,
    horizon_bars: int = 8,
    method: str = "realized_vol",
    atr_col: str = "atr",
    atr_period: int = 14,
) -> pd.DataFrame:
    """Create forward-looking risk labels.

    method:
    - "realized_vol": std of next H M15 returns
    - "future_atr": mean ATR over next H bars (uses atr_col if present, else computes)
    """
    if horizon_bars <= 0:
        raise ValueError("horizon_bars must be > 0.")
    if "close" not in df.columns:
        raise ValueError("make_risk_labels requires 'close' column.")

    out = df.copy()
    m = str(method).lower()
    if m not in {"realized_vol", "future_atr"}:
        raise ValueError("method must be one of {'realized_vol', 'future_atr'}.")

    close = pd.to_numeric(out["close"], errors="coerce").astype(float)
    ret = close.pct_change()

    if m == "realized_vol":
        fut = pd.concat([ret.shift(-i) for i in range(1, horizon_bars + 1)], axis=1)
        y = fut.std(axis=1, ddof=0)
        out["y_risk"] = y
    else:
        if atr_col in out.columns:
            atr = pd.to_numeric(out[atr_col], errors="coerce").astype(float)
        else:
            for c in ("high", "low"):
                if c not in out.columns:
                    raise ValueError(f"future_atr requires '{atr_col}' or OHLC columns including '{c}'.")
            atr = _atr(
                pd.to_numeric(out["high"], errors="coerce").astype(float),
                pd.to_numeric(out["low"], errors="coerce").astype(float),
                close,
                period=int(atr_period),
            )
            out[atr_col] = atr

        fut_atr = pd.concat([atr.shift(-i) for i in range(1, horizon_bars + 1)], axis=1)
        out["y_risk"] = fut_atr.mean(axis=1)

    if len(out) > 0:
        tail_idx = out.index[-min(horizon_bars, len(out)):]
        out.loc[tail_idx, "y_risk"] = np.nan
    return out


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr1 = (high - low).abs()
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def train_risk_model(
    model: RiskModel,
    x: np.ndarray | torch.Tensor,
    y_risk: np.ndarray | torch.Tensor,
    *,
    epochs: int = 25,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Train risk model with Huber loss (SmoothL1). Ignores NaN targets."""
    if torch is None:
        raise ImportError("PyTorch is required for train_risk_model.")

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(dev)
    model.train()

    x_t = torch.as_tensor(x, dtype=torch.float32, device=dev)
    y_t = torch.as_tensor(y_risk, dtype=torch.float32, device=dev).view(-1)
    if x_t.shape[0] != y_t.shape[0]:
        raise ValueError("x and y_risk batch dimensions must match.")

    criterion = nn.SmoothL1Loss(reduction="none")
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    losses = []
    valid_counts = []
    for _ in range(int(epochs)):
        opt.zero_grad()
        pred = model(x_t)
        valid = torch.isfinite(y_t)
        if not torch.any(valid):
            raise ValueError("No finite y_risk values available for training.")
        loss_vec = criterion(pred[valid], y_t[valid])
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

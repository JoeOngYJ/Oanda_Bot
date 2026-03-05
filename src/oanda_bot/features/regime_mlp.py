from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover - optional dependency
    torch = None
    nn = None


class RegimeMLP(nn.Module):  # type: ignore[misc]
    """Simple MLP for 4-class regime classification."""

    def __init__(self, ctx_dim: int = 16, hidden: int = 32, out: int = 4, dropout: float = 0.1):
        if nn is None:
            raise ImportError("PyTorch is required for RegimeMLP.")
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(ctx_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class RegimeTargetConfig:
    slope_high_threshold: float = 8e-4
    slope_low_threshold: float = 2e-4
    adx_high_threshold: float = 0.25
    adx_low_threshold: float = 0.18
    vol_high_threshold: float = 0.80
    spread_high_threshold: float = 0.80
    illiquid_hours_utc: Iterable[int] = (22, 23, 0, 1, 2, 3, 4, 5)


def _pick_first_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def make_regime_targets(ctx_df: pd.DataFrame, config: Optional[RegimeTargetConfig] = None) -> pd.Series:
    """Create 4-class regime labels from heuristic rules.

    Classes:
    - 0: trend
    - 1: range
    - 2: highvol
    - 3: lowliq
    """
    cfg = config or RegimeTargetConfig()
    df = ctx_df.copy()

    slope_col = _pick_first_col(df, ("h1_slope", "ctx_h1_slope", "slope"))
    adx_col = _pick_first_col(df, ("h1_adx", "adx_14", "adx", "ctx_h1_adx"))
    vol_col = _pick_first_col(df, ("h1_vol_pct", "vol_pct", "ctx_h1_vol_pct", "atr_pct"))
    spread_col = _pick_first_col(df, ("spread_est_pct", "spread_est_percentile", "spread_pct", "spread_est"))

    if slope_col is None or adx_col is None or vol_col is None:
        raise ValueError("ctx_df must include slope, ADX, and vol-percentile-like columns.")

    slope = pd.to_numeric(df[slope_col], errors="coerce").fillna(0.0).abs()
    adx = pd.to_numeric(df[adx_col], errors="coerce").fillna(0.0)
    if adx.max() > 1.5:
        adx = adx / 100.0

    vol_pct = pd.to_numeric(df[vol_col], errors="coerce").fillna(0.0)
    if spread_col is not None:
        spread_val = pd.to_numeric(df[spread_col], errors="coerce").fillna(0.0)
        if spread_col in ("spread_est",):
            spread_pct = spread_val.rank(pct=True)
        else:
            spread_pct = spread_val
    else:
        spread_pct = pd.Series(0.0, index=df.index)

    idx_utc = pd.to_datetime(df.index, utc=True, errors="coerce")
    illiquid_mask = pd.Series(idx_utc.hour.isin(set(int(h) for h in cfg.illiquid_hours_utc)), index=df.index)

    trend = (slope >= float(cfg.slope_high_threshold)) & (adx >= float(cfg.adx_high_threshold))
    range_ = (slope <= float(cfg.slope_low_threshold)) & (adx <= float(cfg.adx_low_threshold))
    highvol = vol_pct > float(cfg.vol_high_threshold)
    lowliq = (spread_pct > float(cfg.spread_high_threshold)) | illiquid_mask

    # Priority: lowliq > highvol > trend > range, fallback to range.
    y = pd.Series(1, index=df.index, dtype=np.int64)
    y.loc[trend] = 0
    y.loc[range_] = 1
    y.loc[highvol] = 2
    y.loc[lowliq] = 3
    return y


def train_regime_mlp(
    model: RegimeMLP,
    ctx: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    *,
    epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    if torch is None:
        raise ImportError("PyTorch is required for train_regime_mlp.")

    x_np = np.asarray(ctx, dtype=np.float32)
    y_np = np.asarray(y, dtype=np.int64)
    if x_np.ndim != 2:
        raise ValueError("ctx must be 2D [N, C].")
    if y_np.ndim != 1:
        raise ValueError("y must be 1D [N].")
    if len(x_np) != len(y_np):
        raise ValueError("ctx/y length mismatch.")

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(dev)
    model.train()

    x = torch.tensor(x_np, dtype=torch.float32, device=dev)
    target = torch.tensor(y_np, dtype=torch.long, device=dev)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    losses = []
    for _ in range(int(epochs)):
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))

    return {"loss_history": losses, "final_loss": float(losses[-1]) if losses else np.nan}


def predict_regime_proba(
    model: RegimeMLP,
    ctx: np.ndarray | pd.DataFrame,
    *,
    device: Optional[str] = None,
) -> np.ndarray:
    if torch is None:
        raise ImportError("PyTorch is required for predict_regime_proba.")

    x_np = np.asarray(ctx, dtype=np.float32)
    if x_np.ndim != 2:
        raise ValueError("ctx must be 2D [N, C].")

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(dev)
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(x_np, dtype=torch.float32, device=dev))
        probs = torch.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()

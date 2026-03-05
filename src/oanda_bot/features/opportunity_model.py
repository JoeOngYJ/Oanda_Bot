from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover - optional dependency
    torch = None
    nn = None


class _Chomp1d(nn.Module):  # type: ignore[misc]
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = int(chomp_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class _TemporalBlock(nn.Module):  # type: ignore[misc]
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.chomp1 = _Chomp1d(pad)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.chomp2 = _Chomp1d(pad)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.out_relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.drop2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.out_relu(out + res)


class SharedTCNEncoder(nn.Module):  # type: ignore[misc]
    """Encode seq [B, T, F] -> embedding [B, E] with causal TCN."""

    def __init__(
        self,
        in_features: int,
        channels: tuple[int, ...] = (32, 32),
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        if nn is None:
            raise ImportError("PyTorch is required for SharedTCNEncoder.")
        super().__init__()
        layers = []
        in_ch = int(in_features)
        for i, out_ch in enumerate(channels):
            layers.append(
                _TemporalBlock(
                    in_ch=in_ch,
                    out_ch=int(out_ch),
                    kernel_size=int(kernel_size),
                    dilation=2 ** i,
                    dropout=float(dropout),
                )
            )
            in_ch = int(out_ch)
        self.tcn = nn.Sequential(*layers)
        self.embedding_dim = in_ch

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        if seq.ndim != 3:
            raise ValueError("Expected seq shape [B, T, F].")
        x = seq.transpose(1, 2)  # [B, F, T]
        h = self.tcn(x)  # [B, E, T]
        return h[:, :, -1]  # last time step embedding [B, E]


class OpportunityModel(nn.Module):  # type: ignore[misc]
    """Opportunity classifier.

    Inputs:
    - seq: [B, 128, F]
    - ctx: [B, C]
    - regime: [B, 4]
    Output:
    - p_trade: [B]
    """

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
            raise ImportError("PyTorch is required for OpportunityModel.")
        super().__init__()
        self.encoder = SharedTCNEncoder(
            in_features=seq_features,
            channels=tcn_channels,
            kernel_size=3,
            dropout=dropout,
        )
        head_in = int(self.encoder.embedding_dim + ctx_dim + regime_dim)
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, seq: torch.Tensor, ctx: torch.Tensor, regime: torch.Tensor) -> torch.Tensor:
        z_seq = self.encoder(seq)
        if ctx.ndim != 2 or regime.ndim != 2:
            raise ValueError("ctx and regime must be 2D [B, D].")
        z = torch.cat([z_seq, ctx, regime], dim=1)
        logits = self.head(z).squeeze(-1)
        return torch.sigmoid(logits)


def train_opportunity_model(
    model: OpportunityModel,
    seq: np.ndarray | torch.Tensor,
    ctx: np.ndarray | torch.Tensor,
    regime: np.ndarray | torch.Tensor,
    y_opportunity: np.ndarray | torch.Tensor,
    *,
    epochs: int = 20,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    if torch is None:
        raise ImportError("PyTorch is required for train_opportunity_model.")

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(dev)
    model.train()

    x_seq = torch.as_tensor(seq, dtype=torch.float32, device=dev)
    x_ctx = torch.as_tensor(ctx, dtype=torch.float32, device=dev)
    x_reg = torch.as_tensor(regime, dtype=torch.float32, device=dev)
    y = torch.as_tensor(y_opportunity, dtype=torch.float32, device=dev).view(-1)

    if x_seq.shape[0] != x_ctx.shape[0] or x_seq.shape[0] != x_reg.shape[0] or x_seq.shape[0] != y.shape[0]:
        raise ValueError("Batch dimension mismatch among inputs/targets.")

    criterion = nn.BCELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    losses = []
    for _ in range(int(epochs)):
        opt.zero_grad()
        p = model(x_seq, x_ctx, x_reg)
        loss = criterion(p, y)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu().item()))

    return {"loss_history": losses, "final_loss": float(losses[-1]) if losses else np.nan}

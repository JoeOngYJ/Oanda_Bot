from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover - optional dependency
    torch = None
    nn = None


class _CausalChomp1d(nn.Module):  # type: ignore[misc]
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = int(chomp_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size <= 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class _ResTCNBlock(nn.Module):  # type: ignore[misc]
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.chomp1 = _CausalChomp1d(pad)
        self.gn1 = nn.GroupNorm(num_groups=max(1, min(8, out_ch // 8)), num_channels=out_ch)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.chomp2 = _CausalChomp1d(pad)
        self.gn2 = nn.GroupNorm(num_groups=max(1, min(8, out_ch // 8)), num_channels=out_ch)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout)

        self.res_proj = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.out_act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.chomp1(y)
        y = self.gn1(y)
        y = self.act1(y)
        y = self.drop1(y)

        y = self.conv2(y)
        y = self.chomp2(y)
        y = self.gn2(y)
        y = self.act2(y)
        y = self.drop2(y)

        r = x if self.res_proj is None else self.res_proj(x)
        return self.out_act(y + r)


class SharedTCNEncoder(nn.Module):  # type: ignore[misc]
    """Sequence encoder:
    - input: [B, L=128, F]
    - linear F->64
    - causal residual TCN blocks with group norm
    - attention pooling -> [B, 128]
    """

    def __init__(self, seq_features: int):
        if nn is None:
            raise ImportError("PyTorch required for SharedTCNEncoder.")
        super().__init__()
        self.input_proj = nn.Linear(int(seq_features), 64)
        dilations = [1, 2, 4, 8, 16]
        channels = [64, 64, 96, 128, 128]
        drops = [0.10, 0.10, 0.15, 0.15, 0.20]
        blocks = []
        in_ch = 64
        for d, out_ch, dr in zip(dilations, channels, drops):
            blocks.append(_ResTCNBlock(in_ch, out_ch, kernel_size=3, dilation=d, dropout=dr))
            in_ch = out_ch
        self.tcn = nn.Sequential(*blocks)
        self.attn = nn.Linear(in_ch, 1)
        self.output_dim = 128
        self.out_proj = nn.Linear(in_ch, self.output_dim)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        if seq.ndim != 3:
            raise ValueError("seq must be [B, L, F]")
        x = self.input_proj(seq)  # [B,L,64]
        x = x.transpose(1, 2)  # [B,64,L]
        h = self.tcn(x).transpose(1, 2)  # [B,L,C]
        a = torch.softmax(self.attn(h).squeeze(-1), dim=1).unsqueeze(-1)  # [B,L,1]
        pooled = torch.sum(h * a, dim=1)  # [B,C]
        return self.out_proj(pooled)  # [B,128]


class OpportunityTCNModel(nn.Module):  # type: ignore[misc]
    def __init__(self, seq_features: int, ctx_dim: int):
        super().__init__()
        self.encoder = SharedTCNEncoder(seq_features)
        self.head = nn.Sequential(
            nn.Linear(128 + int(ctx_dim), 96),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(96, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, seq: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        z = self.encoder(seq)
        x = torch.cat([z, ctx], dim=1)
        return torch.sigmoid(self.head(x).squeeze(-1))


class DirectionTCNModel(nn.Module):  # type: ignore[misc]
    """Output probabilities [B,2] with order [long, short]."""

    def __init__(self, seq_features: int, ctx_dim: int):
        super().__init__()
        self.encoder = SharedTCNEncoder(seq_features)
        self.head = nn.Sequential(
            nn.Linear(128 + int(ctx_dim), 96),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(96, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 2),
        )

    def forward_logits(self, seq: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        z = self.encoder(seq)
        x = torch.cat([z, ctx], dim=1)
        return self.head(x)

    def forward(self, seq: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.forward_logits(seq, ctx), dim=1)


@dataclass
class TrainResult:
    best_val_loss: float
    epochs_ran: int


def seed_everything(seed: int = 42) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def train_opportunity_epoch(
    model: OpportunityTCNModel,
    batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 1.0,
) -> float:
    model.train()
    seq, ctx, y = [x.to(device) for x in batch]
    optimizer.zero_grad()
    p = model(seq, ctx)
    loss = nn.functional.binary_cross_entropy(p, y)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    return float(loss.detach().cpu().item())


def eval_opportunity(model: OpportunityTCNModel, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], device: torch.device) -> float:
    model.eval()
    with torch.no_grad():
        seq, ctx, y = [x.to(device) for x in batch]
        p = model(seq, ctx)
        loss = nn.functional.binary_cross_entropy(p, y)
    return float(loss.detach().cpu().item())


def train_direction_epoch(
    model: DirectionTCNModel,
    batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 1.0,
) -> float:
    model.train()
    seq, ctx, y_raw = [x.to(device) for x in batch]
    mask = y_raw != -1
    if not torch.any(mask):
        return 0.0
    y = y_raw[mask]
    optimizer.zero_grad()
    logits = model.forward_logits(seq, ctx)
    loss = nn.functional.cross_entropy(logits[mask], y)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    return float(loss.detach().cpu().item())


def eval_direction(model: DirectionTCNModel, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], device: torch.device) -> float:
    model.eval()
    with torch.no_grad():
        seq, ctx, y_raw = [x.to(device) for x in batch]
        mask = y_raw != -1
        if not torch.any(mask):
            return 0.0
        logits = model.forward_logits(seq, ctx)
        loss = nn.functional.cross_entropy(logits[mask], y_raw[mask])
    return float(loss.detach().cpu().item())

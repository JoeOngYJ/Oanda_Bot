from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .feature_builder import FeatureBuilder
from .labels import make_labels

try:
    import torch
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover - optional dependency
    torch = None

    class Dataset:  # type: ignore[override]
        pass


@dataclass
class FeatureLabelDataset(Dataset):
    """PyTorch Dataset for sequence/context features + labels.

    Supports:
    - precomputed feature store (parquet path or DataFrame)
    - on-the-fly computation from M15/H1/H4 OHLCV dataframes
    """

    instrument: str = ""
    feature_builder: FeatureBuilder = field(default_factory=FeatureBuilder)
    label_kwargs: Dict[str, Any] = None  # type: ignore[assignment]
    precomputed_df: Optional[pd.DataFrame] = None
    parquet_path: Optional[str] = None
    m15_df: Optional[pd.DataFrame] = None
    h1_df: Optional[pd.DataFrame] = None
    h4_df: Optional[pd.DataFrame] = None

    def __post_init__(self) -> None:
        if torch is None:
            raise ImportError("PyTorch is required for FeatureLabelDataset.")

        self.label_kwargs = dict(self.label_kwargs or {})
        self._mode = ""
        self._rows: Optional[pd.DataFrame] = None
        self._valid_times: List[pd.Timestamp] = []

        if self.parquet_path is not None:
            p = Path(self.parquet_path)
            self.precomputed_df = pd.read_parquet(p)

        if self.precomputed_df is not None:
            self._init_precomputed(self.precomputed_df)
            return

        if self.m15_df is None or self.h1_df is None or self.h4_df is None:
            raise ValueError("Provide either precomputed_df/parquet_path or m15_df+h1_df+h4_df.")
        self._init_on_the_fly(self.m15_df, self.h1_df, self.h4_df)

    def _init_precomputed(self, df: pd.DataFrame) -> None:
        self._mode = "precomputed"
        self._rows = df.reset_index(drop=True)

    def _init_on_the_fly(self, m15_df: pd.DataFrame, h1_df: pd.DataFrame, h4_df: pd.DataFrame) -> None:
        self._mode = "on_the_fly"
        self._m15 = self.feature_builder._normalize_ohlcv_index(m15_df)
        self._h1 = self.feature_builder._normalize_ohlcv_index(h1_df)
        self._h4 = self.feature_builder._normalize_ohlcv_index(h4_df)
        self._seq_features = self.feature_builder._build_m15_feature_frame(self._m15).replace([np.inf, -np.inf], np.nan)
        self._labeled = make_labels(self._m15, **self.label_kwargs)

        seq_len = int(self.feature_builder.seq_len)
        for ts in self._labeled.index:
            y_opp = self._labeled.at[ts, "y_opportunity"] if "y_opportunity" in self._labeled.columns else np.nan
            y_dir = self._labeled.at[ts, "y_direction"] if "y_direction" in self._labeled.columns else np.nan
            if pd.isna(y_opp) or pd.isna(y_dir):
                continue
            w = self._seq_features.loc[:ts].tail(seq_len)
            if len(w) != seq_len:
                continue
            if w.isna().any(axis=None):
                continue
            self._valid_times.append(ts)

    def __len__(self) -> int:
        if self._mode == "precomputed":
            assert self._rows is not None
            return len(self._rows)
        return len(self._valid_times)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self._mode == "precomputed":
            assert self._rows is not None
            row = self._rows.iloc[idx]
            seq = self._extract_seq(row)
            ctx = self._extract_ctx(row)
            y_opp = float(row["y_opportunity"]) if "y_opportunity" in row and pd.notna(row["y_opportunity"]) else 0.0
            y_dir = int(row["y_direction"]) if "y_direction" in row and pd.notna(row["y_direction"]) else -1
            meta = self._extract_meta(row)
        else:
            ts = self._valid_times[idx]
            w = self._seq_features.loc[:ts].tail(self.feature_builder.seq_len)
            seq = w.to_numpy(dtype=np.float32)
            ctx = self.feature_builder._build_context_vector(self._m15, self._h1, self._h4, ts).astype(np.float32)
            y_opp = float(self._labeled.at[ts, "y_opportunity"])
            y_dir = int(self._labeled.at[ts, "y_direction"]) if pd.notna(self._labeled.at[ts, "y_direction"]) else -1
            meta = {
                "datetime": ts,
                "instrument": self.instrument,
                "close": float(self._m15.at[ts, "close"]),
                "atr": float(self._seq_features.at[ts, "atr_14"]),
            }

        return {
            "seq": torch.as_tensor(seq, dtype=torch.float32),
            "ctx": torch.as_tensor(ctx, dtype=torch.float32),
            "y_opportunity": torch.tensor(y_opp, dtype=torch.float32),
            "y_direction": torch.tensor(y_dir, dtype=torch.int64),
            "meta": meta,
        }

    @staticmethod
    def _extract_seq(row: pd.Series) -> np.ndarray:
        if "seq" in row and row["seq"] is not None and not (isinstance(row["seq"], float) and np.isnan(row["seq"])):
            arr = np.asarray(row["seq"], dtype=np.float32)
            if arr.ndim != 2:
                raise ValueError("Precomputed 'seq' must be 2D [128,F].")
            return arr

        seq_cols = [c for c in row.index if str(c).startswith("seq_")]
        if not seq_cols:
            raise ValueError("Precomputed row missing 'seq' or 'seq_*' columns.")

        parsed = []
        for c in seq_cols:
            parts = str(c).split("_")
            if len(parts) != 3:
                continue
            t = int(parts[1])
            f = int(parts[2])
            parsed.append((t, f, float(row[c])))
        if not parsed:
            raise ValueError("No parseable seq_* columns found; expected seq_<t>_<f>.")
        t_max = max(x[0] for x in parsed) + 1
        f_max = max(x[1] for x in parsed) + 1
        out = np.zeros((t_max, f_max), dtype=np.float32)
        for t, f, v in parsed:
            out[t, f] = v
        return out

    @staticmethod
    def _extract_ctx(row: pd.Series) -> np.ndarray:
        if "ctx" in row and row["ctx"] is not None and not (isinstance(row["ctx"], float) and np.isnan(row["ctx"])):
            arr = np.asarray(row["ctx"], dtype=np.float32)
            if arr.ndim != 1:
                raise ValueError("Precomputed 'ctx' must be 1D [C].")
            return arr

        ctx_cols = [c for c in row.index if str(c).startswith("ctx_")]
        if not ctx_cols:
            raise ValueError("Precomputed row missing 'ctx' or 'ctx_*' columns.")
        ctx_cols = sorted(ctx_cols, key=lambda x: int(str(x).split("_")[1]))
        return row[ctx_cols].to_numpy(dtype=np.float32)

    @staticmethod
    def _extract_meta(row: pd.Series) -> Dict[str, Any]:
        if "meta" in row and isinstance(row["meta"], dict):
            return row["meta"]
        meta: Dict[str, Any] = {}
        for k in ["datetime", "instrument", "close", "atr"]:
            if k in row:
                meta[k] = row[k]
        return meta


def feature_label_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    if torch is None:
        raise ImportError("PyTorch is required for feature_label_collate_fn.")
    seq = torch.stack([x["seq"] for x in batch], dim=0)
    ctx = torch.stack([x["ctx"] for x in batch], dim=0)
    y_opp = torch.stack([x["y_opportunity"] for x in batch], dim=0)
    y_dir = torch.stack([x["y_direction"] for x in batch], dim=0)
    meta = [x["meta"] for x in batch]
    return {
        "seq": seq,
        "ctx": ctx,
        "y_opportunity": y_opp,
        "y_direction": y_dir,
        "meta": meta,
    }

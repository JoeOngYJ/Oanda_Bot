from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import scripts.xau_session_ingestion as xsi


def _make_csv(path: Path, index: pd.DatetimeIndex) -> Path:
    n = len(index)
    base = 2000.0
    close = base + np.linspace(0.0, 1.0, n)
    open_ = close - 0.1
    high = close + 0.2
    low = close - 0.2
    df = pd.DataFrame(
        {
            "timestamp": index.astype(str),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
        }
    )
    df.to_csv(path, index=False)
    return path


def test_duplicate_timestamp_rejection(tmp_path: Path):
    idx = pd.DatetimeIndex(
        [
            "2025-01-01T00:00:00+00:00",
            "2025-01-01T00:15:00+00:00",
            "2025-01-01T00:15:00+00:00",
            "2025-01-01T00:30:00+00:00",
        ]
    )
    p = _make_csv(tmp_path / "dup.csv", idx)
    with pytest.raises(ValueError, match="Duplicate timestamp"):
        xsi.load_ohlcv(str(p))


def test_broken_cadence_detection(tmp_path: Path):
    idx = pd.DatetimeIndex(
        [
            "2025-01-01T00:00:00+00:00",
            "2025-01-01T00:15:00+00:00",
            "2025-01-01T00:45:00+00:00",
            "2025-01-01T01:00:00+00:00",
        ]
    )
    p = _make_csv(tmp_path / "cadence.csv", idx)
    with pytest.raises(ValueError, match="15-minute cadence"):
        xsi.load_ohlcv(str(p))


def test_correct_bucket_assignment_at_boundaries():
    idx = pd.DatetimeIndex(
        [
            "2025-01-02T00:00:00+00:00",
            "2025-01-02T06:00:00+00:00",
            "2025-01-02T08:00:00+00:00",
            "2025-01-02T09:00:00+00:00",
            "2025-01-02T13:00:00+00:00",
            "2025-01-02T15:00:00+00:00",
        ],
        tz="UTC",
    )
    df = pd.DataFrame({"open": 1.0, "high": 1.1, "low": 0.9, "close": 1.0}, index=idx)
    cfg = xsi.default_session_config("UTC")
    out = xsi.assign_session_bucket(df, cfg, add_helper_columns=False)
    got = out["session_bucket"].astype(str).tolist()
    exp = [
        "asia",
        "pre_london",
        "london_open",
        "london_continuation",
        "ny_open",
        "ny_overlap_postdata",
    ]
    assert got == exp


def test_exactly_one_active_session_per_bar():
    idx = pd.date_range("2025-01-01", periods=96, freq="15min", tz="UTC")
    df = pd.DataFrame({"open": 1.0, "high": 1.2, "low": 0.8, "close": 1.0}, index=idx)
    cfg = xsi.default_session_config("UTC")
    out = xsi.assign_session_bucket(df, cfg, add_helper_columns=True)
    helper_cols = [
        "session_asia",
        "session_pre_london",
        "session_london_open",
        "session_london_cont",
        "session_ny_open",
        "session_ny_overlap_postdata",
    ]
    assert (out[helper_cols].sum(axis=1) == 1).all()
    xsi.validate_session_bucket_coverage(out)

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd


REQUIRED_OHLC_COLUMNS = ("timestamp", "open", "high", "low", "close")


@dataclass(frozen=True)
class SessionWindow:
    """Single session window in local time, start-inclusive and end-exclusive."""

    name: str
    start_hm: str
    end_hm: str


@dataclass(frozen=True)
class SessionConfig:
    """Config-driven session definition set."""

    tz: str
    windows: Sequence[SessionWindow]


def default_session_config(tz: str = "Europe/London") -> SessionConfig:
    """Return default multi-session buckets for XAU_USD research."""
    return SessionConfig(
        tz=tz,
        windows=(
            SessionWindow("asia", "00:00", "06:00"),
            SessionWindow("pre_london", "06:00", "08:00"),
            SessionWindow("london_open", "08:00", "09:00"),
            SessionWindow("london_continuation", "09:00", "13:00"),
            SessionWindow("ny_open", "13:00", "15:00"),
            SessionWindow("ny_overlap_postdata", "15:00", "24:00"),
        ),
    )


def _parse_hm_to_minute(hm: str) -> int:
    if hm == "24:00":
        return 24 * 60
    parts = hm.split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid time format '{hm}'. Expected HH:MM.")
    hh, mm = parts
    h = int(hh)
    m = int(mm)
    if h < 0 or h > 23 or m < 0 or m > 59:
        raise ValueError(f"Invalid time value '{hm}'.")
    return h * 60 + m


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out


def _validate_15m_cadence(index_utc: pd.DatetimeIndex) -> None:
    if len(index_utc) <= 1:
        return
    d = pd.Series(index_utc).diff().iloc[1:]
    bad = d != pd.Timedelta(minutes=15)
    if bool(bad.any()):
        bad_ix = np.where(bad.to_numpy())[0][:5]
        details = ", ".join(
            [
                f"{index_utc[i].isoformat()}->{index_utc[i + 1].isoformat()} ({d.iloc[i]})"
                for i in bad_ix
            ]
        )
        raise ValueError(f"Broken 15-minute cadence detected. Examples: {details}")


def load_ohlcv(path: str) -> pd.DataFrame:
    """Load and validate OHLCV-like bars from CSV/Parquet into UTC-indexed DataFrame.

    Validation rules:
    - required columns: timestamp/open/high/low/close
    - timestamp must be timezone-aware and unique
    - bars must be strictly monotonic increasing
    - bars must follow fixed 15-minute cadence
    """

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {p}")

    if p.suffix.lower() in {".parquet", ".pq"}:
        raw = pd.read_parquet(p)
    elif p.suffix.lower() in {".csv", ".txt"}:
        raw = pd.read_csv(p)
    else:
        raise ValueError(f"Unsupported file format '{p.suffix}'. Use CSV or Parquet.")

    df = _normalize_columns(raw)
    missing = [c for c in REQUIRED_OHLC_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Malformed input: missing required columns {missing}.")

    ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=False)
    if bool(ts.isna().any()):
        raise ValueError("Malformed input: unparseable timestamp values.")
    if ts.dt.tz is None:
        raise ValueError("Timestamp column must be timezone-aware (e.g., UTC offsets).")

    df = df.copy()
    df["timestamp"] = ts.dt.tz_convert("UTC")

    for c in ("open", "high", "low", "close"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
        if bool(df[c].isna().any()):
            raise ValueError(f"Malformed input: column '{c}' contains non-numeric values.")

    if bool(df["timestamp"].duplicated().any()):
        dup_count = int(df["timestamp"].duplicated().sum())
        raise ValueError(f"Duplicate timestamp bars detected: {dup_count} duplicates.")

    if not bool(df["timestamp"].is_monotonic_increasing):
        raise ValueError("Timestamp order must be monotonic increasing.")

    out = df.set_index("timestamp")
    out.index.name = "timestamp"

    _validate_15m_cadence(pd.DatetimeIndex(out.index))
    return out


def detect_session_config_overlaps(session_config: SessionConfig) -> List[str]:
    """Detect collisions/overlaps in a session config over a 24h minute map."""

    owners: List[List[str]] = [[] for _ in range(24 * 60)]
    for w in session_config.windows:
        start = _parse_hm_to_minute(w.start_hm)
        end = _parse_hm_to_minute(w.end_hm)
        if start == end:
            return [f"Session '{w.name}' has zero-length window."]
        if end > start:
            minute_range = range(start, end)
        else:
            minute_range = list(range(start, 24 * 60)) + list(range(0, end))
        for m in minute_range:
            owners[m].append(w.name)

    issues: List[str] = []
    overlap_minutes = [i for i, xs in enumerate(owners) if len(xs) > 1]
    uncovered_minutes = [i for i, xs in enumerate(owners) if len(xs) == 0]
    if overlap_minutes:
        sample = overlap_minutes[:5]
        issues.append(f"Overlapping windows at minutes {sample}.")
    if uncovered_minutes:
        sample = uncovered_minutes[:5]
        issues.append(f"Uncovered windows at minutes {sample}.")
    return issues


def validate_session_bucket_coverage(df: pd.DataFrame) -> None:
    """Ensure all rows have a valid session bucket assignment."""

    if "session_bucket" not in df.columns:
        raise ValueError("Missing 'session_bucket' column.")
    if bool(df["session_bucket"].isna().any()):
        raise ValueError("Session coverage check failed: unassigned rows found.")


def summarize_session_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Return deterministic per-session bar counts."""

    validate_session_bucket_coverage(df)
    s = df["session_bucket"].astype(str).value_counts().sort_index()
    out = s.rename_axis("session_bucket").reset_index(name="bar_count")
    return out


def assign_session_bucket(
    df: pd.DataFrame,
    session_config: SessionConfig,
    add_helper_columns: bool = True,
) -> pd.DataFrame:
    """Assign mutually-exclusive session bucket by local timestamp only (no future data)."""

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a DatetimeIndex.")
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is None:
        raise ValueError("DatetimeIndex must be timezone-aware.")

    issues = detect_session_config_overlaps(session_config)
    if issues:
        raise ValueError(f"Invalid session config: {'; '.join(issues)}")

    idx_local = idx.tz_convert(session_config.tz)
    minute = (idx_local.hour * 60) + idx_local.minute

    out = df.copy()
    bucket = pd.Series(pd.NA, index=out.index, dtype="object")

    for w in session_config.windows:
        start = _parse_hm_to_minute(w.start_hm)
        end = _parse_hm_to_minute(w.end_hm)
        if end > start:
            mask = (minute >= start) & (minute < end)
        else:
            mask = (minute >= start) | (minute < end)
        if bool((bucket.notna() & mask).any()):
            raise ValueError(f"Config collision detected while assigning session '{w.name}'.")
        bucket.loc[mask] = w.name

    out["session_bucket"] = pd.Categorical(bucket, categories=[w.name for w in session_config.windows], ordered=False)
    validate_session_bucket_coverage(out)

    if add_helper_columns:
        helper_names: Dict[str, str] = {
            "asia": "session_asia",
            "pre_london": "session_pre_london",
            "london_open": "session_london_open",
            "london_continuation": "session_london_cont",
            "ny_open": "session_ny_open",
            "ny_overlap_postdata": "session_ny_overlap_postdata",
        }
        for session_name, col_name in helper_names.items():
            out[col_name] = (out["session_bucket"].astype(str) == session_name).astype(int)

        helper_cols = list(helper_names.values())
        active_cnt = out[helper_cols].sum(axis=1)
        if not bool((active_cnt == 1).all()):
            raise ValueError("Session helper columns are not mutually exclusive.")

    return out

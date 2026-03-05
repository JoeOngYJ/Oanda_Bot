"""Data downloader helpers.

This module provides:
- OandaDownloader: a thin wrapper around oandapyV20 InstrumentsCandles endpoint
- download_candles: a small fallback stub returning an empty list (keeps
  compatibility with earlier skeleton code)

Notes:
- The OandaDownloader uses environment variables for credentials: OANDA_API_TOKEN
  and optionally OANDA_ACCOUNT_ID and OANDA_ENV ('practice' or 'live').
"""
from __future__ import annotations

from typing import Optional
import time
import concurrent.futures
import os
import datetime as dt
import pandas as pd

try:
    import oandapyV20
    from oandapyV20 import API
    from oandapyV20.endpoints.instruments import InstrumentsCandles
except Exception:  # pragma: no cover - may not be installed in minimal env
    oandapyV20 = None


def download_candles(symbol: str, timeframe: str, since=None, until=None):
    """Legacy stub kept for compatibility.

    Returns: empty list. Prefer using OandaDownloader or DataWarehouse.import_csv
    for real data.
    """
    return []


class OandaDownloader:
    """Wrapper to fetch historical candles from OANDA using oandapyV20.

    Example usage:
        dl = OandaDownloader({'token': os.environ['OANDA_API_TOKEN']})
        df = dl.download('EUR_USD', granularity='H1', start=..., end=...)
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self.token = cfg.get("token") or os.environ.get("OANDA_API_TOKEN")
        self.account_id = cfg.get("account_id") or os.environ.get("OANDA_ACCOUNT_ID")
        # Allow either OANDA_ENV or TRADING_ENVIRONMENT in .env files so users
        # who prefer a single variable name (TRADING_ENVIRONMENT) aren't forced
        # to set both. Default to 'practice'.
        env = (
            cfg.get("environment")
            or os.environ.get("OANDA_ENV")
            or os.environ.get("TRADING_ENVIRONMENT")
            or "practice"
        ).lower()

        if env not in ("practice", "live"):
            env = "practice"

        self.api = None
        # network/request settings
        self.request_timeout = cfg.get("request_timeout", 30)  # seconds per request
        self.max_retries = cfg.get("max_retries", 3)
        self.retry_backoff = cfg.get("retry_backoff", 2)  # seconds (exponential)

        if oandapyV20 is not None and self.token:
            # oandapyV20 expects an environment key ('practice' or 'live'),
            # not a full URL. Pass the env string so the library can map it
            # to the correct host internally.
            self.api = API(access_token=self.token, environment=env)

    @staticmethod
    def _to_utc_aware(value: Optional[dt.datetime]) -> Optional[dt.datetime]:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=dt.timezone.utc)
        return value.astimezone(dt.timezone.utc)

    @staticmethod
    def _parse_candle_time(candle_time: str) -> tuple[pd.Timestamp, dt.datetime]:
        ts_utc = pd.to_datetime(candle_time, utc=True)
        rec_time = ts_utc.tz_localize(None)
        return rec_time, ts_utc.to_pydatetime()

    @staticmethod
    def _price_requests_bid_ask(price: str) -> bool:
        p = (price or "").upper()
        return "B" in p and "A" in p

    def _parse_candle_record(self, candle: dict, rec_time: pd.Timestamp, write_bid_ask: bool) -> dict:
        bid = candle.get("bid") or {}
        ask = candle.get("ask") or {}
        mid = candle.get("mid") or {}

        def f(x):
            return float(x) if x is not None else None

        if mid:
            open_p, high_p, low_p, close_p = map(f, (mid.get("o"), mid.get("h"), mid.get("l"), mid.get("c")))
            return {
                "time": rec_time,
                "open": open_p,
                "high": high_p,
                "low": low_p,
                "close": close_p,
                "volume": int(candle.get("volume", 0)),
            }

        bo, bh, bl, bc = map(f, (bid.get("o"), bid.get("h"), bid.get("l"), bid.get("c")))
        ao, ah, al, ac = map(f, (ask.get("o"), ask.get("h"), ask.get("l"), ask.get("c")))

        if write_bid_ask and (bid and ask):
            return {
                "time": rec_time,
                "bid_o": bo,
                "bid_h": bh,
                "bid_l": bl,
                "bid_c": bc,
                "ask_o": ao,
                "ask_h": ah,
                "ask_l": al,
                "ask_c": ac,
                "mid_o": (bo + ao) / 2 if bo is not None and ao is not None else None,
                "mid_h": (bh + ah) / 2 if bh is not None and ah is not None else None,
                "mid_l": (bl + al) / 2 if bl is not None and al is not None else None,
                "mid_c": (bc + ac) / 2 if bc is not None and ac is not None else None,
                "spread_c": (ac - bc) if ac is not None and bc is not None else None,
                "volume": int(candle.get("volume", 0)),
            }

        return {
            "time": rec_time,
            "open": (bo + ao) / 2 if bo is not None and ao is not None else None,
            "high": (bh + ah) / 2 if bh is not None and ah is not None else None,
            "low": (bl + al) / 2 if bl is not None and al is not None else None,
            "close": (bc + ac) / 2 if bc is not None and ac is not None else None,
            "volume": int(candle.get("volume", 0)),
        }

    def download(
            self,
            instrument: str,
            granularity: str,
            start: Optional[dt.datetime] = None,
            end: Optional[dt.datetime] = None,
            count: Optional[int] = None,
            price: str = "M",                 # NEW
            store_bid_ask: bool = False       # NEW
        ) -> pd.DataFrame:
        """Download candles and return a pandas DataFrame indexed by UTC datetime.

        - instrument: e.g. 'EUR_USD'
        - granularity: OANDA string (e.g. 'H1', 'M15') or pandas freq
        - start, end: optional datetimes (UTC). If omitted and count is None,
          OANDA may return the most recent candles (behavior depends on endpoint).
        - count: optional number of candles to request
        """
        if self.api is None:
            raise RuntimeError("oandapyV20 not available or OANDA token not set")

        params = {"granularity": granularity}

        # Determine whether we're requesting a specific count of most-recent
        # candles or a time-range. OANDA does not allow 'count' together with
        # 'from'/'to'. If start/end are provided prefer the time-range and
        # omit 'count' (even if provided) to avoid API errors.
        using_time_range = start is not None or end is not None
        if not using_time_range and count is not None:
            params["count"] = count

        # Support requesting different price types: 'M' (mid), 'BA' (bid+ask)
        params["price"] = price
        write_bid_ask = store_bid_ask and self._price_requests_bid_ask(price)

        # OANDA limits responses (count) — page if necessary.
        MAX_PER_REQUEST = 5000
        MAX_ITERATIONS = 100  # Prevent infinite loops
        # remaining is used only when the user asked for a specific count and
        # we're paging through that many candles. If using a time range we
        # rely on advancing the 'from' parameter for pagination and do not
        # manage remaining by count.
        if (not using_time_range) and count is not None and count > MAX_PER_REQUEST:
            remaining = count
        else:
            remaining = None

        records = []
        iteration_count = 0

        # If requesting a time range, OANDA may reject ranges that imply more
        # candles than its maximum per-request. To avoid that we split the
        # requested time window into chunks that each request at most
        # MAX_PER_REQUEST candles.
        if using_time_range:
            # Map common granularities to seconds per candle (approximate for D/W/M)
            gran = granularity.upper()
            sec_map = {
                "S5": 5, "S10": 10, "S15": 15, "S30": 30,
                "M1": 60, "M2": 120, "M4": 240, "M5": 300, "M10": 600, "M15": 900, "M30": 1800,
                "H1": 3600, "H2": 7200, "H3": 10800, "H4": 14400, "H6": 21600, "H8": 28800, "H12": 43200,
                "D": 86400, "W": 86400 * 7, "M": 86400 * 30,
            }
            sec = sec_map.get(gran, 3600)

            # chunk length in seconds so each chunk is at most MAX_PER_REQUEST candles
            chunk_seconds = MAX_PER_REQUEST * sec

            current_from = self._to_utc_aware(start) if start is not None else dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)
            final_to = self._to_utc_aware(end) if end is not None else None

            print(f"DEBUG: Chunking time-range download: start={start}, end={end}, final_to={final_to}")

            while True:
                iteration_count += 1
                if iteration_count > MAX_ITERATIONS:
                    print(f"WARNING: Reached maximum iterations ({MAX_ITERATIONS}), stopping download")
                    print(f"Downloaded {len(records)} records so far")
                    break

                # compute sub-range
                sub_to = final_to if final_to is None else min(final_to, current_from + dt.timedelta(seconds=chunk_seconds - 1))
                params_local = params.copy()
                params_local["from"] = current_from.isoformat()
                if sub_to is not None:
                    params_local["to"] = sub_to.isoformat()

                req = InstrumentsCandles(instrument=instrument, params=params_local)
                # run API request with timeout and retries to avoid hanging
                print(f"Requesting {params_local.get('from')} -> {params_local.get('to')} (gran={granularity})")
                resp = None
                for attempt in range(1, self.max_retries + 1):
                    try:
                        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                            fut = ex.submit(self.api.request, req)
                            done, not_done = concurrent.futures.wait({fut}, timeout=self.request_timeout)
                            if not done:
                                fut.cancel()
                                raise concurrent.futures.TimeoutError()
                            resp = fut.result()
                        break
                    except concurrent.futures.TimeoutError:
                        print(f"Request timed out (attempt {attempt}/{self.max_retries}), backing off...")
                    except Exception as e:
                        print(f"Request error (attempt {attempt}/{self.max_retries}): {e}")
                    time.sleep(self.retry_backoff ** (attempt - 1))
                if resp is None:
                    raise RuntimeError("Failed to fetch data from OANDA after retries")

                candles = resp.get("candles", [])
                if not candles:
                    break

                # Track if we processed any complete candles in this iteration
                complete_candles_found = False
                iteration_last_time = None

                for c in candles:
                    if not c.get("complete", True):
                        continue

                    complete_candles_found = True
                    candle_time = c.get("time")
                    rec_time, iteration_last_time = self._parse_candle_time(candle_time)
                    rec = self._parse_candle_record(c, rec_time, write_bid_ask)
                    records.append(rec)

                # Only advance if we found complete candles, otherwise break to avoid infinite loop
                if not complete_candles_found:
                    print(f"No complete candles found in response, stopping pagination")
                    break

                last_time = iteration_last_time

                # Detect if we're stuck at the same position
                prev_current_from = current_from

                # advance current_from
                current_from = last_time + dt.timedelta(microseconds=1)

                # Check if we're stuck (not advancing)
                if current_from == prev_current_from:
                    # Instead of stopping immediately, force-advance by one
                    # candle's worth of time. This handles servers that may
                    # repeatedly return the same edge candle timestamp. The
                    # MAX_ITERATIONS guard above will still prevent infinite
                    # loops if this happens repeatedly.
                    print(f"WARNING: Not advancing (stuck at {current_from}), forcing advance by one candle ({sec} sec)")
                    current_from = current_from + dt.timedelta(seconds=sec)

                if final_to is not None and current_from > final_to:
                    break

                # do not stop solely because fewer than MAX_PER_REQUEST were
                # returned for this sub-range; continue advancing current_from
                # until we've exceeded the requested final_to. Some sub-ranges
                # (especially near the end of available data) may return fewer
                # candles but we should still attempt the next sub-range until
                # the overall final_to is reached.

        else:
            # Non-time-range (count-based or recent candles) handling
            last_time = None

            while True:
                if remaining is not None:
                    req_count = min(remaining, MAX_PER_REQUEST)
                    params["count"] = req_count
                else:
                    params["count"] = MAX_PER_REQUEST

                if last_time is not None:
                    params["from"] = (last_time + dt.timedelta(microseconds=1)).isoformat()

                print(f"Requesting recent/count-based chunk (params: { {k: params[k] for k in ('count','from') if k in params} })")
                resp = None
                req = InstrumentsCandles(instrument=instrument, params=params)
                for attempt in range(1, self.max_retries + 1):
                    try:
                        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                            fut = ex.submit(self.api.request, req)
                            done, not_done = concurrent.futures.wait({fut}, timeout=self.request_timeout)
                            if not done:
                                fut.cancel()
                                raise concurrent.futures.TimeoutError()
                            resp = fut.result()
                        break
                    except concurrent.futures.TimeoutError:
                        print(f"Request timed out (attempt {attempt}/{self.max_retries}), backing off...")
                    except Exception as e:
                        print(f"Request error (attempt {attempt}/{self.max_retries}): {e}")
                    time.sleep(self.retry_backoff ** (attempt - 1))
                if resp is None:
                    raise RuntimeError("Failed to fetch data from OANDA after retries")

                candles = resp.get("candles", [])
                if not candles:
                    break

                for c in candles:
                    if not c.get("complete", True):
                        continue

                    candle_time = c.get("time")
                    # prefer mid, but if price 'BA' requested, look for 'bid'/'ask'
                    rec_time, _ = self._parse_candle_time(candle_time)
                    rec = self._parse_candle_record(c, rec_time, write_bid_ask)
                    records.append(rec)

                # update pagination state
                last_time = pd.to_datetime(candles[-1].get("time"), utc=True).to_pydatetime()

                # decrement remaining if we were given a count
                if remaining is not None:
                    remaining -= len(candles)
                    if remaining <= 0:
                        break

                # if fewer than the max were returned, we're done
                if len(candles) < MAX_PER_REQUEST:
                    break

        if not records:
            if write_bid_ask:
                cols = ["bid_o", "bid_h", "bid_l", "bid_c", "ask_o", "ask_h", "ask_l", "ask_c", "mid_o", "mid_h", "mid_l", "mid_c", "spread_c", "volume"]
            else:
                cols = ["open", "high", "low", "close", "volume"]
            return pd.DataFrame(columns=cols).set_index(pd.DatetimeIndex([]))

        df = pd.DataFrame.from_records(records).set_index("time")
        df.index.name = "datetime"
        df = df[~df.index.duplicated(keep="first")]
        # ensure numeric dtypes for columns that exist
        float_cols = [
            "open", "high", "low", "close",
            "bid_o", "bid_h", "bid_l", "bid_c",
            "ask_o", "ask_h", "ask_l", "ask_c",
            "mid_o", "mid_h", "mid_l", "mid_c",
            "spread_c",
        ]
        cast_map = {col: float for col in float_cols if col in df.columns}
        if "volume" in df.columns:
            cast_map["volume"] = int
        if cast_map:
            df = df.astype(cast_map)
        return df

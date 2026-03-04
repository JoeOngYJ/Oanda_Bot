"""Small CLI entrypoint to avoid name collisions with system 'scripts' packages.

This module provides a single `main()` function used by the console script
`oanda-download` to run the download routine. We implement a thin wrapper
here rather than importing the top-level `scripts` package which may collide
with system packages (e.g., ROS installs a top-level `scripts`).
"""
from __future__ import annotations

import argparse
import os
import datetime as dt
from pathlib import Path
import tomllib

from oanda_bot.backtesting.data.manager import DataManager
from oanda_bot.backtesting.core.timeframe import Timeframe


def parse_args():
    p = argparse.ArgumentParser(description="Download or import market data into the DataWarehouse")
    # Use None defaults so a provided --config can supply values and CLI flags
    # still act as explicit overrides.
    p.add_argument("--config", help="Path to a TOML configuration file")
    p.add_argument("--instrument", default=None)
    p.add_argument("--tf", default=None, help="Timeframe (OANDA granularity, e.g. H1, M15)")
    p.add_argument("--start", default=None, help="Start datetime ISO (UTC)")
    p.add_argument("--end", default=None, help="End datetime ISO (UTC)")
    p.add_argument("--csv", help="Path to local CSV to import instead of OANDA")
    p.add_argument("--count", type=int, help="Number of candles to request (optional)")
    p.add_argument("--price", choices=["M", "BA"], default="M", help="Price type: M (mid) or BA (bid+ask average)")
    return p.parse_args()


def main():
    args = parse_args()

    # Load config from TOML file if provided. Command-line flags override config.
    file_config = {}
    if args.config:
        cfg_path = Path(args.config)
        if not cfg_path.exists():
            raise SystemExit(f"Config file not found: {cfg_path}")
        with cfg_path.open("rb") as fh:
            file_config = tomllib.load(fh)

    # Build runtime config preferring: CLI args -> config file -> environment -> defaults
    data_dir = os.environ.get("DATA_DIR") or file_config.get("data_dir") or "data/backtesting"
    oanda_cfg = file_config.get("oanda", {})
    oanda_cfg = {
        "token": os.environ.get("OANDA_API_TOKEN") or oanda_cfg.get("token"),
        "account_id": os.environ.get("OANDA_ACCOUNT_ID") or oanda_cfg.get("account_id"),
        "environment": os.environ.get("OANDA_ENV") or oanda_cfg.get("environment") or "practice",
    }

    cfg = {"data_dir": data_dir, "oanda": oanda_cfg}

    dm = DataManager(cfg)

    # Resolve instrument / timeframe / start / end with precedence CLI -> config -> env -> defaults
    instrument = args.instrument or file_config.get("instrument") or os.environ.get("INSTRUMENT") or "EUR_USD"
    tf_str = args.tf or file_config.get("tf") or os.environ.get("TF") or "H1"
    try:
        base_tf = Timeframe.from_oanda_granularity(tf_str)
    except Exception:
        base_tf = getattr(Timeframe, tf_str, Timeframe.H1)

    end = None
    if args.end is not None:
        end = dt.datetime.fromisoformat(args.end)
    elif file_config.get("end"):
        end = dt.datetime.fromisoformat(file_config.get("end"))
    else:
        end = dt.datetime.utcnow()

    start = None
    if args.start is not None:
        start = dt.datetime.fromisoformat(args.start)
    elif file_config.get("start"):
        start = dt.datetime.fromisoformat(file_config.get("start"))

    if args.csv:
        csv_path = Path(args.csv)
        if not csv_path.exists():
            raise SystemExit(f"CSV file not found: {csv_path}")
        print(f"Importing CSV {csv_path} into warehouse for {instrument} {base_tf.name}")
        dm.warehouse.import_csv(csv_path, instrument, base_tf)
        dfs = dm.ensure_data(instrument, base_tf, start, end, [base_tf])
        print("Imported and resampled:", [tf.name for tf in dfs.keys()])
        return

    token = cfg["oanda"].get("token")
    if not token:
        raise RuntimeError("No OANDA token set in OANDA_API_TOKEN and no --csv provided")

    print(f"DEBUG CLI: Downloading {instrument} {base_tf.name} from {start} to {end}")
    print(f"Downloading {instrument} {base_tf.name} from OANDA from {start or 'most-recent'} to {end}")
    dfs = dm.ensure_data(instrument, base_tf, start, end, [base_tf])
    print("Downloaded and saved timeframes:", [tf.name for tf in dfs.keys()])


if __name__ == "__main__":
    main()

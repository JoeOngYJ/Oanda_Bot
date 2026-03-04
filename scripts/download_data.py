"""Small helper to download data using the DataManager and OandaDownloader.

Behavior:
- If OANDA_API_TOKEN is present in the environment, attempt to fetch candles from OANDA.
- Otherwise, if CSV_PATH is provided, import the CSV into the DataWarehouse.

Configure with environment variables:
- OANDA_API_TOKEN (optional)
- OANDA_ACCOUNT_ID (optional)
- OANDA_ENV ('practice' or 'live') optional
- CSV_PATH (optional) - path to local CSV to import instead of calling OANDA
"""
import argparse
import os
import datetime as dt
from pathlib import Path

from oanda_bot.backtesting.data.manager import DataManager
from oanda_bot.backtesting.core.timeframe import Timeframe


def parse_args():
    p = argparse.ArgumentParser(description="Download or import market data into the DataWarehouse")
    p.add_argument("--instrument", default=os.environ.get("INSTRUMENT", "EUR_USD"))
    p.add_argument("--tf", default=os.environ.get("TF", "H1"), help="Timeframe (OANDA granularity, e.g. H1, M15)")
    p.add_argument("--start", help="Start datetime ISO (UTC)")
    p.add_argument("--end", help="End datetime ISO (UTC)")
    p.add_argument("--csv", help="Path to local CSV to import instead of OANDA")
    p.add_argument("--count", type=int, help="Number of candles to request (optional)")
    p.add_argument("--price", choices=["M", "BA"], default="M", help="Price type: M (mid) or BA (bid+ask average)")
    return p.parse_args()


def main():
    args = parse_args()

    cfg = {
        "data_dir": os.environ.get("DATA_DIR", "data/backtesting"),
        "oanda": {
            "token": os.environ.get("OANDA_API_TOKEN"),
            "account_id": os.environ.get("OANDA_ACCOUNT_ID"),
            "environment": os.environ.get("OANDA_ENV", "practice"),
        },
    }

    dm = DataManager(cfg)

    instrument = args.instrument
    try:
        base_tf = Timeframe.from_oanda_granularity(args.tf)
    except Exception:
        # fallback: try to match by name
        base_tf = getattr(Timeframe, args.tf, Timeframe.H1)

    end = dt.datetime.utcnow() if args.end is None else dt.datetime.fromisoformat(args.end)
    start = None if args.start is None else dt.datetime.fromisoformat(args.start)

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

    print(f"Downloading {instrument} {base_tf.name} from OANDA from {start or 'most-recent'} to {end}")
    # DataManager will call OandaDownloader; pass price/count via env if needed
    # For now DataManager delegates to downloader; if you need to pass params like price
    # or count directly, extend DataManager.download wrappers.
    dfs = dm.ensure_data(instrument, base_tf, start, end, [base_tf])
    print("Downloaded and saved timeframes:", [tf.name for tf in dfs.keys()])


if __name__ == "__main__":
    main()

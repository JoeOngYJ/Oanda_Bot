#!/usr/bin/env python3
"""
CSV Import Helper Script

This script helps you import historical market data from CSV files into the
backtesting data warehouse. It's a simpler alternative to downloading from OANDA.

Usage:
    python scripts/import_csv.py --csv path/to/data.csv --instrument EUR_USD --timeframe H1

CSV Format Expected:
    The CSV should have columns: timestamp, open, high, low, close, volume
    Timestamp should be in ISO format (e.g., 2024-01-01T00:00:00)

Example CSV:
    timestamp,open,high,low,close,volume
    2024-01-01T00:00:00,1.0850,1.0875,1.0840,1.0860,1000
    2024-01-01T01:00:00,1.0860,1.0880,1.0850,1.0870,1200
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

# Add parent directory to path so we can import oanda_bot.backtesting modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from oanda_bot.backtesting.data.warehouse import DataWarehouse
from oanda_bot.backtesting.core.timeframe import Timeframe


def validate_csv(csv_path: Path) -> bool:
    """Validate that the CSV has the required columns."""
    try:
        df = pd.read_csv(csv_path, nrows=5)
        required_cols = {'timestamp', 'open', 'high', 'low', 'close', 'volume'}

        # Check for alternative column names
        actual_cols = set(df.columns.str.lower())

        # Map alternative names
        col_mapping = {
            'time': 'timestamp',
            'datetime': 'timestamp',
            'date': 'timestamp',
            'vol': 'volume',
        }

        # Normalize column names
        for alt, standard in col_mapping.items():
            if alt in actual_cols:
                actual_cols.add(standard)

        missing = required_cols - actual_cols
        if missing:
            print(f"ERROR: CSV is missing required columns: {missing}")
            print(f"Found columns: {list(df.columns)}")
            print(f"\nRequired columns: {required_cols}")
            return False

        print(f"✓ CSV validation passed")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Rows: {len(pd.read_csv(csv_path))}")
        return True

    except Exception as e:
        print(f"ERROR: Failed to read CSV: {e}")
        return False


def import_csv_data(csv_path: Path, instrument: str, timeframe: Timeframe, data_dir: Path):
    """Import CSV data into the warehouse."""
    print(f"\nImporting data...")
    print(f"  CSV: {csv_path}")
    print(f"  Instrument: {instrument}")
    print(f"  Timeframe: {timeframe.name}")
    print(f"  Data directory: {data_dir}")

    warehouse = DataWarehouse(data_dir)

    try:
        # Read CSV
        df = pd.read_csv(csv_path)

        # Normalize column names (handle alternatives)
        col_mapping = {
            'time': 'timestamp',
            'datetime': 'timestamp',
            'date': 'timestamp',
            'vol': 'volume',
        }
        df.columns = df.columns.str.lower()
        df = df.rename(columns=col_mapping)

        # Convert timestamp to datetime index
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        df.index.name = 'datetime'

        # Ensure correct column order and types
        df = df[['open', 'high', 'low', 'close', 'volume']]
        df = df.astype({
            'open': float,
            'high': float,
            'low': float,
            'close': float,
            'volume': int
        })

        # Remove duplicates
        df = df[~df.index.duplicated(keep='first')]

        # Sort by timestamp
        df = df.sort_index()

        print(f"\n✓ Processed {len(df)} bars")
        print(f"  Date range: {df.index.min()} to {df.index.max()}")

        # Save to warehouse
        warehouse.save(df, instrument, timeframe)

        print(f"\n✓ Successfully imported data to warehouse")
        print(f"  Saved to: {data_dir / instrument / f'{timeframe.name}.csv'}")

        return True

    except Exception as e:
        print(f"\nERROR: Failed to import data: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Import CSV data into backtesting warehouse",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument("--csv", required=True, help="Path to CSV file")
    parser.add_argument("--instrument", required=True, help="Instrument symbol (e.g., EUR_USD, XAU_USD)")
    parser.add_argument("--timeframe", required=True, help="Timeframe (e.g., M1, M15, M30, H1, H4, D1)")
    parser.add_argument("--data-dir", default="data/backtesting", help="Data directory (default: data/backtesting)")
    parser.add_argument("--validate-only", action="store_true", help="Only validate CSV, don't import")

    args = parser.parse_args()

    # Validate paths
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: CSV file not found: {csv_path}")
        return 1

    data_dir = Path(args.data_dir)

    # Parse timeframe
    try:
        timeframe = Timeframe.from_oanda_granularity(args.timeframe)
    except ValueError:
        try:
            timeframe = Timeframe.from_pandas_freq(args.timeframe)
        except ValueError:
            print(f"ERROR: Invalid timeframe: {args.timeframe}")
            print(f"Valid timeframes: M1, M15, M30, H1, H4, D1")
            return 1

    # Validate CSV
    print("Validating CSV file...")
    if not validate_csv(csv_path):
        return 1

    if args.validate_only:
        print("\n✓ Validation complete (--validate-only specified, not importing)")
        return 0

    # Import data
    if import_csv_data(csv_path, args.instrument, timeframe, data_dir):
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())

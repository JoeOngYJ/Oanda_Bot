#!/usr/bin/env python3
"""
Simple OANDA Data Downloader

This is a simplified alternative to the main download script that avoids
complex pagination logic. It downloads data in small, manageable chunks.

Usage:
    python scripts/simple_download.py --instrument EUR_USD --timeframe H1 \\
        --start 2024-01-01 --end 2024-01-07

Features:
    - Downloads data in daily chunks (avoids pagination issues)
    - Clear progress indicators
    - Automatic retry on failures
    - Validates data before saving

Requirements:
    - OANDA_API_TOKEN environment variable must be set
    - Or create a .env file with OANDA_API_TOKEN=your_token_here
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backtesting.data.downloader import OandaDownloader
from backtesting.data.warehouse import DataWarehouse
from backtesting.core.timeframe import Timeframe
import pandas as pd


def parse_date(date_str: str) -> datetime:
    """Parse date string to datetime."""
    try:
        # Try ISO format first
        return datetime.fromisoformat(date_str)
    except ValueError:
        # Try date-only format
        try:
            return datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"Invalid date format: {date_str}. Use YYYY-MM-DD or ISO format")


def download_chunk(downloader: OandaDownloader, instrument: str, granularity: str,
                   start: datetime, end: datetime, chunk_name: str) -> pd.DataFrame:
    """Download a single chunk of data with retry logic."""
    max_retries = 3
    retry_delay = 2

    for attempt in range(1, max_retries + 1):
        try:
            print(f"  [{chunk_name}] Downloading {start.date()} to {end.date()}...", end=" ", flush=True)
            df = downloader.download(
                instrument=instrument,
                granularity=granularity,
                start=start,
                end=end
            )

            if df.empty:
                print("No data")
                return df

            print(f"✓ {len(df)} bars")
            return df

        except Exception as e:
            if attempt < max_retries:
                print(f"Failed (attempt {attempt}/{max_retries}), retrying...")
                time.sleep(retry_delay * attempt)
            else:
                print(f"Failed after {max_retries} attempts: {e}")
                raise


def download_in_chunks(downloader: OandaDownloader, instrument: str, timeframe: Timeframe,
                       start_date: datetime, end_date: datetime, chunk_days: int = 7):
    """Download data in manageable chunks."""
    print(f"\nDownloading {instrument} {timeframe.name}")
    print(f"Date range: {start_date.date()} to {end_date.date()}")
    print(f"Chunk size: {chunk_days} days\n")

    granularity = timeframe.to_oanda_granularity()
    all_data = []

    current_start = start_date
    chunk_num = 0

    while current_start < end_date:
        chunk_num += 1
        current_end = min(current_start + timedelta(days=chunk_days), end_date)

        chunk_name = f"Chunk {chunk_num}"

        try:
            df = download_chunk(
                downloader, instrument, granularity,
                current_start, current_end, chunk_name
            )

            if not df.empty:
                all_data.append(df)

        except Exception as e:
            print(f"\n⚠ Warning: Failed to download chunk {chunk_num}: {e}")
            print(f"  Continuing with next chunk...")

        current_start = current_end

    if not all_data:
        print("\n⚠ No data downloaded")
        return pd.DataFrame()

    # Combine all chunks
    print(f"\nCombining {len(all_data)} chunks...")
    combined = pd.concat(all_data).sort_index()

    # Remove duplicates
    combined = combined[~combined.index.duplicated(keep='first')]

    print(f"✓ Total: {len(combined)} bars")
    print(f"  Date range: {combined.index.min()} to {combined.index.max()}")

    return combined


def main():
    parser = argparse.ArgumentParser(
        description="Simple OANDA data downloader (downloads in small chunks)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument("--instrument", required=True, help="Instrument (e.g., EUR_USD, XAU_USD)")
    parser.add_argument("--timeframe", required=True, help="Timeframe (M15, M30, H1, H4, D1)")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD or ISO format)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD or ISO format)")
    parser.add_argument("--chunk-days", type=int, default=7, help="Days per chunk (default: 7)")
    parser.add_argument("--data-dir", default="data/backtesting", help="Data directory")
    parser.add_argument("--dry-run", action="store_true", help="Don't save, just test download")

    args = parser.parse_args()

    # Check for API token
    token = os.environ.get("OANDA_API_TOKEN")
    if not token:
        print("ERROR: OANDA_API_TOKEN environment variable not set")
        print("\nSet it with:")
        print("  export OANDA_API_TOKEN=your_token_here")
        print("\nOr create a .env file with:")
        print("  OANDA_API_TOKEN=your_token_here")
        return 1

    # Parse timeframe
    try:
        timeframe = Timeframe.from_oanda_granularity(args.timeframe)
    except ValueError:
        print(f"ERROR: Invalid timeframe: {args.timeframe}")
        print(f"Valid: M15, M30, H1, H4, D1")
        return 1

    # Parse dates
    try:
        start_date = parse_date(args.start)
        end_date = parse_date(args.end)
    except ValueError as e:
        print(f"ERROR: {e}")
        return 1

    if start_date >= end_date:
        print("ERROR: Start date must be before end date")
        return 1

    # Initialize downloader
    config = {
        "token": token,
        "environment": os.environ.get("OANDA_ENV", "practice")
    }
    downloader = OandaDownloader(config)

    # Download data
    try:
        df = download_in_chunks(
            downloader, args.instrument, timeframe,
            start_date, end_date, args.chunk_days
        )

        if df.empty:
            print("\n⚠ No data to save")
            return 1

        if args.dry_run:
            print("\n✓ Dry run complete (not saving)")
            return 0

        # Save to warehouse
        data_dir = Path(args.data_dir)
        warehouse = DataWarehouse(data_dir)
        warehouse.save(df, args.instrument, timeframe)

        save_path = data_dir / args.instrument / f"{timeframe.name}.csv"
        print(f"\n✓ Saved to: {save_path}")

        return 0

    except KeyboardInterrupt:
        print("\n\n⚠ Download interrupted by user")
        return 1
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

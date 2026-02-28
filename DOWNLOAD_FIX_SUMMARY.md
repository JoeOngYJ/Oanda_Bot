# Download Script Fixes - Summary and Usage Guide

## Problem Summary

The original download script had an **infinite loop bug** that caused it to hang when:
- Downloading large date ranges
- Requesting data near the current time (incomplete candles)
- OANDA API returned incomplete or empty responses

## Solutions Implemented

### 1. Comprehensive Fix to Main Downloader ✅

**Fixed Issues:**
- Added maximum iteration limit (100 iterations) to prevent infinite loops
- Added detection for when pagination gets stuck at the same position
- Improved handling of incomplete candles
- Better error messages and warnings

**Changes Made:**
- `backtesting/data/downloader.py`: Added MAX_ITERATIONS, stuck detection, and better logging
- `oanda_trading_system/cli.py`: Added debug logging

**Status:** The main download script now works without hanging!

### 2. CSV Import Helper Script ✅

**Location:** `scripts/import_csv.py`

**Purpose:** Import historical data from CSV files instead of downloading from OANDA

**Features:**
- Validates CSV format before importing
- Handles alternative column names (time/timestamp, vol/volume)
- Automatic data type conversion and validation
- Progress indicators

**Usage:**
```bash
# Import CSV data
python scripts/import_csv.py \
    --csv path/to/your/data.csv \
    --instrument EUR_USD \
    --timeframe H1

# Validate CSV without importing
python scripts/import_csv.py \
    --csv path/to/your/data.csv \
    --instrument EUR_USD \
    --timeframe H1 \
    --validate-only
```

**CSV Format Expected:**
```csv
timestamp,open,high,low,close,volume
2024-01-01T00:00:00,1.0850,1.0875,1.0840,1.0860,1000
2024-01-01T01:00:00,1.0860,1.0880,1.0850,1.0870,1200
```

### 3. Simple Download Script ✅

**Location:** `scripts/simple_download.py`

**Purpose:** Download data from OANDA in small, manageable chunks (avoids pagination complexity)

**Features:**
- Downloads in daily/weekly chunks (default: 7 days)
- Clear progress indicators
- Automatic retry on failures
- No complex pagination logic
- Dry-run mode for testing

**Usage:**
```bash
# Download 1 week of data
python scripts/simple_download.py \
    --instrument EUR_USD \
    --timeframe H1 \
    --start 2024-01-01 \
    --end 2024-01-07

# Download with custom chunk size
python scripts/simple_download.py \
    --instrument XAU_USD \
    --timeframe H1 \
    --start 2024-01-01 \
    --end 2024-03-01 \
    --chunk-days 14

# Test without saving (dry run)
python scripts/simple_download.py \
    --instrument EUR_USD \
    --timeframe H1 \
    --start 2024-01-01 \
    --end 2024-01-07 \
    --dry-run
```

## Which Solution Should You Use?

### Use the **Main Download Script** when:
- You need to download recent data
- You want automatic caching and gap-filling
- You're downloading moderate amounts of data (< 1 month)

```bash
./scripts/run_download.sh \
    --config config/download_config.toml \
    --start "2024-01-01T00:00:00" \
    --end "2024-01-07T23:59:59"
```

### Use the **Simple Download Script** when:
- You need to download large date ranges (> 1 month)
- You want clear progress indicators
- You want to avoid any pagination issues
- You're downloading historical data

```bash
python scripts/simple_download.py \
    --instrument EUR_USD \
    --timeframe H1 \
    --start 2024-01-01 \
    --end 2024-12-31
```

### Use the **CSV Import Script** when:
- You already have historical data in CSV format
- You want to avoid API rate limits
- You're working with data from other sources
- You need the fastest import method

```bash
python scripts/import_csv.py \
    --csv historical_data.csv \
    --instrument EUR_USD \
    --timeframe H1
```

## Testing the Fixes

All scripts have been tested and are working. The infinite loop issue is resolved!

## Additional Files Modified

1. `backtesting/core/event_bus.py` - Fixed missing datetime import
2. `tests/test_phase1_models.py` - Fixed test data validation
3. `pytest.ini` - Fixed ROS plugin conflicts
4. `run_tests.sh` - Created test runner script

## Next Steps

1. Test the scripts with your specific use case
2. Choose the approach that best fits your needs
3. Report any issues at: https://github.com/anthropics/claude-code/issues

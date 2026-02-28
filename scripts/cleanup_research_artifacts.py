#!/usr/bin/env python3
"""Archive and prune intermediate research artifacts to save disk space."""

from __future__ import annotations

import argparse
import datetime as dt
import tarfile
from pathlib import Path
from typing import List


INTERMEDIATE_PATTERNS = [
    "regime_research_*_bar_regimes.csv",
    "multiframe_regime_labels_*.csv",
    "universe_research_*_windows.csv",
    "universe_research_*_correlation.csv",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Archive and optionally delete intermediate research files.")
    p.add_argument("--output-dir", default="data/research")
    p.add_argument("--archive-dir", default="data/research/archive")
    p.add_argument("--older-than-days", type=int, default=3)
    p.add_argument("--delete-after-archive", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _collect(output_dir: Path, older_than_days: int) -> List[Path]:
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=older_than_days)
    picked: List[Path] = []
    for pattern in INTERMEDIATE_PATTERNS:
        for path in output_dir.glob(pattern):
            mtime = dt.datetime.fromtimestamp(path.stat().st_mtime, tz=dt.timezone.utc)
            if mtime <= cutoff and path.is_file():
                picked.append(path)
    return sorted(set(picked))


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    archive_dir = Path(args.archive_dir)
    archive_dir.mkdir(parents=True, exist_ok=True)

    files = _collect(output_dir, args.older_than_days)
    if not files:
        print("No intermediate files matched retention policy.")
        return 0

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    archive_path = archive_dir / f"research_intermediate_{stamp}.tar.gz"

    print(f"Matched {len(files)} files for archive:")
    for p in files:
        print(f" - {p}")

    if args.dry_run:
        print("Dry run enabled; no archive written.")
        return 0

    with tarfile.open(archive_path, mode="w:gz") as tar:
        for path in files:
            tar.add(path, arcname=path.relative_to(output_dir.parent))

    print(f"Archive written: {archive_path}")

    if args.delete_after_archive:
        for path in files:
            path.unlink(missing_ok=True)
        print("Deleted original files after archiving.")
    else:
        print("Original files preserved (use --delete-after-archive to prune).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

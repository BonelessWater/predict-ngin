#!/usr/bin/env python3
"""
Merge CLOB shards into main directory and import to database.

Usage:
    python merge_clob_shards.py [num_shards]
"""

import sys
import shutil
from pathlib import Path
from src.trading.data_modules.database import PredictionMarketDB


def merge_shards(num_shards: int = 4):
    """Merge shard directories into main polymarket_clob directory."""

    output_dir = Path("data/polymarket_clob")
    output_dir.mkdir(parents=True, exist_ok=True)

    total_files = 0

    for shard_id in range(num_shards):
        shard_dir = Path(f"data/polymarket_clob_shard{shard_id}")

        if not shard_dir.exists():
            print(f"Shard {shard_id} not found: {shard_dir}")
            continue

        files = list(shard_dir.glob("market_*.json"))
        print(f"Shard {shard_id}: {len(files)} files")

        for f in files:
            dest = output_dir / f.name
            shutil.copy2(f, dest)
            total_files += 1

    print(f"\nMerged {total_files} files to {output_dir}")

    # Clean up shard directories
    cleanup = input("Delete shard directories? [y/N]: ").strip().lower()
    if cleanup == 'y':
        for shard_id in range(num_shards):
            shard_dir = Path(f"data/polymarket_clob_shard{shard_id}")
            if shard_dir.exists():
                shutil.rmtree(shard_dir)
                print(f"Deleted {shard_dir}")

    # Import to database
    print("\nImporting to database...")
    db = PredictionMarketDB()
    db.import_polymarket_clob()
    db.close()

    print("\nDone!")


if __name__ == "__main__":
    num_shards = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    merge_shards(num_shards)

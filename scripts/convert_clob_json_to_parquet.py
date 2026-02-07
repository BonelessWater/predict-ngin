#!/usr/bin/env python3
"""
Convert Polymarket CLOB JSON files to parquet price files.

Writes part files named prices_YYYY-MM_partNNNNN.parquet into output dir.
PriceStore will pick them up via glob.

Usage:
  python scripts/convert_clob_json_to_parquet.py --input-dir data/polymarket_clob --output-dir data/parquet/prices
  python scripts/convert_clob_json_to_parquet.py --input-dir data/polymarket_clob_shard0 --output-dir data/parquet/prices_shard0
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pyarrow as pa
import pyarrow.parquet as pq


def month_from_ts(ts: int) -> str:
    return datetime.utcfromtimestamp(ts).strftime("%Y-%m")


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert CLOB JSON to parquet.")
    parser.add_argument("--input-dir", default="data/polymarket_clob", help="Input directory with market_*.json")
    parser.add_argument("--output-dir", default="data/parquet/prices", help="Output parquet directory")
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=500_000,
        help="Flush buffer after this many rows per month",
    )
    parser.add_argument(
        "--pattern",
        default="market_*.json",
        help="Glob pattern for market files",
    )
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob(args.pattern))
    if not files:
        print(f"No files found in {in_dir} with pattern {args.pattern}")
        return 1

    buffers: Dict[str, Dict[str, List]] = {}
    part_counters = defaultdict(int)

    def flush_month(month: str) -> None:
        buf = buffers.get(month)
        if not buf or not buf["market_id"]:
            return
        table = pa.table(buf)
        part_idx = part_counters[month]
        part_counters[month] += 1
        filepath = out_dir / f"prices_{month}_part{part_idx:05d}.parquet"
        pq.write_table(table, filepath, compression="snappy")
        buffers[month] = {
            "market_id": [],
            "outcome": [],
            "timestamp": [],
            "price": [],
        }

    def ensure_buffer(month: str) -> Dict[str, List]:
        if month not in buffers:
            buffers[month] = {
                "market_id": [],
                "outcome": [],
                "timestamp": [],
                "price": [],
            }
        return buffers[month]

    total_rows = 0
    for i, filepath in enumerate(files, start=1):
        try:
            with filepath.open(encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        market_info = data.get("market_info", {})
        market_id = str(market_info.get("id", ""))
        if not market_id:
            continue

        price_history = data.get("price_history", {})
        for outcome, hist_data in price_history.items():
            history = hist_data.get("history", [])
            if not history:
                continue
            for point in history:
                ts = point.get("t")
                price = point.get("p")
                if ts is None or price is None:
                    continue
                try:
                    ts_int = int(ts)
                    price_f = float(price)
                except Exception:
                    continue
                month = month_from_ts(ts_int)
                buf = ensure_buffer(month)
                buf["market_id"].append(market_id)
                buf["outcome"].append(outcome)
                buf["timestamp"].append(ts_int)
                buf["price"].append(price_f)
                total_rows += 1
                if len(buf["market_id"]) >= args.chunk_rows:
                    flush_month(month)

        if i % 1000 == 0:
            print(f"Processed {i}/{len(files)} files, rows={total_rows:,}")

    # Flush remaining buffers
    for month in list(buffers.keys()):
        flush_month(month)

    print(f"Done. Total rows written: {total_rows:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

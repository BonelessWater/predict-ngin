#!/usr/bin/env python3
"""
Process order-filled parquet into processed trades parquet.

Wrapper around src/trading/data_modules.fetcher.DataFetcher so all logic lives
in the data modules.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.trading.data_modules.fetcher import DataFetcher


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert order-filled events to trades parquet.")
    parser.add_argument(
        "--order-filled-dir",
        default="data/parquet/order_filled_events",
        help="Input order-filled parquet directory",
    )
    parser.add_argument(
        "--output-dir",
        default="data/parquet/trades",
        help="Output trades parquet directory",
    )
    parser.add_argument(
        "--markets-dir",
        default="data/parquet/markets",
        help="Markets parquet directory (for token->market mapping)",
    )
    parser.add_argument("--state-file", default="", help="Optional state file path")
    parser.add_argument("--reset", action="store_true", help="Ignore existing state and start fresh")
    parser.add_argument("--max-files", type=int, default=0, help="Max files to process (0 = no limit)")
    parser.add_argument(
        "--fetch-missing-tokens",
        action="store_true",
        help="Fetch missing token mappings from Gamma API",
    )
    args = parser.parse_args()

    fetcher = DataFetcher()
    result = fetcher.process_polymarket_trades_from_order_filled(
        order_filled_dir=args.order_filled_dir,
        output_dir=args.output_dir,
        markets_dir=args.markets_dir,
        state_path=args.state_file or None,
        reset=args.reset,
        max_files=args.max_files,
        fetch_missing_tokens=args.fetch_missing_tokens,
    )

    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

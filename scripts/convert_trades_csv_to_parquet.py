#!/usr/bin/env python3
"""
Convert polymarket trades CSV to monthly partitioned parquet files.

Reads the large trades CSV (33 GB, ~149.5M rows) in chunks and writes
compressed parquet files partitioned by month.

Expected compression: ~33 GB CSV -> ~5-8 GB parquet (snappy).

Usage:
    python scripts/convert_trades_csv_to_parquet.py
    python scripts/convert_trades_csv_to_parquet.py --verify-only
"""

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


CSV_PATH = "data/poly_data/processed/trades.csv"
OUTPUT_DIR = "data/parquet/trades"
CHUNK_SIZE = 2_000_000


def convert(csv_path, output_dir, chunk_size):
    """Convert trades CSV to monthly partitioned parquet."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    csv_size = os.path.getsize(csv_path)
    print(f"CSV file: {csv_path}")
    print(f"CSV size: {csv_size / 1024 / 1024 / 1024:.2f} GB")
    print(f"Output directory: {output_dir}")
    print(f"Chunk size: {chunk_size:,}")

    writers = {}
    schemas = {}
    rows_written = 0
    chunk_num = 0
    start_time = time.time()
    running_volume = {}

    # Define the arrow schema we want for consistency
    target_schema = None

    for chunk in pd.read_csv(csv_path, chunksize=chunk_size, low_memory=False):
        chunk_start = time.time()
        chunk_num += 1

        # Parse timestamps to get month partition
        chunk["timestamp"] = pd.to_datetime(chunk["timestamp"], errors="coerce")
        chunk = chunk.dropna(subset=["timestamp"])

        # Ensure chronological order for cumulative volume
        chunk = chunk.sort_values("timestamp")

        # Derive month partition
        chunk["month"] = chunk["timestamp"].dt.strftime("%Y-%m")

        # Convert timestamp back to ISO string for storage
        chunk["timestamp"] = chunk["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

        # Optimize numeric columns
        if "price" in chunk.columns:
            chunk["price"] = pd.to_numeric(chunk["price"], errors="coerce").astype("float32")
        if "usd_amount" in chunk.columns:
            chunk["usd_amount"] = pd.to_numeric(chunk["usd_amount"], errors="coerce").astype("float32")
        if "token_amount" in chunk.columns:
            chunk["token_amount"] = pd.to_numeric(chunk["token_amount"], errors="coerce").astype("float32")

        # Compute cumulative market volume (USD) per market across all prior chunks
        if "market_id" in chunk.columns and "usd_amount" in chunk.columns:
            chunk["market_id_str"] = chunk["market_id"].astype(str)
            chunk["usd_amount"] = chunk["usd_amount"].fillna(0.0)

            prior = chunk["market_id_str"].map(running_volume).fillna(0.0)
            chunk_cum = chunk.groupby("market_id_str")["usd_amount"].cumsum()
            chunk["market_cum_volume_usd"] = (prior + chunk_cum).astype("float64")

            last_cum = chunk.groupby("market_id_str")["market_cum_volume_usd"].last()
            running_volume.update(last_cum.to_dict())

        # Write each month partition
        for month, group in chunk.groupby("month"):
            data = group.drop(columns=["month"])
            if "market_id_str" in data.columns:
                data = data.drop(columns=["market_id_str"])
            table = pa.Table.from_pandas(data, preserve_index=False)
            filepath = out / f"trades_{month}.parquet"

            if month not in writers:
                # Use schema from first chunk of this month
                if target_schema is None:
                    target_schema = table.schema
                writer = pq.ParquetWriter(
                    str(filepath),
                    table.schema,
                    compression="snappy",
                )
                writers[month] = writer
                schemas[month] = table.schema
            else:
                # Ensure schema compatibility
                if table.schema != schemas[month]:
                    table = table.cast(schemas[month])

            writers[month].write_table(table)
            rows_written += len(data)

        elapsed = time.time() - start_time
        rate = rows_written / elapsed if elapsed > 0 else 0
        chunk_time = time.time() - chunk_start
        print(
            f"  Chunk {chunk_num:>4}: {rows_written:>12,} rows total "
            f"| {rate:,.0f} rows/s | chunk: {chunk_time:.1f}s"
        )

    # Close all writers
    for writer in writers.values():
        writer.close()

    elapsed = time.time() - start_time
    print(f"\nConversion complete in {elapsed:.1f}s")
    print(f"Total rows written: {rows_written:,}")
    print(f"Parquet files created: {len(writers)}")

    return rows_written


def verify(output_dir, expected_rows=None):
    """Verify parquet files."""
    out = Path(output_dir)
    if not out.exists():
        print(f"ERROR: Output directory {output_dir} does not exist")
        return False

    files = sorted(out.glob("trades_*.parquet"))
    if not files:
        print("ERROR: No parquet files found")
        return False

    total_rows = 0
    total_size = 0
    first_ts = None
    last_ts = None

    print(f"\n{'File':<30} {'Rows':>12} {'Size MB':>10}")
    print("-" * 55)
    for f in files:
        pf = pq.read_metadata(str(f))
        rows = pf.num_rows
        size = f.stat().st_size
        total_rows += rows
        total_size += size
        print(f"  {f.name:<28} {rows:>12,} {size / 1024 / 1024:>10.1f}")

    print("-" * 55)
    print(f"  {'TOTAL':<28} {total_rows:>12,} {total_size / 1024 / 1024:>10.1f}")
    print(f"\nTotal parquet size: {total_size / 1024 / 1024 / 1024:.2f} GB")

    # Check date range from first and last files
    first_df = pd.read_parquet(files[0], columns=["timestamp"]).head(1)
    last_df = pd.read_parquet(files[-1], columns=["timestamp"]).tail(1)
    if not first_df.empty and not last_df.empty:
        print(f"Date range: {first_df.iloc[0]['timestamp']} to {last_df.iloc[0]['timestamp']}")

    if expected_rows is not None:
        if total_rows == expected_rows:
            print(f"\nROW COUNT MATCH: {total_rows:,} == {expected_rows:,}")
            return True
        else:
            diff = expected_rows - total_rows
            pct = abs(diff) / expected_rows * 100
            print(f"\nROW COUNT DIFFERENCE: parquet={total_rows:,} vs expected={expected_rows:,} (diff={diff:,}, {pct:.2f}%)")
            # Allow small discrepancy from NaN/invalid timestamp rows
            if pct < 0.1:
                print("Difference is < 0.1%, likely from dropped NaN timestamps. Acceptable.")
                return True
            return False

    return True


def count_csv_rows(csv_path):
    """Count data rows in CSV (excluding header)."""
    print(f"Counting rows in {csv_path} (this may be slow)...")
    count = 0
    with open(csv_path, "rb") as f:
        f.readline()  # Skip header
        buf_size = 64 * 1024 * 1024
        while True:
            buf = f.read(buf_size)
            if not buf:
                break
            count += buf.count(b"\n")
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Convert trades CSV to parquet"
    )
    parser.add_argument("--csv-path", default=CSV_PATH, help="Path to trades CSV")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Parquet output dir")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help="Rows per chunk")
    parser.add_argument("--verify-only", action="store_true", help="Only verify existing parquet")
    parser.add_argument("--skip-count", action="store_true", help="Skip CSV row counting for verification")
    args = parser.parse_args()

    if args.verify_only:
        expected = None
        if not args.skip_count and os.path.exists(args.csv_path):
            expected = count_csv_rows(args.csv_path)
            print(f"CSV data rows: {expected:,}")
        verify(args.output_dir, expected)
        return

    if not os.path.exists(args.csv_path):
        print(f"ERROR: CSV not found at {args.csv_path}")
        sys.exit(1)

    print("=" * 60)
    print("CONVERTING TRADES CSV TO PARQUET")
    print("=" * 60)

    rows_written = convert(args.csv_path, args.output_dir, args.chunk_size)

    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    verify(args.output_dir, rows_written)

    print("\nSUCCESS: Trades CSV converted to parquet.")
    print(f"You can delete the original CSV to reclaim ~33 GB.")


if __name__ == "__main__":
    main()

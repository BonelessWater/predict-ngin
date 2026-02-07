#!/usr/bin/env python3
"""
Convert polymarket_prices table from SQLite to monthly partitioned parquet files.

Reads the 148M-row polymarket_prices table in chunks and writes compressed
parquet files partitioned by month. Drops redundant columns (id, datetime)
to save space.

Expected compression: ~130 GB SQLite -> ~5-10 GB parquet (snappy).

Usage:
    python scripts/convert_prices_to_parquet.py
    python scripts/convert_prices_to_parquet.py --chunk-size 10000000
    python scripts/convert_prices_to_parquet.py --verify-only
"""

import argparse
import os
import sqlite3
import sys
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


DB_PATH = "data/prediction_markets.db"
OUTPUT_DIR = "data/parquet/prices"
CHUNK_SIZE = 5_000_000


def get_total_rows(conn):
    """Get total row count from polymarket_prices."""
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM polymarket_prices")
    return cur.fetchone()[0]


def convert(db_path, output_dir, chunk_size):
    """Convert polymarket_prices to partitioned parquet.

    Uses rowid-based pagination instead of OFFSET for much better
    performance on large tables.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)

    # Get max rowid (fast) instead of COUNT (slow)
    cur = conn.cursor()
    cur.execute("SELECT MAX(rowid) FROM polymarket_prices")
    max_rowid = cur.fetchone()[0] or 0
    print(f"Max rowid in polymarket_prices: {max_rowid:,}")
    print(f"Output directory: {output_dir}")
    print(f"Chunk size: {chunk_size:,}")
    sys.stdout.flush()

    # Track parquet writers per month to use appending
    writers = {}
    schemas = {}
    rows_written = 0
    last_rowid = 0
    start_time = time.time()

    while last_rowid < max_rowid:
        chunk_start = time.time()
        query = f"""
            SELECT market_id, outcome, timestamp, price
            FROM polymarket_prices
            WHERE rowid > {last_rowid} AND rowid <= {last_rowid + chunk_size}
        """
        chunk = pd.read_sql_query(query, conn)
        last_rowid += chunk_size

        if chunk.empty:
            continue

        # Optimize dtypes
        chunk["timestamp"] = pd.to_numeric(chunk["timestamp"], errors="coerce")
        chunk["price"] = pd.to_numeric(chunk["price"], errors="coerce").astype("float32")
        chunk = chunk.dropna(subset=["timestamp"])

        # Derive month partition column
        chunk["month"] = pd.to_datetime(
            chunk["timestamp"], unit="s", errors="coerce"
        ).dt.strftime("%Y-%m")
        chunk = chunk.dropna(subset=["month"])

        # Write each month partition
        for month, group in chunk.groupby("month"):
            data = group.drop(columns=["month"])
            table = pa.Table.from_pandas(data, preserve_index=False)
            filepath = out / f"prices_{month}.parquet"

            if month not in writers:
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
        pct = min(100, last_rowid / max_rowid * 100)
        chunk_time = time.time() - chunk_start
        print(
            f"  {rows_written:>12,} rows ({pct:5.1f}% of rowid range) "
            f"| {rate:,.0f} rows/s | chunk: {chunk_time:.1f}s",
            flush=True,
        )

    # Close all writers
    for writer in writers.values():
        writer.close()

    conn.close()
    elapsed = time.time() - start_time
    print(f"\nConversion complete in {elapsed:.1f}s")
    print(f"Total rows written: {rows_written:,}")
    print(f"Parquet files created: {len(writers)}")

    return rows_written, max_rowid


def verify(output_dir, expected_rows=None):
    """Verify parquet files match expected row count."""
    out = Path(output_dir)
    if not out.exists():
        print(f"ERROR: Output directory {output_dir} does not exist")
        return False

    files = sorted(out.glob("prices_*.parquet"))
    if not files:
        print("ERROR: No parquet files found")
        return False

    total_rows = 0
    total_size = 0
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

    if expected_rows is not None:
        if total_rows == expected_rows:
            print(f"ROW COUNT MATCH: {total_rows:,} == {expected_rows:,}")
            return True
        else:
            diff = expected_rows - total_rows
            print(f"ROW COUNT MISMATCH: parquet={total_rows:,} vs expected={expected_rows:,} (diff={diff:,})")
            return False

    return True


def spot_check(db_path, output_dir):
    """Spot-check a random market's data between SQLite and parquet."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Pick a market with moderate data
    cur.execute("""
        SELECT market_id, COUNT(*) as cnt
        FROM polymarket_prices
        GROUP BY market_id
        HAVING cnt BETWEEN 1000 AND 10000
        ORDER BY RANDOM()
        LIMIT 1
    """)
    row = cur.fetchone()
    if not row:
        print("No suitable market found for spot check")
        conn.close()
        return

    market_id, expected_count = row[0], row[1]
    print(f"\nSpot-checking market: {market_id} (expected {expected_count:,} rows)")

    # Load from SQLite
    sqlite_df = pd.read_sql_query(
        "SELECT market_id, outcome, timestamp, price FROM polymarket_prices WHERE market_id = ?",
        conn,
        params=(market_id,),
    )
    conn.close()

    # Load from parquet
    out = Path(output_dir)
    parquet_dfs = []
    for f in sorted(out.glob("prices_*.parquet")):
        df = pd.read_parquet(f, filters=[("market_id", "==", market_id)])
        if not df.empty:
            parquet_dfs.append(df)

    if not parquet_dfs:
        print(f"  WARNING: Market {market_id} not found in parquet files")
        return

    parquet_df = pd.concat(parquet_dfs).sort_values("timestamp").reset_index(drop=True)
    sqlite_df = sqlite_df.sort_values("timestamp").reset_index(drop=True)

    # Compare row counts
    if len(sqlite_df) == len(parquet_df):
        print(f"  Row count match: {len(sqlite_df):,}")
    else:
        print(f"  ROW COUNT MISMATCH: sqlite={len(sqlite_df):,} vs parquet={len(parquet_df):,}")
        return

    # Compare a few values
    sqlite_df["price"] = sqlite_df["price"].astype("float32")
    sqlite_df["timestamp"] = pd.to_numeric(sqlite_df["timestamp"], errors="coerce")
    parquet_df["timestamp"] = pd.to_numeric(parquet_df["timestamp"], errors="coerce")

    merged = sqlite_df.merge(
        parquet_df, on=["market_id", "outcome", "timestamp"], suffixes=("_sq", "_pq")
    )
    if len(merged) == len(sqlite_df):
        price_match = (merged["price_sq"] - merged["price_pq"]).abs().max()
        print(f"  All rows matched on join. Max price diff: {price_match:.6f}")
    else:
        print(f"  WARNING: Only {len(merged):,} of {len(sqlite_df):,} rows matched on join")


def main():
    parser = argparse.ArgumentParser(
        description="Convert polymarket_prices SQLite table to parquet"
    )
    parser.add_argument("--db-path", default=DB_PATH, help="SQLite database path")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Parquet output dir")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help="Rows per chunk")
    parser.add_argument("--verify-only", action="store_true", help="Only verify existing parquet")
    parser.add_argument("--spot-check", action="store_true", help="Spot-check a random market")
    args = parser.parse_args()

    if args.verify_only:
        verify(args.output_dir)
        if args.spot_check and os.path.exists(args.db_path):
            spot_check(args.db_path, args.output_dir)
        return

    if not os.path.exists(args.db_path):
        print(f"ERROR: Database not found at {args.db_path}")
        sys.exit(1)

    print("=" * 60)
    print("CONVERTING polymarket_prices TO PARQUET")
    print("=" * 60)

    rows_written, _ = convert(args.db_path, args.output_dir, args.chunk_size)

    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    ok = verify(args.output_dir, rows_written)

    if args.spot_check:
        spot_check(args.db_path, args.output_dir)

    if ok:
        print("\nSUCCESS: Prices converted to parquet successfully.")
        print(f"You can now drop the polymarket_prices table from SQLite to reclaim ~130 GB.")
    else:
        print("\nWARNING: Row count mismatch detected. Do NOT drop the SQLite table yet.")


if __name__ == "__main__":
    main()

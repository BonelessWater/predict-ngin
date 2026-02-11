#!/usr/bin/env python3
"""
Download and munge Kalshi dataset from HuggingFace to project schema.

Source: https://huggingface.co/datasets/TrevorJS/kalshi-trades

- 154M trades (16 parquet shards)
- 17M markets (4 parquet shards)
- Output: data/kalshi/markets.parquet, data/kalshi/trades/trades_YYYY-MM.parquet
- Optional: data/kalshi/prices/prices_YYYY-MM.parquet (derived from trades)

Usage:
    # Download markets + trades to project schema (~5.7 GB)
    python scripts/data/download_kalshi_hf.py

    # Also derive OHLCV prices from trades (for arbitrage backtest)
    python scripts/data/download_kalshi_hf.py --derive-prices

    # Filter markets by minimum volume
    python scripts/data/download_kalshi_hf.py --min-volume 200000

    # Output to custom directory
    python scripts/data/download_kalshi_hf.py --output-dir data/kalshi
"""

import argparse
import sys
import time
from pathlib import Path

REPO_ID = "TrevorJS/kalshi-trades"

# Project schema (from fetcher.py)
TRADES_COLUMNS = [
    "trade_id", "ticker", "event_ticker", "yes_price", "no_price",
    "count", "taker_side", "created_time", "timestamp_unix",
]
MARKETS_COLUMNS = [
    "ticker", "event_ticker", "series_ticker", "market_type", "title", "subtitle",
    "status", "result", "yes_bid", "yes_ask", "no_bid", "no_ask", "last_price",
    "volume", "volume_24h", "open_interest", "created_time", "open_time",
    "close_time", "expiration_time", "settlement_value", "category",
]


def _ensure_deps():
    """Ensure required dependencies. DuckDB preferred for hf:// streaming; fallback to huggingface_hub."""
    try:
        import pandas
        import pyarrow
        import pyarrow.parquet
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install: pip install pandas pyarrow")
        sys.exit(1)
    try:
        import duckdb
        return True  # DuckDB available
    except ImportError:
        try:
            import huggingface_hub
            print("DuckDB not found. Using huggingface_hub fallback (downloads ~5.7GB first).")
            print("  For faster streaming: pip install duckdb")
            return False
        except ImportError:
            print("Install duckdb or huggingface_hub: pip install duckdb  # or pip install huggingface_hub")
            sys.exit(1)


def _run_fallback_hub(output_dir: Path, min_volume: float) -> None:
    """Fallback when DuckDB unavailable: download via huggingface_hub and process with pandas."""
    from huggingface_hub import snapshot_download
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    print("\n[Fallback] Downloading dataset via huggingface_hub (~5.7 GB)...")
    cache = Path(snapshot_download(REPO_ID, repo_type="dataset"))
    print(f"  Cached at: {cache}")

    # Markets
    print("\n[1/3] Loading markets...")
    market_files = sorted(cache.glob("markets-*.parquet"))
    if not market_files:
        print("  No markets-*.parquet found")
        return
    dfs = []
    for i, f in enumerate(market_files):
        print(f"  Reading {f.name} ({i+1}/{len(market_files)})...")
        dfs.append(pd.read_parquet(f))
    df = pd.concat(dfs, ignore_index=True)
    df = df[df["volume"] >= min_volume]
    if df.empty:
        print("  No markets above min_volume")
        return

    df["subtitle"] = (df.get("yes_sub_title", "") or "").fillna("") + " / " + (df.get("no_sub_title", "") or "").fillna("")
    for col in ["series_ticker", "expiration_time", "category"]:
        df[col] = ""
    df["settlement_value"] = None
    out = df[[c for c in MARKETS_COLUMNS if c in df.columns]].copy()
    for c in MARKETS_COLUMNS:
        if c not in out.columns:
            out[c] = "" if c != "settlement_value" else None
    out = out[MARKETS_COLUMNS]
    out["title"] = out["title"].astype(str).str[:200]
    out["subtitle"] = out["subtitle"].astype(str).str[:200]
    output_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(out, preserve_index=False), output_dir / "markets.parquet", compression="zstd", compression_level=3)
    print(f"  Saved {len(out):,} markets")

    map_df = pd.concat([pd.read_parquet(f)[["ticker", "event_ticker"]] for f in market_files]).drop_duplicates("ticker")
    event_map = dict(zip(map_df["ticker"], map_df["event_ticker"]))
    del dfs, df, out, map_df

    # Trades
    print("\n[2/3] Loading trades...")
    trade_files = sorted(cache.glob("trades-*.parquet"))
    trades_dir = output_dir / "trades"
    trades_dir.mkdir(parents=True, exist_ok=True)
    monthly: dict[str, list] = {}
    for i, f in enumerate(trade_files):
        tdf = pd.read_parquet(f)
        tdf["yes_price"] = tdf["yes_price"] / 100.0
        tdf["no_price"] = tdf["no_price"] / 100.0
        tdf["timestamp_unix"] = pd.to_datetime(tdf["created_time"], utc=True).astype("int64") // 10**9
        tdf["event_ticker"] = tdf["ticker"].map(lambda x: event_map.get(x, "")).fillna("")
        tdf["created_time"] = tdf["created_time"].astype(str)
        tdf = tdf[TRADES_COLUMNS]
        for month, grp in tdf.groupby(pd.to_datetime(tdf["created_time"]).dt.to_period("M").astype(str).str[:7]):
            if month not in monthly:
                monthly[month] = []
            monthly[month].append(grp)
        print(f"  [{i+1}/{len(trade_files)}] {f.name}: {len(tdf):,} trades")
        del tdf

    total = 0
    for month, grps in sorted(monthly.items()):
        combined = pd.concat(grps, ignore_index=True)
        total += len(combined)
        out_path = trades_dir / f"trades_{month}.parquet"
        pq.write_table(pa.Table.from_pandas(combined, preserve_index=False), out_path, compression="zstd", compression_level=3)
        print(f"  Wrote {out_path.name}: {len(combined):,} rows")
    print(f"  Total trades: {total:,}")


def _load_markets_hf(output_dir: Path, min_volume: float) -> "dict[str, str]":
    """Load markets from HF, transform to project schema, write. Returns ticker->event_ticker map."""
    import duckdb
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    print("\n[1/3] Loading markets from HuggingFace...")
    t0 = time.time()

    print("  Connecting DuckDB, loading httpfs...")
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    print(f"  Ready in {time.time() - t0:.1f}s")

    print("  Fetching markets (hf://datasets/.../markets-*.parquet, ~1.1 GB)...")
    print("  (First fetch streams ~1.1 GB over network - may take 5-15 min depending on connection)")
    sys.stdout.flush()
    fetch_start = time.time()
    df = _duckdb_query_with_retry(con, """
        SELECT
            ticker,
            event_ticker,
            market_type,
            title,
            COALESCE(yes_sub_title, '') || ' / ' || COALESCE(no_sub_title, '') as subtitle,
            status,
            result,
            yes_bid, yes_ask, no_bid, no_ask, last_price,
            volume, volume_24h, open_interest,
            created_time::varchar as created_time,
            open_time::varchar as open_time,
            close_time::varchar as close_time
        FROM read_parquet('hf://datasets/TrevorJS/kalshi-trades/markets-*.parquet')
        WHERE volume >= ?
    """, [min_volume])

    fetch_elapsed = time.time() - fetch_start
    print(f"  Fetched {len(df):,} markets in {fetch_elapsed:.1f}s ({fetch_elapsed/60:.1f} min)")

    con.close()

    if df.empty:
        print("  No markets above min_volume. Try --min-volume 0")
        return {}

    print("  Transforming schema...")
    # Add missing columns (HF dataset doesn't have these)
    df["series_ticker"] = ""
    df["expiration_time"] = ""
    df["settlement_value"] = None
    df["category"] = ""
    df["liquidity"] = 0

    # Reorder to match project schema
    out = df[[c for c in MARKETS_COLUMNS if c in df.columns]].copy()
    for col in MARKETS_COLUMNS:
        if col not in out.columns:
            out[col] = "" if col != "settlement_value" else None

    out = out[MARKETS_COLUMNS]
    out["title"] = out["title"].astype(str).str[:200]
    out["subtitle"] = out["subtitle"].astype(str).str[:200]
    out["volume"] = out["volume"].fillna(0).astype("int64")
    out["volume_24h"] = out["volume_24h"].fillna(0).astype("int64")
    out["open_interest"] = out["open_interest"].fillna(0).astype("int64")

    out_path = output_dir / "markets.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Writing {out_path}...")
    pq.write_table(
        pa.Table.from_pandas(out, preserve_index=False),
        out_path, compression="zstd", compression_level=3,
    )
    print(f"  [1/3] Done: {len(out):,} markets in {time.time() - t0:.1f}s total")

    return dict(zip(out["ticker"], out["event_ticker"]))


def _duckdb_query_with_retry(con, sql: str, params=None, max_retries: int = 5):
    """Execute DuckDB query with retry on HTTP 429 rate limit."""
    import duckdb
    last_err = None
    for attempt in range(max_retries):
        try:
            if params is not None:
                return con.execute(sql, params).fetchdf()
            return con.execute(sql).fetchdf()
        except Exception as e:
            last_err = e
            err_str = str(e).lower()
            if "429" in err_str or "rate" in err_str or "too many requests" in err_str:
                wait = min(60 * (2 ** attempt), 300)  # 60s, 120s, 240s, 300s, 300s
                print(f"  HTTP 429 rate limit - waiting {wait}s before retry ({attempt + 1}/{max_retries})...")
                sys.stdout.flush()
                time.sleep(wait)
            else:
                raise
    raise last_err


def _load_trades_hf(output_dir: Path, event_ticker_map: dict) -> None:
    """Load trades from HF, join event_ticker, transform, write monthly parquet."""
    import duckdb
    import pyarrow as pa
    import pyarrow.parquet as pq

    trades_dir = output_dir / "trades"
    trades_dir.mkdir(parents=True, exist_ok=True)

    print("\n[2/3] Loading trades from HuggingFace (streaming by month)...")
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")

    # Get distinct months
    print("  Scanning trades for month list (reads ~4.5 GB)...")
    sys.stdout.flush()
    t0 = time.time()
    months_df = _duckdb_query_with_retry(con, """
        SELECT DISTINCT strftime(created_time::timestamp, '%Y-%m') as month
        FROM read_parquet('hf://datasets/TrevorJS/kalshi-trades/trades-*.parquet')
        ORDER BY month
    """)
    months = months_df["month"].tolist()
    print(f"  Months: {months[0]} .. {months[-1]} ({len(months)} total) in {time.time() - t0:.1f}s")

    total_rows = 0
    for i, month in enumerate(months):
        # Query this month's trades (with retry on 429)
        df = _duckdb_query_with_retry(con, """
            SELECT
                trade_id,
                ticker,
                count,
                yes_price::double / 100.0 as yes_price,
                no_price::double / 100.0 as no_price,
                taker_side,
                created_time::varchar as created_time,
                epoch(created_time::timestamp)::bigint as timestamp_unix
            FROM read_parquet('hf://datasets/TrevorJS/kalshi-trades/trades-*.parquet')
            WHERE strftime(created_time::timestamp, '%Y-%m') = ?
        """, [month])

        if df.empty:
            continue

        # Add event_ticker from map (tickers not in map get "")
        df["event_ticker"] = df["ticker"].map(lambda t: event_ticker_map.get(t, "")).fillna("")

        # Reorder columns
        df = df[TRADES_COLUMNS]

        out_path = trades_dir / f"trades_{month}.parquet"
        pq.write_table(
            pa.Table.from_pandas(df, preserve_index=False),
            out_path, compression="zstd", compression_level=3,
        )
        total_rows += len(df)
        print(f"  [{i+1}/{len(months)}] {month}: {len(df):,} trades -> {out_path.name}")

        # Brief pause between months to reduce rate-limit risk
        if i < len(months) - 1:
            time.sleep(2)

    con.close()
    print(f"  Total trades: {total_rows:,}")


def _derive_prices_from_trades(output_dir: Path) -> None:
    """Aggregate trades into daily OHLCV candlesticks for KalshiPriceStore."""
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    trades_dir = output_dir / "trades"
    prices_dir = output_dir / "prices"
    prices_dir.mkdir(parents=True, exist_ok=True)

    print("\n[3/3] Deriving OHLCV prices from trades...")
    trade_files = sorted(trades_dir.glob("trades_*.parquet"))
    if not trade_files:
        print("  No trade files found. Run without --derive-prices first.")
        return

    total_rows = 0
    for f in trade_files:
        month = f.stem.replace("trades_", "")
        df = pd.read_parquet(f)
        if df.empty:
            continue
        df = df.sort_values("timestamp_unix")
        df["date"] = pd.to_datetime(df["timestamp_unix"], unit="s", utc=True).dt.date
        agg = df.groupby(["ticker", "date"]).agg(
            open=("yes_price", "first"),
            high=("yes_price", "max"),
            low=("yes_price", "min"),
            close=("yes_price", "last"),
            volume=("count", "sum"),
        ).reset_index()
        agg["mean"] = (agg["open"] + agg["high"] + agg["low"] + agg["close"]) / 4
        agg["open_interest"] = 0
        agg["timestamp"] = (pd.to_datetime(agg["date"]).astype("int64") // 10**9).astype("int64")
        out = agg[["ticker", "timestamp", "open", "high", "low", "close", "mean", "volume", "open_interest"]]
        out_path = prices_dir / f"prices_{month}.parquet"
        pq.write_table(pa.Table.from_pandas(out, preserve_index=False), out_path, compression="zstd", compression_level=3)
        total_rows += len(out)
        print(f"  {month}: {len(out):,} daily candles -> {out_path.name}")

    print(f"  Total price rows: {total_rows:,}")


def main():
    parser = argparse.ArgumentParser(
        description="Download Kalshi dataset from HuggingFace and munge to project schema",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--output-dir",
        default="data/kalshi",
        help="Output directory (default: data/kalshi)",
    )
    parser.add_argument(
        "--min-volume",
        type=float,
        default=0,
        help="Minimum market volume to include (default: 0)",
    )
    parser.add_argument(
        "--derive-prices",
        action="store_true",
        help="Derive OHLCV prices from trades (for arbitrage backtest)",
    )
    parser.add_argument(
        "--derive-prices-only",
        action="store_true",
        help="Skip download; only derive prices from existing trades (requires data/kalshi/trades/)",
    )
    parser.add_argument(
        "--use-hub-fallback",
        action="store_true",
        help="Use huggingface_hub to download first, then process locally (avoids HF rate limits)",
    )
    args = parser.parse_args()

    use_duckdb = _ensure_deps() and not args.use_hub_fallback
    output_dir = Path(args.output_dir)

    print("=" * 60)
    print("KALSHI HUGGINGFACE DATASET")
    print("=" * 60)
    print(f"Source: huggingface.co/datasets/{REPO_ID}")
    print(f"Output: {output_dir}")
    print(f"Min volume: {args.min_volume:,.0f}")
    print(f"Derive prices: {args.derive_prices}")
    if args.use_hub_fallback:
        print("Mode: huggingface_hub (download first, avoids rate limits)")
    print("=" * 60)

    t0 = time.time()

    if args.derive_prices_only:
        _derive_prices_from_trades(output_dir)
        elapsed = time.time() - t0
        print(f"\nDone in {elapsed:.1f}s")
        return

    if use_duckdb:
        _load_markets_hf(output_dir, args.min_volume)
        print("  Loading ticker->event_ticker map for trades (full scan)...")
        map_t0 = time.time()
        import duckdb
        con = duckdb.connect()
        con.execute("INSTALL httpfs; LOAD httpfs;")
        map_df = _duckdb_query_with_retry(con, """
            SELECT ticker, event_ticker FROM read_parquet('hf://datasets/TrevorJS/kalshi-trades/markets-*.parquet')
        """)
        event_map = dict(zip(map_df["ticker"], map_df["event_ticker"]))
        con.close()
        print(f"  Loaded {len(event_map):,} mappings in {time.time() - map_t0:.1f}s")
        _load_trades_hf(output_dir, event_map)
    else:
        _run_fallback_hub(output_dir, args.min_volume)

    if args.derive_prices:
        _derive_prices_from_trades(output_dir)
    else:
        print("\n[3/3] Skipped (use --derive-prices to create OHLCV from trades)")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed / 60:.1f} minutes")
    print(f"  Markets: {output_dir / 'markets.parquet'}")
    print(f"  Trades:  {output_dir / 'trades'}")
    if args.derive_prices:
        print(f"  Prices:  {output_dir / 'prices'}")


if __name__ == "__main__":
    main()

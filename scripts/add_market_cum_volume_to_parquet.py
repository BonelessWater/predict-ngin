#!/usr/bin/env python3
"""
Add cumulative USD liquidity proxy to existing trades parquet files.

Computes cumulative USD volume per market across all months in
chronological order and writes the column into each monthly parquet.
The column is named market_cum_liquidity_usd.
"""

import argparse
from pathlib import Path
from typing import Dict

try:
    import polars as pl
except ImportError:  # pragma: no cover
    pl = None

import pandas as pd


def _update_with_polars(path: Path, running: Dict[str, float], force: bool) -> None:
    df = pl.read_parquet(path)
    if "market_cum_liquidity_usd" in df.columns and not force:
        return

    if "market_id" not in df.columns or "usd_amount" not in df.columns:
        return

    df = df.with_columns(
        pl.col("market_id")
        .cast(pl.Utf8, strict=False)
        .str.replace(r"\.0$", "")
        .alias("market_id_str"),
        pl.col("timestamp").cast(pl.Utf8, strict=False).alias("timestamp_str"),
        pl.col("usd_amount").cast(pl.Float64, strict=False).fill_null(0.0).alias("usd_amount"),
    )

    df = df.sort(["market_id_str", "timestamp_str"])

    if running:
        prior_df = pl.DataFrame(
            {"market_id_str": list(running.keys()), "prior_cum": list(running.values())}
        )
        df = df.join(prior_df, on="market_id_str", how="left")
    else:
        df = df.with_columns(pl.lit(0.0).alias("prior_cum"))

    df = df.with_columns(
        pl.col("prior_cum").fill_null(0.0),
        pl.col("usd_amount").cum_sum().over("market_id_str").alias("chunk_cum"),
    )
    df = df.with_columns(
        (pl.col("prior_cum") + pl.col("chunk_cum")).alias("market_cum_liquidity_usd")
    ).drop(["prior_cum", "chunk_cum", "timestamp_str"])

    last_vals = (
        df.group_by("market_id_str")
        .agg(pl.col("market_cum_liquidity_usd").max().alias("last_cum"))
    )
    running.update(
        dict(zip(last_vals["market_id_str"].to_list(), last_vals["last_cum"].to_list()))
    )

    df = df.drop(["market_id_str"])
    tmp_path = path.with_suffix(".tmp.parquet")
    df.write_parquet(tmp_path, compression="snappy")
    tmp_path.replace(path)


def _update_with_pandas(path: Path, running: Dict[str, float], force: bool) -> None:
    df = pd.read_parquet(path)
    if "market_cum_liquidity_usd" in df.columns and not force:
        return

    if "market_id" not in df.columns or "usd_amount" not in df.columns:
        return

    df["market_id_str"] = df["market_id"].astype(str).str.replace(".0", "", regex=False)
    df["usd_amount"] = pd.to_numeric(df["usd_amount"], errors="coerce").fillna(0.0)
    df = df.sort_values(["market_id_str", "timestamp"])

    prior = df["market_id_str"].map(running).fillna(0.0)
    df["market_cum_liquidity_usd"] = prior + df.groupby("market_id_str")["usd_amount"].cumsum()

    last_cum = df.groupby("market_id_str")["market_cum_liquidity_usd"].last()
    running.update(last_cum.to_dict())

    df = df.drop(columns=["market_id_str"])
    tmp_path = path.with_suffix(".tmp.parquet")
    df.to_parquet(tmp_path, index=False)
    tmp_path.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add cumulative market liquidity (USD volume proxy) to trades parquet files."
    )
    parser.add_argument("--parquet-dir", default="data/parquet/trades", help="Trades parquet directory")
    parser.add_argument("--force", action="store_true", help="Recompute even if column exists")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of files (for testing)")
    args = parser.parse_args()

    out = Path(args.parquet_dir)
    files = sorted(out.glob("trades_*.parquet"))
    if args.limit:
        files = files[: args.limit]

    running: Dict[str, float] = {}

    for f in files:
        if pl is not None:
            _update_with_polars(f, running, args.force)
        else:
            _update_with_pandas(f, running, args.force)
        print(f"Updated {f.name}")


if __name__ == "__main__":
    main()

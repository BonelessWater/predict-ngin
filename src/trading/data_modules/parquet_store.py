"""
Parquet-based data store for large tables (prices, trades).

Provides pandas-compatible loading with date/market filtering.
Small metadata tables remain in SQLite.

Usage:
    from trading.data_modules.parquet_store import TradeStore, PriceStore

    trades = TradeStore().load_trades(min_usd=100, start_date="2024-01-01")
    prices = PriceStore().get_price_history("12345", "YES")
"""

from pathlib import Path
import os
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

try:
    import polars as pl
except ImportError:  # pragma: no cover - optional dependency
    pl = None

def _require_polars() -> bool:
    return os.getenv("PREDICT_NGIN_FORCE_POLARS", "").strip().lower() in {
        "1", "true", "yes", "on"
    }


DEFAULT_PARQUET_DIR = "data/parquet"


class TradeStore:
    """Load trade data from monthly partitioned parquet files."""

    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = Path(base_dir or f"{DEFAULT_PARQUET_DIR}/trades")

    def available(self) -> bool:
        """Check if parquet trade files exist."""
        return self.base_dir.exists() and any(self.base_dir.glob("trades_*.parquet"))

    def list_months(self) -> List[str]:
        """List available months (YYYY-MM)."""
        files = sorted(self.base_dir.glob("trades_*.parquet"))
        return [f.stem.replace("trades_", "") for f in files]

    def _select_files(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[Path]:
        """Select parquet files based on date range."""
        files = sorted(self.base_dir.glob("trades_*.parquet"))
        if not files:
            return []

        if start_date:
            start_month = start_date[:7]  # YYYY-MM
            files = [f for f in files if f.stem.replace("trades_", "") >= start_month]
        if end_date:
            end_month = end_date[:7]
            files = [f for f in files if f.stem.replace("trades_", "") <= end_month]

        return files

    def load_trades(
        self,
        min_usd: float = 10.0,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: Optional[int] = None,
        columns: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """
        Load trades with optional filtering.

        Args:
            min_usd: Minimum USD trade size
            start_date: Filter trades after this date (YYYY-MM-DD)
            end_date: Filter trades before this date (YYYY-MM-DD)
            limit: Maximum rows to return

        Returns:
            DataFrame with trade data
        """
        if _require_polars() and pl is None:
            raise RuntimeError("Polars is required but not installed.")

        if pl is not None:
            try:
                return self._load_trades_polars(min_usd, start_date, end_date, limit, columns)
            except Exception:
                # Fallback to pandas path if polars hits a schema/type edge case
                if _require_polars():
                    raise

        files = self._select_files(start_date, end_date)
        if not files:
            return pd.DataFrame()

        dfs = []
        total_rows = 0

        for f in files:
            df = pd.read_parquet(f, columns=list(columns) if columns else None)

            # Apply USD filter
            if "usd_amount" in df.columns:
                df = df[df["usd_amount"] >= min_usd]

            # Apply date filters (more precise than file-level)
            if start_date and "timestamp" in df.columns:
                df = df[df["timestamp"] >= start_date]
            if end_date and "timestamp" in df.columns:
                df = df[df["timestamp"] <= end_date]

            if df.empty:
                continue

            dfs.append(df)
            total_rows += len(df)

            if limit and total_rows >= limit:
                break

        if not dfs:
            return pd.DataFrame()

        result = pd.concat(dfs, ignore_index=True)
        if "timestamp" in result.columns:
            result = result.sort_values("timestamp")

        if limit:
            result = result.head(limit)

        return result

    def _load_trades_polars(
        self,
        min_usd: float = 10.0,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: Optional[int] = None,
        columns: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """Load trades with polars for faster parallel IO."""
        files = self._select_files(start_date, end_date)
        if not files:
            return pd.DataFrame()

        try:
            lf = pl.scan_parquet(
                [str(f) for f in files],
                columns=list(columns) if columns else None,
                missing_columns="insert",
            )
            schema = lf.schema

            if "usd_amount" in schema:
                lf = lf.filter(pl.col("usd_amount") >= min_usd)

            # NOTE: timestamp in parquet is stored as ISO-like strings.
            # We keep string comparisons to match the pandas fallback behavior.
            if start_date and "timestamp" in schema:
                lf = lf.filter(pl.col("timestamp").cast(pl.Utf8, strict=False) >= start_date)
            if end_date and "timestamp" in schema:
                lf = lf.filter(pl.col("timestamp").cast(pl.Utf8, strict=False) <= end_date)

            df = lf.collect()
            if df.is_empty():
                return pd.DataFrame()

            if "timestamp" in df.columns:
                df = df.sort("timestamp")

            if limit:
                df = df.head(limit)

            return df.to_pandas()
        except Exception:
            # Some files have mixed dtypes (e.g., market_id int vs float). Read per-file and cast.
            dfs = []
            total_rows = 0
            for f in files:
                df = pl.read_parquet(
                    str(f),
                    columns=list(columns) if columns else None,
                    missing_columns="insert",
                )

                # Normalize market_id dtype to avoid concat schema errors
                if "market_id" in df.columns:
                    df = df.with_columns(
                        pl.col("market_id").cast(pl.Float64, strict=False)
                    )

                # Apply USD filter
                if "usd_amount" in df.columns:
                    df = df.filter(pl.col("usd_amount") >= min_usd)

                # Apply date filters (more precise than file-level)
                if start_date and "timestamp" in df.columns:
                    df = df.filter(pl.col("timestamp").cast(pl.Utf8, strict=False) >= start_date)
                if end_date and "timestamp" in df.columns:
                    df = df.filter(pl.col("timestamp").cast(pl.Utf8, strict=False) <= end_date)

                if df.is_empty():
                    continue

                dfs.append(df)
                total_rows += df.height

                if limit and total_rows >= limit:
                    break

            if not dfs:
                return pd.DataFrame()

            df = pl.concat(dfs, how="vertical")
            if "timestamp" in df.columns:
                df = df.sort("timestamp")
            if limit:
                df = df.head(limit)
            return df.to_pandas()


class PriceStore:
    """Load price data from monthly partitioned parquet files."""

    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = Path(base_dir or f"{DEFAULT_PARQUET_DIR}/prices")
        self._cache = {}

    def available(self) -> bool:
        """Check if parquet price files exist."""
        return self.base_dir.exists() and any(self.base_dir.glob("prices_*.parquet"))

    def get_price_history(
        self,
        market_id: str,
        outcome: str = "YES",
    ) -> pd.DataFrame:
        """
        Load price history for a specific market and outcome.

        Args:
            market_id: Market identifier
            outcome: YES or NO

        Returns:
            DataFrame with columns [market_id, outcome, timestamp, price]
            sorted by timestamp
        """
        cache_key = (str(market_id), outcome)
        if cache_key in self._cache:
            return self._cache[cache_key]

        dfs = []
        for f in sorted(self.base_dir.glob("prices_*.parquet")):
            try:
                df = pd.read_parquet(
                    f,
                    filters=[
                        ("market_id", "==", str(market_id)),
                        ("outcome", "==", outcome),
                    ],
                )
                if not df.empty:
                    dfs.append(df)
            except Exception:
                # If predicate pushdown fails, read and filter manually
                df = pd.read_parquet(f)
                df = df[
                    (df["market_id"] == str(market_id))
                    & (df["outcome"] == outcome)
                ]
                if not df.empty:
                    dfs.append(df)

        if not dfs:
            result = pd.DataFrame(
                columns=["market_id", "outcome", "timestamp", "price"]
            )
        else:
            result = pd.concat(dfs, ignore_index=True)
            result["timestamp"] = pd.to_numeric(result["timestamp"], errors="coerce")
            result["price"] = pd.to_numeric(result["price"], errors="coerce")
            result = result.dropna(subset=["timestamp", "price"])
            result = result.sort_values("timestamp").reset_index(drop=True)

        self._cache[cache_key] = result
        return result

    def load_prices_for_markets(
        self,
        market_ids: List[str],
        outcome: str = "YES",
    ) -> pd.DataFrame:
        """
        Load prices for multiple markets at once.

        Args:
            market_ids: List of market identifiers
            outcome: YES or NO

        Returns:
            DataFrame with price data for all specified markets
        """
        market_ids_str = [str(m) for m in market_ids]
        dfs = []
        for f in sorted(self.base_dir.glob("prices_*.parquet")):
            try:
                df = pd.read_parquet(
                    f,
                    filters=[
                        ("market_id", "in", market_ids_str),
                        ("outcome", "==", outcome),
                    ],
                )
            except Exception:
                df = pd.read_parquet(f)
                df = df[
                    (df["market_id"].isin(market_ids_str))
                    & (df["outcome"] == outcome)
                ]
            if not df.empty:
                dfs.append(df)

        if not dfs:
            return pd.DataFrame()

        result = pd.concat(dfs, ignore_index=True)
        result["timestamp"] = pd.to_numeric(result["timestamp"], errors="coerce")
        result["price"] = pd.to_numeric(result["price"], errors="coerce")
        return result.sort_values("timestamp").reset_index(drop=True)

    def price_at_or_after(
        self,
        market_id: str,
        outcome: str,
        timestamp: int,
    ) -> Optional[dict]:
        """
        Get the price at or after a given timestamp.

        Args:
            market_id: Market identifier
            outcome: YES or NO
            timestamp: Unix timestamp (seconds)

        Returns:
            Dict with timestamp and price, or None
        """
        df = self.get_price_history(market_id, outcome)
        if df.empty:
            return None

        timestamps = df["timestamp"].to_numpy()
        idx = int(np.searchsorted(timestamps, timestamp, side="left"))
        if idx >= len(df):
            return None

        row = df.iloc[idx]
        return {
            "timestamp": int(row["timestamp"]),
            "price": float(row["price"]),
        }

    def last_price(
        self,
        market_id: str,
        outcome: str,
    ) -> Optional[dict]:
        """
        Get the last known price for a market/outcome.

        Args:
            market_id: Market identifier
            outcome: YES or NO

        Returns:
            Dict with timestamp and price, or None
        """
        df = self.get_price_history(market_id, outcome)
        if df.empty:
            return None
        row = df.iloc[-1]
        return {
            "timestamp": int(row["timestamp"]),
            "price": float(row["price"]),
        }

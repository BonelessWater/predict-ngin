"""
Cross-platform price store for arbitrage backtesting.

Provides unified price access across Polymarket and Kalshi, normalizing
timestamps and price formats for spread computation.

Both platforms store prices as 0.0-1.0 probabilities in our parquet files
(Kalshi raw API returns cents but the fetcher converts to 0-1).

Usage:
    from trading.arbitrage.cross_platform_price_store import CrossPlatformPriceStore

    store = CrossPlatformPriceStore(
        polymarket_prices_dir="data/polymarket/prices",
        kalshi_prices_dir="data/kalshi/prices",
    )
    spread = store.get_spread("poly_id", "KALSHI-TICKER", timestamp)
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..data_modules.parquet_store import PriceStore


# ---------------------------------------------------------------------------
# Kalshi price loader (reads kalshi/prices/ parquet files)
# ---------------------------------------------------------------------------

class KalshiPriceStore:
    """
    Load Kalshi candlestick prices from monthly parquet files.

    Kalshi parquet schema (written by fetcher):
        ticker, event_ticker, timestamp (unix seconds),
        open, high, low, close, mean, volume, open_interest

    This adapts to the PriceStore interface used elsewhere.
    """

    def __init__(self, base_dir: str = "data/kalshi/prices", cache_size: int = 200):
        self.base_dir = Path(base_dir)
        self._cache: OrderedDict[str, pd.DataFrame] = OrderedDict()
        self._cache_size = cache_size
        self._all_prices: Optional[pd.DataFrame] = None

    def available(self) -> bool:
        return self.base_dir.exists() and any(self.base_dir.glob("prices_*.parquet"))

    def _load_all(self) -> pd.DataFrame:
        """Lazy-load all Kalshi prices into memory (indexed by ticker)."""
        if self._all_prices is not None:
            return self._all_prices

        files = sorted(self.base_dir.glob("prices_*.parquet"))
        if not files:
            self._all_prices = pd.DataFrame()
            return self._all_prices

        dfs = []
        for f in files:
            try:
                df = pd.read_parquet(f)
                dfs.append(df)
            except Exception:
                continue

        if not dfs:
            self._all_prices = pd.DataFrame()
            return self._all_prices

        self._all_prices = pd.concat(dfs, ignore_index=True)

        # Normalize columns
        if "timestamp" in self._all_prices.columns:
            self._all_prices["timestamp"] = pd.to_numeric(
                self._all_prices["timestamp"], errors="coerce"
            )
        if "close" in self._all_prices.columns:
            self._all_prices["close"] = pd.to_numeric(
                self._all_prices["close"], errors="coerce"
            )

        self._all_prices = self._all_prices.dropna(subset=["timestamp"])
        self._all_prices = self._all_prices.sort_values(
            ["ticker", "timestamp"]
        ).reset_index(drop=True)
        return self._all_prices

    def get_price_history(self, ticker: str) -> pd.DataFrame:
        """
        Get OHLCV price history for a Kalshi market ticker.

        Returns DataFrame with columns:
            ticker, timestamp (unix seconds), open, high, low, close,
            mean, volume, open_interest
        """
        cache_key = ticker
        if cache_key in self._cache:
            self._cache.move_to_end(cache_key)
            return self._cache[cache_key].copy()

        all_prices = self._load_all()
        if all_prices.empty:
            result = pd.DataFrame()
        else:
            result = all_prices[all_prices["ticker"] == ticker].copy()
            result = result.sort_values("timestamp").reset_index(drop=True)

        if len(self._cache) >= self._cache_size:
            self._cache.popitem(last=False)
        self._cache[cache_key] = result
        return result.copy()

    def price_at_or_before(
        self, ticker: str, timestamp: int
    ) -> Optional[Dict[str, Any]]:
        """Get the Kalshi close price at or before a timestamp."""
        df = self.get_price_history(ticker)
        if df.empty:
            return None

        ts_array = df["timestamp"].to_numpy()
        idx = int(np.searchsorted(ts_array, timestamp, side="right")) - 1
        if idx < 0:
            return None

        row = df.iloc[idx]
        return {
            "timestamp": int(row["timestamp"]),
            "price": float(row["close"]),
        }

    def price_at_or_after(
        self, ticker: str, timestamp: int
    ) -> Optional[Dict[str, Any]]:
        """Get the Kalshi close price at or after a timestamp."""
        df = self.get_price_history(ticker)
        if df.empty:
            return None

        ts_array = df["timestamp"].to_numpy()
        idx = int(np.searchsorted(ts_array, timestamp, side="left"))
        if idx >= len(df):
            return None

        row = df.iloc[idx]
        return {
            "timestamp": int(row["timestamp"]),
            "price": float(row["close"]),
        }

    def last_price(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Get the last known Kalshi price."""
        df = self.get_price_history(ticker)
        if df.empty:
            return None
        row = df.iloc[-1]
        return {
            "timestamp": int(row["timestamp"]),
            "price": float(row["close"]),
        }


# ---------------------------------------------------------------------------
# Cross-platform unified store
# ---------------------------------------------------------------------------

class CrossPlatformPriceStore:
    """
    Unified price store spanning Polymarket and Kalshi.

    Normalizes both platforms to a common interface:
    - Prices are 0.0-1.0 (probability)
    - Timestamps are unix seconds
    - Outcome is YES (we track YES price; NO = 1 - YES)

    For spread computation, both platforms must have a price at or near
    the requested timestamp. We use the nearest available price within
    a configurable tolerance window.
    """

    def __init__(
        self,
        polymarket_prices_dir: str = "data/polymarket/prices",
        kalshi_prices_dir: str = "data/kalshi/prices",
        max_timestamp_gap: int = 86400,  # 24h tolerance for price alignment
        cache_size: int = 200,
    ):
        """
        Args:
            polymarket_prices_dir: Path to Polymarket monthly price parquet files.
            kalshi_prices_dir: Path to Kalshi monthly price parquet files.
            max_timestamp_gap: Maximum seconds between prices to consider aligned.
            cache_size: LRU cache size for each store.
        """
        self.poly_store = PriceStore(polymarket_prices_dir, cache_size=cache_size)
        self.kalshi_store = KalshiPriceStore(kalshi_prices_dir, cache_size=cache_size)
        self.max_timestamp_gap = max_timestamp_gap

    def available(self) -> Tuple[bool, bool]:
        """Check availability of each platform's data."""
        return self.poly_store.available(), self.kalshi_store.available()

    # -------------------------------------------------------------------
    # Per-platform access
    # -------------------------------------------------------------------

    def get_polymarket_price(
        self, market_id: str, timestamp: int, outcome: str = "YES"
    ) -> Optional[Dict[str, Any]]:
        """Get Polymarket price at or after timestamp."""
        return self.poly_store.price_at_or_after(market_id, outcome, timestamp)

    def get_kalshi_price(
        self, ticker: str, timestamp: int
    ) -> Optional[Dict[str, Any]]:
        """Get Kalshi YES price at or after timestamp."""
        return self.kalshi_store.price_at_or_after(ticker, timestamp)

    def get_polymarket_history(
        self, market_id: str, outcome: str = "YES"
    ) -> pd.DataFrame:
        """Full Polymarket price history for a market."""
        return self.poly_store.get_price_history(market_id, outcome)

    def get_kalshi_history(self, ticker: str) -> pd.DataFrame:
        """Full Kalshi OHLCV history for a market."""
        return self.kalshi_store.get_price_history(ticker)

    # -------------------------------------------------------------------
    # Cross-platform spread
    # -------------------------------------------------------------------

    def get_spread(
        self,
        poly_id: str,
        kalshi_ticker: str,
        timestamp: int,
        outcome: str = "YES",
    ) -> Optional[Dict[str, Any]]:
        """
        Compute the price spread at a given timestamp.

        spread = polymarket_yes_price - kalshi_yes_price

        Positive spread means Polymarket is more expensive (buy Kalshi, sell Poly).
        Negative spread means Kalshi is more expensive (buy Poly, sell Kalshi).

        Returns:
            Dict with poly_price, kalshi_price, spread, abs_spread, timestamp,
            or None if prices unavailable within tolerance.
        """
        poly_point = self.poly_store.price_at_or_after(poly_id, outcome, timestamp)
        kalshi_point = self.kalshi_store.price_at_or_after(kalshi_ticker, timestamp)

        if poly_point is None or kalshi_point is None:
            return None

        # Check time alignment
        gap = abs(poly_point["timestamp"] - kalshi_point["timestamp"])
        if gap > self.max_timestamp_gap:
            return None

        poly_price = float(poly_point["price"])
        kalshi_price = float(kalshi_point["price"])
        spread = poly_price - kalshi_price

        return {
            "poly_price": poly_price,
            "kalshi_price": kalshi_price,
            "spread": spread,
            "abs_spread": abs(spread),
            "poly_timestamp": poly_point["timestamp"],
            "kalshi_timestamp": kalshi_point["timestamp"],
            "timestamp": max(poly_point["timestamp"], kalshi_point["timestamp"]),
            "gap_seconds": gap,
        }

    def build_spread_series(
        self,
        poly_id: str,
        kalshi_ticker: str,
        outcome: str = "YES",
        resample_freq: str = "1D",
    ) -> pd.DataFrame:
        """
        Build a time-aligned spread series for a matched pair.

        Resamples both price series to a common frequency and computes
        the spread at each point.

        Args:
            poly_id: Polymarket market ID
            kalshi_ticker: Kalshi market ticker
            outcome: YES or NO
            resample_freq: Pandas frequency string (default: daily)

        Returns:
            DataFrame with columns: timestamp, poly_price, kalshi_price,
            spread, abs_spread
        """
        poly_hist = self.get_polymarket_history(poly_id, outcome)
        kalshi_hist = self.get_kalshi_history(kalshi_ticker)

        if poly_hist.empty or kalshi_hist.empty:
            return pd.DataFrame()

        # Convert Polymarket timestamps to datetime
        if not pd.api.types.is_datetime64_any_dtype(poly_hist["timestamp"]):
            poly_hist["datetime"] = pd.to_datetime(
                poly_hist["timestamp"], unit="s", errors="coerce", utc=True
            )
        else:
            poly_hist["datetime"] = poly_hist["timestamp"]

        # Convert Kalshi timestamps
        kalshi_hist["datetime"] = pd.to_datetime(
            kalshi_hist["timestamp"], unit="s", errors="coerce", utc=True
        )

        # Resample to common frequency (forward-fill)
        poly_resampled = (
            poly_hist.set_index("datetime")["price"]
            .resample(resample_freq)
            .last()
            .ffill()
        )
        kalshi_resampled = (
            kalshi_hist.set_index("datetime")["close"]
            .resample(resample_freq)
            .last()
            .ffill()
        )

        # Align on overlapping dates
        aligned = pd.DataFrame({
            "poly_price": poly_resampled,
            "kalshi_price": kalshi_resampled,
        }).dropna()

        if aligned.empty:
            return pd.DataFrame()

        aligned["spread"] = aligned["poly_price"] - aligned["kalshi_price"]
        aligned["abs_spread"] = aligned["spread"].abs()
        aligned = aligned.reset_index().rename(columns={"datetime": "timestamp"})
        aligned["timestamp_unix"] = (
            aligned["timestamp"].astype("int64") // 10**9
        )

        return aligned

    def get_all_spreads_at(
        self,
        pairs: List[Tuple[str, str]],
        timestamp: int,
        outcome: str = "YES",
    ) -> pd.DataFrame:
        """
        Get spreads for multiple pairs at a single timestamp.

        Args:
            pairs: List of (polymarket_id, kalshi_ticker) tuples
            timestamp: Unix timestamp
            outcome: YES or NO

        Returns:
            DataFrame with one row per pair.
        """
        rows = []
        for poly_id, kalshi_ticker in pairs:
            spread_data = self.get_spread(poly_id, kalshi_ticker, timestamp, outcome)
            if spread_data is not None:
                rows.append({
                    "polymarket_id": poly_id,
                    "kalshi_ticker": kalshi_ticker,
                    **spread_data,
                })

        return pd.DataFrame(rows) if rows else pd.DataFrame()

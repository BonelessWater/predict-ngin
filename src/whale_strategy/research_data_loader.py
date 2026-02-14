"""
Load trades, prices, and market metadata from data/research by category.

Supports category-level whale strategy backtesting using the canonical
data/research structure: {category}/trades.parquet, prices.parquet, markets_filtered.csv.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

# Blacklisted categories per strategy spec (low signal quality)
BLACKLISTED_CATEGORIES = {"sports_entertainment", "celebrity_gossip"}


def _normalize_category(name: str) -> str:
    """Normalize category name for blacklist check."""
    return name.lower().replace(" ", "_").replace("-", "_")


def get_research_categories(research_dir: Path) -> List[str]:
    """List category directories that have trades.parquet."""
    if not research_dir.exists():
        return []
    cats = []
    for d in research_dir.iterdir():
        if d.is_dir() and (d / "trades.parquet").exists():
            norm = _normalize_category(d.name)
            if norm not in BLACKLISTED_CATEGORIES:
                cats.append(d.name)
    return sorted(cats)


def load_research_trades(
    research_dir: Path,
    categories: Optional[List[str]] = None,
    min_usd: float = 10.0,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load trades from data/research/{category}/trades.parquet.

    Normalizes columns to backtest format:
    - market_id, datetime, price, usd_amount
    - maker (= proxyWallet), maker_direction (= side)
    - taker (= proxyWallet), taker_direction (= opposite of side for CLOB semantics)
    - category

    For whale following we use maker/maker_direction (proxyWallet + side) as the
    active trader and their direction.
    """
    research_dir = Path(research_dir)
    if categories is None:
        categories = get_research_categories(research_dir)
    if not categories:
        return pd.DataFrame()

    dfs = []
    for cat in categories:
        p = research_dir / cat / "trades.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)

        # Normalize user column
        if "proxyWallet" not in df.columns or df["proxyWallet"].isna().all():
            user = df.get("taker", df.get("maker", pd.Series([""] * len(df)))).fillna("").astype(str)
        else:
            user = df["proxyWallet"].fillna("").astype(str)

        df = df[user.str.startswith("0x", na=False)].copy()
        if df.empty:
            continue

        # USD amount
        p_col = pd.to_numeric(df.get("price", 0), errors="coerce")
        s_col = pd.to_numeric(df.get("size", 0), errors="coerce")
        fallback_usd = p_col * s_col
        if "usd_amount" in df.columns:
            df["usd_amount"] = pd.to_numeric(df["usd_amount"], errors="coerce").fillna(fallback_usd)
        else:
            df["usd_amount"] = fallback_usd
        df["usd_amount"] = df["usd_amount"].fillna(0).clip(lower=0)

        # Filter by min_usd
        df = df[df["usd_amount"] >= min_usd]

        # Side/direction
        side = df.get("side", "").fillna("").astype(str).str.upper()
        if side.isin(["BUY", "SELL"]).sum() == 0 and "outcomeIndex" in df.columns:
            # outcomeIndex 0 = YES, 1 = NO; BUY YES, SELL NO etc.
            side = np.where(df["outcomeIndex"] == 0, "BUY", "SELL")
            side = pd.Series(side, index=df.index).astype(str)
        df["maker_direction"] = side
        df["taker_direction"] = side.map({"BUY": "SELL", "SELL": "BUY"})

        # Backtest columns (use conditionId for resolution matching when available)
        df["maker"] = user
        df["taker"] = user
        if "conditionId" in df.columns and df["conditionId"].notna().any():
            df["market_id"] = df["conditionId"].astype(str).str.strip()
        else:
            df["market_id"] = df.get("market_id", "").astype(str).str.strip()
        df["price"] = pd.to_numeric(df.get("price", 0), errors="coerce")
        df["datetime"] = pd.to_datetime(
            pd.to_numeric(df.get("timestamp", 0), errors="coerce"), unit="s", errors="coerce"
        )
        df["category"] = cat

        # Token amount for PnL
        if "token_amount" not in df.columns and "size" in df.columns:
            df["token_amount"] = pd.to_numeric(df["size"], errors="coerce")
        elif "token_amount" not in df.columns:
            df["token_amount"] = df["usd_amount"] / df["price"].replace(0, np.nan)

        df = df.dropna(subset=["datetime", "market_id", "price"])
        if df.empty:
            continue

        # Date filter
        if start_date:
            df = df[df["datetime"] >= start_date]
        if end_date:
            df = df[df["datetime"] <= end_date]

        dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    out = pd.concat(dfs, ignore_index=True)
    out = out.sort_values("datetime").reset_index(drop=True)
    return out


def load_research_markets(research_dir: Path, categories: Optional[List[str]] = None) -> pd.DataFrame:
    """Load markets_filtered.csv from each category, add category column."""
    research_dir = Path(research_dir)
    if categories is None:
        categories = get_research_categories(research_dir)

    dfs = []
    for cat in categories:
        p = research_dir / cat / "markets_filtered.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p, low_memory=False)
        if df.empty or "conditionId" not in df.columns:
            continue
        df = df.copy()
        df["market_id"] = df["conditionId"].astype(str).str.strip()
        df["category"] = cat
        dfs.append(df)

    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def load_research_prices(
    research_dir: Path,
    categories: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Load prices from data/research/{category}/prices.parquet.

    Returns dict mapping market_id -> DataFrame with columns [timestamp, price].
    """
    research_dir = Path(research_dir)
    if categories is None:
        categories = get_research_categories(research_dir)

    all_prices: Dict[str, List[Dict]] = {}
    for cat in categories:
        p = research_dir / cat / "prices.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        if df.empty or "market_id" not in df.columns:
            continue
        for mid, g in df.groupby("market_id"):
            mid_str = str(mid).strip()
            g = g.sort_values("timestamp")
            if mid_str not in all_prices:
                all_prices[mid_str] = []
            for _, row in g.iterrows():
                all_prices[mid_str].append({
                    "timestamp": row.get("timestamp"),
                    "price": row.get("price"),
                })

    # Dedupe and sort per market
    out = {}
    for mid, rows in all_prices.items():
        if not rows:
            continue
        df = pd.DataFrame(rows)
        df = df.dropna(subset=["timestamp", "price"])
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df = df.drop_duplicates(subset=["timestamp"], keep="last")
        df = df.sort_values("timestamp")
        out[mid] = df

    return out


class ResearchPriceStore:
    """
    Price store that reads from data/research/{category}/prices.parquet.

    Implements the interface expected by polymarket_realistic_backtest
    (price_at_or_before, get_price_history).
    """

    def __init__(self, research_dir: Path, categories: Optional[List[str]] = None):
        self.research_dir = Path(research_dir)
        self.categories = categories or get_research_categories(self.research_dir)
        self._price_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._load_prices()

    def _load_prices(self) -> None:
        """Load all price data into cache."""
        for cat in self.categories:
            p = self.research_dir / cat / "prices.parquet"
            if not p.exists():
                continue
            df = pd.read_parquet(p)
            if df.empty or "market_id" not in df.columns:
                continue
            for mid, g in df.groupby("market_id"):
                mid_str = str(mid).strip().replace(".0", "")
                g = g.sort_values("timestamp")
                ts = pd.to_numeric(g["timestamp"], errors="coerce").dropna().astype("int64").to_numpy()
                pr = pd.to_numeric(g["price"], errors="coerce").dropna().astype("float64").to_numpy()
                if len(ts) > 0 and len(pr) > 0:
                    if mid_str in self._price_cache:
                        # Merge if market appears in multiple categories
                        old_ts, old_pr = self._price_cache[mid_str]
                        ts = np.concatenate([old_ts, ts])
                        pr = np.concatenate([old_pr, pr])
                        idx = np.argsort(ts)
                        ts = ts[idx]
                        pr = pr[idx]
                    self._price_cache[mid_str] = (ts, pr)

    def get_price_history(self, market_id: str, outcome: str = "YES") -> pd.DataFrame:
        """Return price history DataFrame for backtest compatibility."""
        mid = str(market_id).strip().replace(".0", "")
        if mid not in self._price_cache:
            return pd.DataFrame(columns=["timestamp", "price"])
        ts, pr = self._price_cache[mid]
        return pd.DataFrame({"timestamp": ts, "price": pr})

    def price_at_or_before(self, market_id: str, ts: pd.Timestamp) -> Optional[float]:
        """Get price at or before given timestamp (for backtest exit pricing)."""
        mid = str(market_id).strip().replace(".0", "")
        if mid not in self._price_cache:
            return None
        timestamps, prices = self._price_cache[mid]
        ts_end = int(pd.Timestamp(ts).replace(hour=23, minute=59, second=59).timestamp())
        idx = int(np.searchsorted(timestamps, ts_end, side="right")) - 1
        if idx < 0:
            return None
        return float(prices[idx])

    def close(self) -> None:
        """No-op for compatibility with ClobPriceStore."""
        pass


def load_resolution_winners(
    research_dir: Path,
    db_path: Optional[str] = None,
) -> Dict[str, str]:
    """
    Load resolution winners: market_id -> "YES" or "NO".

    Tries in order:
    1. data/research/resolutions.csv (columns: market_id, winner)
    2. SQLite polymarket_resolutions table (if db_path exists)
    """
    research_dir = Path(research_dir)
    winners: Dict[str, str] = {}

    # 1. CSV
    csv_path = research_dir / "resolutions.csv"
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            if "market_id" in df.columns and "winner" in df.columns:
                for _, row in df.iterrows():
                    mid = str(row["market_id"]).strip().replace(".0", "")
                    w = str(row["winner"]).strip().upper()
                    if w in ("YES", "NO"):
                        winners[mid] = w
                return winners
        except Exception:
            pass

    # 2. DB fallback
    if db_path:
        try:
            from .polymarket_whales import load_resolution_winners as db_load
            return db_load(db_path=db_path)
        except Exception:
            pass

    return winners

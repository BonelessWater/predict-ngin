"""
Data loading utilities for Manifold Markets prediction market data.
"""

import json
import glob
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional


def load_manifold_data(data_dir: str = "data/manifold", auto_fetch: bool = True) -> pd.DataFrame:
    """
    Load all Manifold bets from JSON files.

    Args:
        data_dir: Directory containing bets_*.json files
        auto_fetch: If True, fetch data from API if not found locally

    Returns:
        DataFrame with all bets, sorted by datetime
    """
    bets = []
    bet_files = sorted(glob.glob(f"{data_dir}/bets_*.json"))

    # Auto-fetch if no data exists
    if not bet_files and auto_fetch:
        print("No local data found. Fetching from Manifold API...")
        from .fetcher import DataFetcher
        fetcher = DataFetcher(str(Path(data_dir).parent))
        fetcher.fetch_manifold_bets()
        fetcher.fetch_manifold_markets()
        bet_files = sorted(glob.glob(f"{data_dir}/bets_*.json"))

    for f in bet_files:
        try:
            with open(f, encoding="utf-8") as file:
                bets.extend(json.load(file))
        except (json.JSONDecodeError, IOError):
            continue

    if not bets:
        raise ValueError(f"No bets found in {data_dir}. Run with auto_fetch=True or use DataFetcher.")

    df = pd.DataFrame(bets)
    df["datetime"] = pd.to_datetime(df["createdTime"], unit="ms")
    df["date"] = df["datetime"].dt.date
    df["month"] = df["datetime"].dt.to_period("M")
    df["amount_abs"] = df["amount"].abs()
    df = df.sort_values("datetime").reset_index(drop=True)

    return df


def load_markets(data_dir: str = "data/manifold") -> pd.DataFrame:
    """
    Load all Manifold markets from JSON files.

    Args:
        data_dir: Directory containing markets_*.json files

    Returns:
        DataFrame with all markets
    """
    markets = []

    for f in glob.glob(f"{data_dir}/markets_*.json"):
        try:
            with open(f, encoding="utf-8") as file:
                markets.extend(json.load(file))
        except (json.JSONDecodeError, IOError):
            continue

    return pd.DataFrame(markets)


def build_resolution_map(markets_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Build a mapping from market ID to resolution data.

    Args:
        markets_df: DataFrame of markets

    Returns:
        Dictionary mapping market ID to resolution info
    """
    resolution_data = {}

    for _, m in markets_df.iterrows():
        mid = m["id"]
        if m.get("isResolved") and m.get("resolution") in ["YES", "NO"]:
            resolution_data[mid] = {
                "resolution": 1.0 if m["resolution"] == "YES" else 0.0,
                "resolved_time": m.get("resolutionTime", m.get("closeTime", 0)),
                "question": str(m.get("question", "")),
                "liquidity": m.get("totalLiquidity", 1000),
                "volume": m.get("volume", 0),
            }

    return resolution_data


def train_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.3
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data temporally into train and test sets.

    Args:
        df: DataFrame with datetime column
        train_ratio: Fraction of data for training (default 30%)

    Returns:
        Tuple of (train_df, test_df)
    """
    split_date = df["datetime"].quantile(train_ratio)
    train_df = df[df["datetime"] <= split_date]
    test_df = df[df["datetime"] > split_date]

    return train_df, test_df

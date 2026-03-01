"""
Whale identification and surprise win rate analysis.

Whales = traders with:
  - Capital >= $50k (estimated as 5x max position, rolling)
  - OR 95th percentile of traders by volume in the market (rolling, no look-ahead)

Surprise win rate = actual win rate - expected win rate
  - Expected: market-implied (e.g. YES @ 50c = 50%)
  - Actual: from resolution
"""

from typing import Dict, Optional, Set, Tuple

import numpy as np
import pandas as pd

MIN_CAPITAL_WHALE = 50_000
CAPITAL_MULTIPLIER = 5  # Estimate capital as 5x max position


def _is_whale_at_trade(
    trader: str,
    market_id: str,
    trade_idx: int,
    trades_df: pd.DataFrame,
    trader_max_position: Dict[str, float],
    market_trader_volumes: Dict[str, Dict[str, float]],
    volume_only: bool = False,
    volume_percentile: float = 95.0,
) -> bool:
    """
    Check if trader is a whale at trade time (no look-ahead).
    Uses only data from trades before trade_idx.
    """
    if not volume_only:
        # Capital criterion: 5x max position >= 50k
        max_pos = trader_max_position.get(trader, 0)
        if max_pos * CAPITAL_MULTIPLIER >= MIN_CAPITAL_WHALE:
            return True

    # Volume percentile: in top N% for this market
    mvol = market_trader_volumes.get(market_id, {})
    if not mvol:
        return False
    vol_list = sorted(mvol.values(), reverse=True)
    if len(vol_list) < 5:  # Need enough traders for percentile
        return False
    threshold = np.percentile(vol_list, volume_percentile)
    return mvol.get(trader, 0) >= threshold


def identify_whales_rolling(
    trades_df: pd.DataFrame,
    trader_col: str = "maker",
    volume_only: bool = False,
    volume_percentile: float = 95.0,
) -> pd.DataFrame:
    """
    Mark each trade with is_whale (vectorized, no look-ahead).

    Whale = capital >= 50k OR 95th percentile volume in market.
    If volume_only=True, only 95th percentile volume (no capital criterion).

    Capital criterion: causal via rolling cummax shifted by 1 within each
    trader group — uses only trades that occurred before the current trade.

    Volume criterion: compares each trade's cumulative volume-to-date against
    the per-market Nth-percentile of final cumulative volumes (a valid
    approximation for fixed training windows that avoids O(N²) rolling
    percentile computation).
    """
    df = trades_df.sort_values("datetime").reset_index(drop=True).copy()
    df["_mid"] = (
        df["market_id"].astype(str).str.strip().str.replace(".0", "", regex=False)
    )

    # --- Capital criterion (vectorized, causal) ---
    if not volume_only:
        # Running max of position size per trader, including current row
        df["_rmax"] = df.groupby(trader_col)["usd_amount"].cummax()
        # Shift within each trader group to get max BEFORE this trade
        prev_max = df.groupby(trader_col)["_rmax"].shift(1).fillna(0.0)
        is_capital_whale = prev_max * CAPITAL_MULTIPLIER >= MIN_CAPITAL_WHALE
        df.drop(columns=["_rmax"], inplace=True)
    else:
        is_capital_whale = pd.Series(False, index=df.index)

    # --- Volume percentile criterion (vectorized) ---
    # Cumulative volume per (market, trader) BEFORE current trade
    cum_vol = df.groupby(["_mid", trader_col])["usd_amount"].cumsum()
    df["_cum_vol_before"] = cum_vol - df["usd_amount"]

    # Final cumulative volumes per (market, trader) — used to derive threshold
    final_vols = (
        df.groupby(["_mid", trader_col])["usd_amount"].sum().reset_index()
    )
    final_vols.columns = ["_mid", trader_col, "_total_vol"]

    n_traders_map = final_vols.groupby("_mid")[trader_col].count()
    threshold_map = (
        final_vols.groupby("_mid")["_total_vol"]
        .quantile(volume_percentile / 100.0)
    )

    df["_n_traders"] = df["_mid"].map(n_traders_map)
    df["_vol_threshold"] = df["_mid"].map(threshold_map)

    is_vol_whale = (
        (df["_n_traders"] >= 5) &
        (df["_cum_vol_before"] >= df["_vol_threshold"])
    )

    df["is_whale"] = is_capital_whale | is_vol_whale
    df.drop(
        columns=["_mid", "_cum_vol_before", "_n_traders", "_vol_threshold"],
        inplace=True,
    )
    return df


def calculate_surprise_metrics(
    whale_trades: pd.DataFrame,
    resolution_winners: Dict[str, str],
    direction_col: str = "maker_direction",
) -> Dict:
    """
    Expected WR = price for BUY YES, 1-price for SELL NO.
    Actual WR = fraction of correct predictions.
    Surprise WR = actual - expected.
    """
    df = whale_trades.copy()
    df["market_id_str"] = df["market_id"].astype(str).str.replace(".0", "", regex=False)
    df["winner"] = df["market_id_str"].map(resolution_winners)
    df = df.dropna(subset=["winner"])
    if len(df) < 5:
        return {
            "expected_win_rate": np.nan,
            "actual_win_rate": np.nan,
            "surprise_win_rate": np.nan,
            "sample_size": len(df),
        }

    direction = df[direction_col].str.lower()
    price = df["price"].astype(float)
    winner = df["winner"].str.upper()

    # Expected: BUY YES @ p -> expected WR = p; SELL NO @ p -> expected WR = 1-p
    expected = np.where(
        direction == "buy",
        price,
        1 - price,
    )
    df["expected_wr"] = expected

    # Actual: correct = (BUY & YES) or (SELL & NO)
    correct = (
        ((direction == "buy") & (winner == "YES")) |
        ((direction == "sell") & (winner == "NO"))
    )
    df["correct"] = correct

    return {
        "expected_win_rate": float(expected.mean()),
        "actual_win_rate": float(correct.mean()),
        "surprise_win_rate": float(correct.mean() - expected.mean()),
        "sample_size": len(df),
    }


def _filter_unfavored_trades(
    df: pd.DataFrame,
    direction_col: str = "maker_direction",
    max_price: float = 0.40,
) -> pd.DataFrame:
    """
    Keep only unfavored (underdog) trades: BUY at <= max_price, SELL at >= (1-max_price).
    E.g. max_price=0.40: BUY YES at 40c or less, SELL NO at 60c+ (short favorite).
    """
    direction = df[direction_col].str.lower()
    price = df["price"].astype(float)
    mask = (
        ((direction == "buy") & (price <= max_price)) |
        ((direction == "sell") & (price >= (1 - max_price)))
    )
    return df[mask].copy()


def calculate_performance_score_with_surprise(
    whale_address: str,
    trades_df: pd.DataFrame,
    resolution_winners: Dict[str, str],
    direction_col: str = "maker_direction",
    min_trades: int = 10,
    unfavored_only: bool = False,
    unfavored_max_price: float = 0.40,
) -> Optional[Dict]:
    """
    Score whales based on beating market expectations (surprise), not raw win rate.
    If unfavored_only=True, only include trades at <= unfavored_max_price (BUY) or
    >= (1-unfavored_max_price) (SELL) - i.e. underdog positions.
    """
    whale_trades = trades_df[
        (trades_df["maker"] == whale_address) &
        (trades_df["market_id"].astype(str).isin(resolution_winners.keys()))
    ].copy()
    if unfavored_only:
        whale_trades = _filter_unfavored_trades(
            whale_trades, direction_col, unfavored_max_price
        )
    if len(whale_trades) < min_trades:
        return None

    surprise_metrics = calculate_surprise_metrics(
        whale_trades, resolution_winners, direction_col
    )
    expected_wr = surprise_metrics["expected_win_rate"]
    actual_wr = surprise_metrics["actual_win_rate"]
    surprise_wr = surprise_metrics["surprise_win_rate"]

    # ROI and Sharpe from resolved trades
    direction = whale_trades[direction_col].str.lower()
    price = whale_trades["price"].astype(float)
    winner = whale_trades["market_id"].astype(str).map(resolution_winners)
    resolution = winner.map({"YES": 1.0, "NO": 0.0})
    roi = np.where(
        direction == "buy",
        (resolution - price) / np.maximum(price, 0.01),
        (price - resolution) / np.maximum(1 - price, 0.01),
    )
    avg_roi = float(np.mean(roi))
    sharpe = float(np.mean(roi) / (np.std(roi) + 1e-6))

    # Surprise score: +15% surprise = 10, 0% = 5, -15% = 0
    surprise_score = min(max(surprise_wr / 0.15 * 5 + 5, 0), 10)
    roi_score = min(avg_roi * 10, 10)
    sharpe_score = min(max(sharpe, 0) * 2, 10)

    raw_score = 0.50 * surprise_score + 0.30 * roi_score + 0.20 * sharpe_score

    return {
        "score": min(float(raw_score), 10),
        "expected_win_rate": expected_wr,
        "actual_win_rate": actual_wr,
        "surprise_win_rate": surprise_wr,
        "avg_roi": avg_roi,
        "sharpe": sharpe,
        "sample_size": len(whale_trades),
    }


def build_surprise_positive_whale_set(
    train_trades: pd.DataFrame,
    resolution_winners: Dict[str, str],
    min_surprise: float = 0.0,
    min_trades: int = 10,
    min_actual_win_rate: float = 0.50,
    require_positive_surprise: bool = True,
    direction_col: str = "maker_direction",
    trader_col: str = "maker",
    volume_percentile: float = 95.0,
) -> Tuple[Set[str], Dict[str, float], Dict[str, float]]:
    """
    Build set of volume whales filtered by WR >= min_actual_win_rate and surprise > min_surprise.

    Returns:
        whale_set: Set of whale addresses meeting performance bar
        whale_scores: Dict[whale_address, score] for filter_and_score_signals
        whale_winrates: Dict[whale_address, actual_win_rate] for Kelly sizing
    """
    trades_with_whale = identify_whales_rolling(
        train_trades,
        trader_col=trader_col,
        volume_only=True,
        volume_percentile=volume_percentile,
    )
    whale_trades = trades_with_whale[trades_with_whale["is_whale"]]

    if whale_trades.empty:
        return set(), {}, {}

    whale_addresses = set(whale_trades[trader_col].unique())

    # Vectorized scoring for all whale addresses at once.
    # Use train_trades (not just whale_trades) to match the original per-address
    # behavior which scores based on all resolved trades by each address.
    resolved_ids = set(resolution_winners.keys())
    wr = train_trades[
        train_trades[trader_col].isin(whale_addresses) &
        train_trades["market_id"].astype(str).isin(resolved_ids)
    ].copy()

    if wr.empty:
        return set(), {}, {}

    wr["_mid"] = wr["market_id"].astype(str).str.replace(".0", "", regex=False)
    wr["_winner"] = wr["_mid"].map(resolution_winners)
    wr = wr.dropna(subset=["_winner"])
    if wr.empty:
        return set(), {}, {}

    dir_lower = wr[direction_col].str.lower()
    price = wr["price"].astype(float)
    winner_up = wr["_winner"].str.upper()

    wr["_expected_wr"] = np.where(dir_lower == "buy", price, 1.0 - price)
    wr["_correct"] = (
        ((dir_lower == "buy") & (winner_up == "YES")) |
        ((dir_lower == "sell") & (winner_up == "NO"))
    )
    resolution_val = winner_up.map({"YES": 1.0, "NO": 0.0})
    wr["_roi"] = np.where(
        dir_lower == "buy",
        (resolution_val - price) / np.maximum(price, 0.01),
        (price - resolution_val) / np.maximum(1.0 - price, 0.01),
    )

    # Aggregate per whale
    stats = wr.groupby(trader_col).agg(
        sample_size=(trader_col, "count"),
        actual_win_rate=("_correct", "mean"),
        expected_win_rate=("_expected_wr", "mean"),
        avg_roi=("_roi", "mean"),
    ).reset_index()

    # Population std (ddof=0) to match original np.std behaviour
    roi_std = wr.groupby(trader_col)["_roi"].std(ddof=0).rename("roi_std")
    stats = stats.join(roi_std, on=trader_col)
    stats["roi_std"] = stats["roi_std"].fillna(0.0)

    stats["surprise_win_rate"] = stats["actual_win_rate"] - stats["expected_win_rate"]
    stats["sharpe"] = stats["avg_roi"] / (stats["roi_std"] + 1e-6)

    # Apply filters
    stats = stats[stats["sample_size"] >= min_trades]
    stats = stats[stats["actual_win_rate"] >= min_actual_win_rate]
    if require_positive_surprise:
        stats = stats[stats["surprise_win_rate"] > min_surprise]
    else:
        stats = stats[stats["surprise_win_rate"] >= min_surprise]

    if stats.empty:
        return set(), {}, {}

    # Compute composite scores (mirrors calculate_performance_score_with_surprise)
    surprise_score = (stats["surprise_win_rate"] / 0.15 * 5.0 + 5.0).clip(0.0, 10.0)
    roi_score = (stats["avg_roi"] * 10.0).clip(upper=10.0)
    sharpe_score = stats["sharpe"].clip(lower=0.0) * 2.0
    stats["score"] = (
        0.50 * surprise_score + 0.30 * roi_score + 0.20 * sharpe_score
    ).clip(upper=10.0)

    whale_set: Set[str] = set(stats[trader_col])
    whale_scores: Dict[str, float] = stats.set_index(trader_col)["score"].to_dict()
    whale_winrates: Dict[str, float] = (
        stats.set_index(trader_col)["actual_win_rate"].to_dict()
    )
    return whale_set, whale_scores, whale_winrates


def build_volume_whale_set(
    trades_df: pd.DataFrame,
    trader_col: str = "maker",
    volume_percentile: float = 95.0,
    default_score: float = 7.0,
    default_winrate: float = 0.5,
) -> Tuple[Set[str], Dict[str, float], Dict[str, float]]:
    """
    Build whale set from Nth percentile volume only (no capital, no resolution).

    Returns whale_set, whale_scores (default), whale_winrates (default).
    Use when resolution data is unavailable or when filtering by resolved markets is not desired.
    """
    trades_with_whale = identify_whales_rolling(
        trades_df,
        trader_col=trader_col,
        volume_only=True,
        volume_percentile=volume_percentile,
    )
    whale_trades = trades_with_whale[trades_with_whale["is_whale"]]
    whale_set = set(whale_trades[trader_col].unique())
    whale_scores = {addr: default_score for addr in whale_set}
    whale_winrates = {addr: default_winrate for addr in whale_set}
    return whale_set, whale_scores, whale_winrates

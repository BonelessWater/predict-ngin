"""
Whale-following strategy with category-level limits and "last signal wins" logic.

Implements:
- Whale quality scoring (W)
- Category-level exposure limits (30% max per category)
- Position sizing with fractional Kelly
- Conflicting signals: close prior, enter new (last signal wins)
- Tier allocation (Tier 1/2/3 by score)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any

import numpy as np
import pandas as pd

from .whale_scoring import (
    calculate_whale_score,
    qualifies_as_whale_signal,
    WHALE_CRITERIA,
    MIN_WHALE_SCORE,
)

# Risk limits from spec
RISK_LIMITS = {
    "max_single_position_pct": 0.05,   # 5% of capital
    "max_single_whale_pct": 0.20,      # 20% to any whale
    "max_single_market_pct": 0.08,    # 8% to any market
    "max_category_pct": 0.30,           # 30% to any category
    "min_reserve_pct": 0.10,           # 10% cash reserve
    "max_position_usd": 50_000,
    "min_position_usd": 5_000,
}

# Tier allocation
TIER_TARGETS = {
    "TIER_1": 0.40,   # W >= 8.0, 40% of capital
    "TIER_2": 0.35,   # 7.0 <= W < 8.0
    "TIER_3": 0.15,   # 6.0 <= W < 7.0
}
TIER_MAX_POSITION = {"TIER_1": 50_000, "TIER_2": 30_000, "TIER_3": 20_000}


def _classify_tier(whale_score: float) -> str:
    if whale_score >= 8.0:
        return "TIER_1"
    elif whale_score >= 7.0:
        return "TIER_2"
    elif whale_score >= 6.0:
        return "TIER_3"
    return "NONE"


def _estimate_win_probability(
    whale_score: float,
    market_price: float,
    historical_whale_winrate: float,
) -> float:
    """Blend whale historical accuracy with market-implied probability."""
    p_market = market_price
    p_whale = historical_whale_winrate
    confidence = whale_score / 10.0
    return confidence * p_whale + (1 - confidence) * p_market


def _kelly_fraction(
    p: float,
    entry_price: float,
    side: str,
    kelly_mult: float = 0.25,
) -> float:
    """Fractional Kelly: f* = (1/4) * [(p*b - q) / b]."""
    q = 1 - p
    if side.upper() == "BUY":
        b = (1 - entry_price) / max(entry_price, 0.01)  # net odds for YES
    else:
        b = entry_price / max(1 - entry_price, 0.01)    # net odds for NO
    if b <= 0:
        return 0.0
    f = (p * b - q) / b
    return max(0.0, min(kelly_mult * f, 1.0))


@dataclass
class WhaleSignal:
    """Qualified whale signal."""
    market_id: str
    category: str
    whale_address: str
    side: str  # BUY or SELL
    price: float
    size_usd: float
    score: float
    datetime: pd.Timestamp
    historical_winrate: float = 0.5


@dataclass
class Position:
    """Open position."""
    market_id: str
    category: str
    side: str
    entry_price: float
    size_usd: float
    whale_address: str
    whale_score: float
    entry_date: pd.Timestamp


@dataclass
class StrategyState:
    """Current portfolio state for limit checks."""
    positions: List[Position] = field(default_factory=list)
    total_capital: float = 1_000_000
    category_exposure: Dict[str, float] = field(default_factory=dict)
    whale_exposure: Dict[str, float] = field(default_factory=dict)
    market_exposure: Dict[str, float] = field(default_factory=dict)
    tier_exposure: Dict[str, float] = field(default_factory=dict)

    def deployed(self) -> float:
        return sum(p.size_usd for p in self.positions)

    def available(self) -> float:
        reserve = self.total_capital * RISK_LIMITS["min_reserve_pct"]
        return max(0, self.total_capital - self.deployed() - reserve)

    def category_exposure_pct(self, category: str) -> float:
        return self.category_exposure.get(category, 0) / self.total_capital

    def whale_exposure_pct(self, whale: str) -> float:
        return self.whale_exposure.get(whale, 0) / self.total_capital

    def market_exposure_pct(self, market_id: str) -> float:
        return self.market_exposure.get(market_id, 0) / self.total_capital


def calculate_position_size(
    signal: WhaleSignal,
    state: StrategyState,
    market_liquidity: float,
) -> Optional[float]:
    """
    Calculate position size with Kelly + constraints.

    Returns None if trade should be skipped.
    """
    # Base Kelly size
    p = _estimate_win_probability(
        signal.score, signal.price, signal.historical_winrate,
    )
    side = "BUY" if signal.side.upper() == "BUY" else "SELL"
    kelly_frac = _kelly_fraction(p, signal.price, side)
    base_size = kelly_frac * state.available()

    # Hard caps
    size = min(
        base_size,
        RISK_LIMITS["max_position_usd"],
        market_liquidity * 0.03,
        TIER_MAX_POSITION.get(_classify_tier(signal.score), 20_000),
    )

    if size < RISK_LIMITS["min_position_usd"]:
        return None

    # Category limit
    cat_pct = state.category_exposure_pct(signal.category)
    if cat_pct >= RISK_LIMITS["max_category_pct"]:
        size *= 0.5
    if cat_pct >= RISK_LIMITS["max_category_pct"] * 1.2:
        return None  # Over limit, skip

    # Market limit
    if state.market_exposure_pct(signal.market_id) >= RISK_LIMITS["max_single_market_pct"]:
        return None

    # Whale limit
    if state.whale_exposure_pct(signal.whale_address) >= RISK_LIMITS["max_single_whale_pct"]:
        size *= 0.5

    # Tier overallocation
    tier = _classify_tier(signal.score)
    tier_pct = state.tier_exposure.get(tier, 0) / state.total_capital
    target = TIER_TARGETS.get(tier, 0.15)
    if tier_pct > target * 1.1:
        size *= target / max(tier_pct, 0.01)

    return min(size, state.available())


def find_conflicting_position(
    signal: WhaleSignal,
    positions: List[Position],
) -> Optional[Position]:
    """Find existing position in same market on opposite side."""
    for pos in positions:
        if pos.market_id == signal.market_id and pos.side != signal.side:
            return pos
    return None


def handle_conflicting_signal(
    new_signal: WhaleSignal,
    existing: Position,
    state: StrategyState,
) -> Dict[str, Any]:
    """
    Last-signal-wins: close existing, return action for new entry.

    Returns dict with: action='CLOSE_AND_FLIP', close_position, new_signal.
    """
    return {
        "action": "CLOSE_AND_FLIP",
        "close_position": existing,
        "new_signal": new_signal,
        "reason": "CONFLICTING_WHALE_SIGNAL",
    }


def filter_and_score_signals(
    trades_df: pd.DataFrame,
    resolution_winners: Dict[str, str],
    markets_df: pd.DataFrame,
    whale_set: Optional[Set[str]] = None,
    min_usd: float = 50_000,
    role: str = "maker",
    min_position_size_override: Optional[float] = None,
    min_market_liquidity_override: Optional[float] = None,
    whale_scores_override: Optional[Dict[str, float]] = None,
    whale_winrates_override: Optional[Dict[str, float]] = None,
) -> List[WhaleSignal]:
    """
    Filter trades to qualified whale signals with scores.

    If whale_set is None, uses all traders that pass min criteria.
    If whale_scores_override is provided, uses those scores instead of calculate_whale_score.
    If whale_winrates_override is provided, uses those for historical_winrate (for Kelly sizing).
    """
    trader_col = role
    direction_col = f"{role}_direction"
    min_pos = (
        min_position_size_override
        if min_position_size_override is not None
        else WHALE_CRITERIA["min_position_size"]
    )
    min_liq = (
        min_market_liquidity_override
        if min_market_liquidity_override is not None
        else WHALE_CRITERIA["min_market_liquidity"]
    )
    size_threshold = max(min_usd, min_pos)

    # --- Build market metadata (vectorized) ---
    market_meta: dict = {}
    market_liquidity: dict = {}
    if not markets_df.empty and "market_id" in markets_df.columns:
        mdf = markets_df.copy()
        mdf["_mid"] = mdf["market_id"].astype(str).str.strip()
        market_meta = mdf.set_index("_mid").to_dict("index")
        liq_col = next(
            (c for c in ["liquidityNum", "liquidity", "volume", "volumeNum"]
             if c in mdf.columns),
            None,
        )
        if liq_col:
            liq_series = pd.to_numeric(mdf[liq_col], errors="coerce").fillna(100_000)
            market_liquidity = dict(zip(mdf["_mid"], liq_series))

    # --- Pre-filter trades (vectorized) ---
    mask = trades_df["usd_amount"] >= size_threshold
    if whale_set is not None:
        mask &= trades_df[trader_col].isin(whale_set)
    filtered = trades_df[mask].copy()
    if filtered.empty:
        return []

    filtered["_mid"] = (
        filtered["market_id"].astype(str).str.strip().str.replace(".0", "", regex=False)
    )

    # Apply market liquidity filter (vectorized)
    if market_liquidity and min_liq > 0:
        filtered["_liq"] = filtered["_mid"].map(market_liquidity).fillna(100_000)
        filtered = filtered[filtered["_liq"] >= min_liq]
        if filtered.empty:
            return []

    # --- Assign scores (vectorized when override provided) ---
    if whale_scores_override is not None:
        filtered["_score"] = filtered[trader_col].map(whale_scores_override).fillna(0.0)
    else:
        filtered["_score"] = filtered.apply(
            lambda row: calculate_whale_score(
                row, trades_df, resolution_winners,
                market_meta.get(row["_mid"], {}), role,
            ),
            axis=1,
        )

    filtered = filtered[filtered["_score"] >= MIN_WHALE_SCORE]
    if filtered.empty:
        return []

    # --- Precompute whale win rates (vectorized when not overridden) ---
    if whale_winrates_override is not None:
        whale_winrates: dict = whale_winrates_override
    else:
        whale_winrates = {}
        resolved_ids = set(resolution_winners.keys())
        res_mask = trades_df["market_id"].astype(str).isin(resolved_ids)
        if res_mask.any():
            rt = trades_df[res_mask].copy()
            rt["_winner"] = rt["market_id"].astype(str).map(resolution_winners)
            rt = rt.dropna(subset=["_winner"])
            if not rt.empty:
                dir_l = rt[direction_col].str.lower()
                w_up = rt["_winner"].str.upper()
                rt["_profitable"] = (
                    ((dir_l == "buy") & (w_up == "YES")) |
                    ((dir_l == "sell") & (w_up == "NO"))
                )
                wr_stats = rt.groupby(trader_col)["_profitable"].agg(["mean", "count"])
                whale_winrates = wr_stats.loc[wr_stats["count"] >= 5, "mean"].to_dict()

    # --- Build signals from the small pre-filtered set ---
    signals = []
    for _, row in filtered.iterrows():
        mid = row["_mid"]
        ttr = None
        meta = market_meta.get(mid, {})
        if meta:
            end = meta.get("endDateIso") or meta.get("endDate") or meta.get("closedTime")
            if end:
                try:
                    ttr = (pd.to_datetime(end) - row["datetime"]).days
                except Exception:
                    pass
        if ttr is not None and ttr > WHALE_CRITERIA["max_time_to_resolution"]:
            continue

        winrate = whale_winrates.get(row[trader_col], 0.5)
        signals.append(WhaleSignal(
            market_id=mid,
            category=row.get("category", "Unknown"),
            whale_address=row[trader_col],
            side=row[direction_col].upper() if pd.notna(row[direction_col]) else "BUY",
            price=float(row["price"]),
            size_usd=float(row["usd_amount"]),
            score=row["_score"],
            datetime=row["datetime"],
            historical_winrate=winrate,
        ))

    return signals

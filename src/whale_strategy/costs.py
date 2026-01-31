"""
Polymarket cost model with market impact.

Implements square-root market impact model (Almgren-Chriss style)
with explicit assumptions for different position sizes.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict


@dataclass
class CostAssumptions:
    """Cost model assumptions for a position size category."""

    name: str
    base_spread: float  # Bid-ask spread
    base_slippage: float  # Base slippage
    impact_coefficient: float  # Market impact multiplier
    liquidity_threshold: float  # Minimum liquidity to trade
    description: str


COST_ASSUMPTIONS = {
    "small": CostAssumptions(
        name="Small Retail ($100-500)",
        base_spread=0.02,
        base_slippage=0.005,
        impact_coefficient=0.5,
        liquidity_threshold=500,
        description="Retail-sized orders with minimal market footprint",
    ),
    "medium": CostAssumptions(
        name="Medium ($1,000-5,000)",
        base_spread=0.025,
        base_slippage=0.01,
        impact_coefficient=1.0,
        liquidity_threshold=5000,
        description="Mid-sized orders requiring moderate liquidity",
    ),
    "large": CostAssumptions(
        name="Large ($10,000-20,000)",
        base_spread=0.03,
        base_slippage=0.02,
        impact_coefficient=2.0,
        liquidity_threshold=25000,
        description="Large orders with significant market footprint",
    ),
}


class CostModel:
    """
    Trading cost model for prediction markets.

    Uses square-root market impact model:
        Total Cost = Base Spread/2 + Base Slippage + Size Impact

    Where:
        Size Impact = impact_coef * base_slippage * sqrt(trade_size / liquidity)
    """

    def __init__(
        self,
        base_spread: float = 0.02,
        base_slippage: float = 0.005,
        impact_coefficient: float = 0.5,
        max_impact: float = 0.15,
    ):
        """
        Initialize cost model.

        Args:
            base_spread: Bid-ask spread (default 2%)
            base_slippage: Base slippage (default 0.5%)
            impact_coefficient: Market impact multiplier
            max_impact: Maximum allowed impact (default 15%)
        """
        self.base_spread = base_spread
        self.base_slippage = base_slippage
        self.impact_coefficient = impact_coefficient
        self.max_impact = max_impact

    @classmethod
    def from_assumptions(cls, category: str) -> "CostModel":
        """Create model from predefined assumptions."""
        if category not in COST_ASSUMPTIONS:
            raise ValueError(f"Unknown category: {category}")

        assumptions = COST_ASSUMPTIONS[category]
        return cls(
            base_spread=assumptions.base_spread,
            base_slippage=assumptions.base_slippage,
            impact_coefficient=assumptions.impact_coefficient,
        )

    def calculate_one_way_cost(
        self,
        trade_size: float,
        liquidity: float
    ) -> float:
        """
        Calculate one-way trading cost.

        Args:
            trade_size: Size of the trade in dollars
            liquidity: Market liquidity in dollars

        Returns:
            One-way cost as decimal (e.g., 0.02 = 2%)
        """
        if liquidity <= 0:
            liquidity = 1000  # Default

        # Square-root impact model
        size_impact = (
            self.impact_coefficient *
            self.base_slippage *
            np.sqrt(trade_size / liquidity)
        )
        size_impact = min(size_impact, self.max_impact)

        return self.base_spread / 2 + self.base_slippage + size_impact

    def calculate_round_trip_cost(
        self,
        trade_size: float,
        liquidity: float
    ) -> float:
        """Calculate round-trip (entry + exit) cost."""
        return 2 * self.calculate_one_way_cost(trade_size, liquidity)

    def calculate_entry_price(
        self,
        base_price: float,
        trade_size: float,
        liquidity: float
    ) -> float:
        """Calculate effective entry price after costs."""
        cost = self.calculate_one_way_cost(trade_size, liquidity)
        return base_price * (1 + cost)

    def calculate_exit_price(
        self,
        base_price: float,
        trade_size: float,
        liquidity: float
    ) -> float:
        """Calculate effective exit price after costs."""
        cost = self.calculate_one_way_cost(trade_size, liquidity)
        return base_price * (1 - cost)

    def calculate_break_even_win_rate(
        self,
        trade_size: float,
        liquidity: float
    ) -> float:
        """
        Calculate break-even win rate for binary markets.

        For binary markets with symmetric payoffs:
            break_even = 0.5 + round_trip_cost / 2
        """
        round_trip = self.calculate_round_trip_cost(trade_size, liquidity)
        return 0.5 + round_trip / 2


# Default cost model (small retail)
DEFAULT_COST_MODEL = CostModel()

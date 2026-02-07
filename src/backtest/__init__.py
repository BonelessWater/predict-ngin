"""
Backtest module for strategy evaluation.

Provides backtesting engines and validation tools.
"""

from .engine import BacktestEngine, TradingEngine, BacktestResult
from .costs import CostModel, COST_ASSUMPTIONS
from .walk_forward import WalkForwardValidator, WalkForwardResult, WalkForwardWindow

# Re-export portfolio for convenience
from trading.portfolio import (
    PortfolioConstraints,
    PortfolioState,
    PositionSizer,
)

__all__ = [
    "BacktestEngine",
    "TradingEngine",
    "BacktestResult",
    "CostModel",
    "COST_ASSUMPTIONS",
    "WalkForwardValidator",
    "WalkForwardResult",
    "WalkForwardWindow",
    "PortfolioConstraints",
    "PortfolioState",
    "PositionSizer",
]

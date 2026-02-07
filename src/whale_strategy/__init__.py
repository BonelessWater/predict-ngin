"""
Whale Following Strategy for Prediction Markets

A bias-free backtesting framework for whale-following strategies on prediction markets.
Supports multiple whale identification methods, market categorization, and position sizing analysis.
"""

from .strategy import identify_whales, WHALE_METHODS, identify_whales_rolling
from .backtest import (
    run_backtest,
    BacktestResult,
    print_result,
    run_position_size_analysis,
    print_position_size_analysis,
    run_rolling_backtest,
)
from .polymarket_whales import (
    load_polymarket_trades,
    calculate_trader_stats,
    identify_polymarket_whales,
    generate_whale_signals,
    run_polymarket_whale_analysis,
    POLY_WHALE_METHODS,
)

__version__ = "1.0.0"
__all__ = [
    "identify_whales",
    "identify_whales_rolling",
    "WHALE_METHODS",
    "run_backtest",
    "BacktestResult",
    "print_result",
    "run_position_size_analysis",
    "print_position_size_analysis",
    "run_rolling_backtest",
    "load_polymarket_trades",
    "calculate_trader_stats",
    "identify_polymarket_whales",
    "generate_whale_signals",
    "run_polymarket_whale_analysis",
    "POLY_WHALE_METHODS",
]

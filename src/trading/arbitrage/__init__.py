"""
Cross-platform arbitrage infrastructure for prediction markets.

Matches equivalent markets across Polymarket and Kalshi, detects
price discrepancies, and backtests arbitrage strategies.
"""

from .market_matcher import MarketMatcher, MatchedPair
from .cross_platform_price_store import CrossPlatformPriceStore
from .arbitrage_strategy import ArbitrageStrategy, ArbitrageSignal
from .cross_platform_backtest import (
    ArbitrageBacktestConfig,
    ArbitrageBacktestResult,
    run_arbitrage_backtest,
    print_arbitrage_result,
)

__all__ = [
    "MarketMatcher",
    "MatchedPair",
    "CrossPlatformPriceStore",
    "ArbitrageStrategy",
    "ArbitrageSignal",
    "ArbitrageBacktestConfig",
    "ArbitrageBacktestResult",
    "run_arbitrage_backtest",
    "print_arbitrage_result",
]

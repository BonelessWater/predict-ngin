"""
Whale Following Strategy for Prediction Markets

A bias-free backtesting framework for whale-following strategies on prediction markets.
Supports multiple whale identification methods, market categorization, and position sizing analysis.
"""

from .data import load_manifold_data, load_markets, build_resolution_map
from .fetcher import DataFetcher, ensure_data_exists
from .whales import identify_whales, WHALE_METHODS
from .backtest import run_backtest, BacktestResult, print_result
from .costs import CostModel, COST_ASSUMPTIONS
from .categories import categorize_market, CATEGORIES
from .reporting import generate_quantstats_report, generate_all_reports

__version__ = "1.0.0"
__all__ = [
    "load_manifold_data",
    "load_markets",
    "build_resolution_map",
    "DataFetcher",
    "ensure_data_exists",
    "identify_whales",
    "WHALE_METHODS",
    "run_backtest",
    "BacktestResult",
    "print_result",
    "CostModel",
    "COST_ASSUMPTIONS",
    "categorize_market",
    "CATEGORIES",
    "generate_quantstats_report",
    "generate_all_reports",
]

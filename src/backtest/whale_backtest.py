"""
Whale Strategy Backtest - Core implementation.
Tests whale-following strategy with multiple detection methods.

Usage:
    python src/backtest/whale_backtest.py              # All markets
    python src/backtest/whale_backtest.py --xrp        # XRP only
    python src/backtest/whale_backtest.py --contracts  # Per-contract breakdown
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import json
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.config import get_config, Config


class Position(Enum):
    NONE = 0
    LONG = 1
    SHORT = 2


@dataclass
class Trade:
    """Record of a single trade."""
    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    side: str  # 'long' or 'short'
    size: float
    pnl: float
    pnl_pct: float
    costs: float
    market: str
    method: str
    consensus: float


@dataclass
class ContractResult:
    """Results for a single contract/market."""
    market: str
    trades: int
    wins: int
    losses: int
    total_pnl: float
    total_costs: float
    win_rate: float
    avg_pnl: float
    max_win: float
    max_loss: float


@dataclass
class BacktestResult:
    """Complete backtest results."""
    method: str
    initial_capital: float
    final_capital: float
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate: float
    num_trades: int
    total_costs: float
    trades: List[Trade]
    equity_curve: List[dict]
    contract_results: List[ContractResult]


def load_trade_data(xrp_only: bool = False) -> pd.DataFrame:
    """Load collected trade data with trader information."""
    config = get_config()

    # Use bulk data which has full trade details
    bulk_dir = config.data.parquet_dir / 'bulk'

    df = pd.DataFrame()

    # Load main bulk file
    main_files = list(bulk_dir.glob('all_trades_*.parquet'))
    if main_files:
        df = pd.read_parquet(sorted(main_files)[-1])

    if df.empty:
        return df

    # Deduplicate
    if 'transactionHash' in df.columns:
        df = df.drop_duplicates(subset=['transactionHash'])

    # Parse timestamp
    if 'timestamp' in df.columns:
        df['timestamp_dt'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')

    # Filter for XRP if requested
    if xrp_only:
        xrp_mask = (
            df['title'].str.lower().str.contains('xrp', na=False) |
            df['slug'].str.lower().str.contains('xrp', na=False)
        )
        df = df[xrp_mask]

    return df.sort_values('timestamp_dt') if 'timestamp_dt' in df.columns else df


def detect_whales(trades: pd.DataFrame, method: str, config: Config) -> set:
    """
    Detect whales using specified method.

    Methods:
        - top_n: Top N traders by volume
        - percentile: Traders above X percentile
        - trade_size: Traders with avg trade > threshold
    """
    if trades.empty:
        return set()

    trader_col = 'proxyWallet' if 'proxyWallet' in trades.columns else 'trader_id'
    size_col = 'size' if 'size' in trades.columns else 'volume'

    if trader_col not in trades.columns or size_col not in trades.columns:
        return set()

    # Aggregate by trader
    trader_stats = trades.groupby(trader_col).agg({
        size_col: ['sum', 'mean', 'count']
    })
    trader_stats.columns = ['total_volume', 'avg_trade', 'trade_count']
    trader_stats = trader_stats.reset_index()

    if method == 'top_n':
        # Top N by volume
        n = config.whale.top_n
        whales = set(trader_stats.nlargest(n, 'total_volume')[trader_col])

    elif method == 'percentile':
        # Above percentile threshold
        pct = config.whale.percentile
        threshold = trader_stats['total_volume'].quantile(pct / 100)
        whales = set(trader_stats[trader_stats['total_volume'] >= threshold][trader_col])

    elif method == 'trade_size':
        # Average trade size above threshold
        min_size = config.whale.min_trade_size
        whales = set(trader_stats[trader_stats['avg_trade'] >= min_size][trader_col])

    else:
        whales = set()

    return whales


def calculate_whale_consensus(
    trades: pd.DataFrame,
    whales: set,
    current_time: datetime,
    lookback_hours: int = 24
) -> Tuple[float, str]:
    """
    Calculate whale consensus (buy vs sell) over lookback period.
    Returns (consensus_strength, direction).
    """
    if not whales or trades.empty:
        return 0.0, 'neutral'

    trader_col = 'proxyWallet' if 'proxyWallet' in trades.columns else 'trader_id'
    size_col = 'size' if 'size' in trades.columns else 'volume'

    # Filter to lookback period
    lookback_start = current_time - timedelta(hours=lookback_hours)
    mask = (trades['timestamp_dt'] >= lookback_start) & (trades['timestamp_dt'] <= current_time)
    recent = trades[mask]

    # Filter to whale trades
    whale_trades = recent[recent[trader_col].isin(whales)]

    if whale_trades.empty:
        return 0.0, 'neutral'

    # Calculate buy vs sell volume
    if 'side' in whale_trades.columns:
        buys = whale_trades[whale_trades['side'].str.upper() == 'BUY'][size_col].sum()
        sells = whale_trades[whale_trades['side'].str.upper() == 'SELL'][size_col].sum()
    else:
        # Infer from price movement or default to balanced
        buys = whale_trades[size_col].sum() / 2
        sells = whale_trades[size_col].sum() / 2

    total = buys + sells
    if total == 0:
        return 0.0, 'neutral'

    buy_pct = buys / total
    sell_pct = sells / total

    if buy_pct > sell_pct:
        return buy_pct, 'buy'
    elif sell_pct > buy_pct:
        return sell_pct, 'sell'
    else:
        return 0.5, 'neutral'


def calculate_costs(trade_size: float, config: Config) -> float:
    """Calculate transaction costs (fees + slippage)."""
    fee = trade_size * config.costs.taker_fee
    slippage = config.costs.calculate_slippage(trade_size, True) * trade_size
    return fee + slippage


def run_backtest(
    trades: pd.DataFrame,
    method: str,
    config: Config
) -> BacktestResult:
    """
    Run whale-following backtest on trade data.
    """
    empty_result = BacktestResult(
        method=method,
        initial_capital=config.strategy.initial_capital,
        final_capital=config.strategy.initial_capital,
        total_return_pct=0, sharpe_ratio=0, max_drawdown_pct=0,
        win_rate=0, num_trades=0, total_costs=0,
        trades=[], equity_curve=[], contract_results=[]
    )

    if trades.empty or 'price' not in trades.columns:
        return empty_result

    # Get market column
    market_col = 'title' if 'title' in trades.columns else 'market_slug'

    # Group trades by market for per-contract analysis
    markets = trades[market_col].unique() if market_col in trades.columns else ['all']

    # Create hourly price series per market
    trades['hour'] = trades['timestamp_dt'].dt.floor('h')

    all_results = []

    for market in markets:
        market_trades = trades[trades[market_col] == market] if market_col in trades.columns else trades

        if len(market_trades) < 10:  # Skip markets with few trades
            continue

        # Get price series for this market
        price_series = market_trades.groupby('hour').agg({
            'price': 'mean',
            'size': 'sum'
        }).reset_index()
        price_series = price_series.sort_values('hour')

        if len(price_series) < 5:
            continue

        # Run backtest for this market
        result = run_single_market_backtest(
            market_trades, price_series, market, method, config
        )
        if result:
            all_results.append(result)

    # Combine results
    if not all_results:
        return empty_result

    # Aggregate all trades
    all_trades_list = []
    total_costs = 0
    combined_equity = []

    for r in all_results:
        all_trades_list.extend(r['trades'])
        total_costs += r['costs']

    # Calculate aggregate metrics
    total_pnl = sum(t.pnl for t in all_trades_list)
    final_capital = config.strategy.initial_capital + total_pnl

    wins = len([t for t in all_trades_list if t.pnl > 0])
    win_rate = wins / len(all_trades_list) * 100 if all_trades_list else 0

    total_return_pct = total_pnl / config.strategy.initial_capital * 100

    # Contract results
    contract_results = calculate_contract_results(all_trades_list)

    return BacktestResult(
        method=method,
        initial_capital=config.strategy.initial_capital,
        final_capital=final_capital,
        total_return_pct=total_return_pct,
        sharpe_ratio=0,  # Would need proper equity curve
        max_drawdown_pct=0,
        win_rate=win_rate,
        num_trades=len(all_trades_list),
        total_costs=total_costs,
        trades=all_trades_list,
        equity_curve=[],
        contract_results=contract_results
    )


def run_single_market_backtest(
    trades: pd.DataFrame,
    prices: pd.DataFrame,
    market: str,
    method: str,
    config: Config
) -> Optional[dict]:
    """Run backtest for a single market/contract."""

    if prices.empty or len(prices) < 3:
        return None

    capital = config.strategy.initial_capital
    position = Position.NONE
    position_size = 0
    entry_price = 0
    entry_date = None

    market_trades = []
    total_costs = 0

    # Skip first few periods for warmup
    start_idx = min(3, len(prices) - 1)

    for i in range(start_idx, len(prices)):
        row = prices.iloc[i]
        current_date = row['hour']
        current_price = row['price']

        # Get historical trades up to now (no look-ahead)
        hist_trades = trades[trades['timestamp_dt'] <= current_date]

        if len(hist_trades) < 5:
            continue

        # Detect whales
        whales = detect_whales(hist_trades, method, config)

        if not whales:
            continue

        # Calculate consensus
        consensus, direction = calculate_whale_consensus(
            hist_trades, whales, current_date, lookback_hours=6
        )

        threshold = config.strategy.consensus_threshold

        # Trading logic
        if position == Position.NONE:
            if consensus >= threshold and direction == 'buy':
                # Enter long
                trade_val = capital * config.strategy.position_size_pct
                costs = calculate_costs(trade_val, config)
                total_costs += costs

                position_size = (trade_val - costs) / current_price
                entry_price = current_price
                entry_date = current_date
                position = Position.LONG

        elif position == Position.LONG:
            # Exit on sell signal or consensus drop
            should_exit = (direction == 'sell' and consensus >= threshold) or \
                         (consensus < threshold * 0.8)

            if should_exit:
                exit_value = position_size * current_price
                costs = calculate_costs(exit_value, config)
                total_costs += costs

                pnl = (current_price - entry_price) * position_size - costs
                pnl_pct = pnl / (position_size * entry_price) * 100

                market_trades.append(Trade(
                    entry_date=entry_date,
                    exit_date=current_date,
                    entry_price=entry_price,
                    exit_price=current_price,
                    side='long',
                    size=position_size * entry_price,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    costs=costs,
                    market=market,
                    method=method,
                    consensus=consensus
                ))

                position = Position.NONE
                position_size = 0

    # Close open position at end
    if position == Position.LONG and len(prices) > 0:
        final_price = prices.iloc[-1]['price']
        pnl = (final_price - entry_price) * position_size
        market_trades.append(Trade(
            entry_date=entry_date,
            exit_date=prices.iloc[-1]['hour'],
            entry_price=entry_price,
            exit_price=final_price,
            side='long',
            size=position_size * entry_price,
            pnl=pnl,
            pnl_pct=pnl / (position_size * entry_price) * 100 if position_size > 0 else 0,
            costs=0,
            market=market,
            method=method,
            consensus=0
        ))

    if not market_trades:
        return None

    return {
        'market': market,
        'trades': market_trades,
        'costs': total_costs
    }


def calculate_contract_results(trades: List[Trade]) -> List[ContractResult]:
    """Calculate per-contract performance."""
    if not trades:
        return []

    # Group by market
    by_market = {}
    for t in trades:
        market = t.market
        if market not in by_market:
            by_market[market] = []
        by_market[market].append(t)

    results = []
    for market, market_trades in by_market.items():
        wins = len([t for t in market_trades if t.pnl > 0])
        losses = len([t for t in market_trades if t.pnl <= 0])
        total_pnl = sum(t.pnl for t in market_trades)
        total_costs = sum(t.costs for t in market_trades)

        pnls = [t.pnl for t in market_trades]

        results.append(ContractResult(
            market=market,
            trades=len(market_trades),
            wins=wins,
            losses=losses,
            total_pnl=total_pnl,
            total_costs=total_costs,
            win_rate=wins / len(market_trades) * 100 if market_trades else 0,
            avg_pnl=np.mean(pnls) if pnls else 0,
            max_win=max(pnls) if pnls else 0,
            max_loss=min(pnls) if pnls else 0
        ))

    return sorted(results, key=lambda x: -x.total_pnl)


def print_results(result: BacktestResult, show_contracts: bool = False):
    """Print backtest results."""
    print(f"\n{'='*60}")
    print(f"WHALE STRATEGY: {result.method.upper()}")
    print(f"{'='*60}")
    print(f"Initial Capital:  ${result.initial_capital:,.2f}")
    print(f"Final Capital:    ${result.final_capital:,.2f}")
    print(f"Total Return:     {result.total_return_pct:+.2f}%")
    print(f"Sharpe Ratio:     {result.sharpe_ratio:.2f}")
    print(f"Max Drawdown:     {result.max_drawdown_pct:.2f}%")
    print(f"Win Rate:         {result.win_rate:.1f}%")
    print(f"Total Trades:     {result.num_trades}")
    print(f"Total Costs:      ${result.total_costs:,.2f}")

    if show_contracts and result.contract_results:
        print(f"\n{'='*60}")
        print("PER-CONTRACT PERFORMANCE")
        print(f"{'='*60}")
        print(f"{'Market':<40} {'Trades':>6} {'Win%':>6} {'PnL':>12}")
        print("-"*60)
        for cr in result.contract_results:
            market_short = cr.market[:38] + '..' if len(cr.market) > 40 else cr.market
            print(f"{market_short:<40} {cr.trades:>6} {cr.win_rate:>5.1f}% ${cr.total_pnl:>10,.2f}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Whale Strategy Backtest')
    parser.add_argument('--xrp', action='store_true', help='XRP markets only')
    parser.add_argument('--contracts', action='store_true', help='Show per-contract breakdown')
    parser.add_argument('--method', choices=['top_n', 'percentile', 'trade_size', 'all'],
                       default='all', help='Whale detection method')

    args = parser.parse_args()
    config = get_config()

    print("="*60)
    print("POLYMARKET WHALE STRATEGY BACKTEST")
    print("="*60)

    # Load data
    print(f"\nLoading {'XRP' if args.xrp else 'all'} market data...")
    trades = load_trade_data(xrp_only=args.xrp)

    if trades.empty:
        print("No trade data found!")
        print("Run: python src/data/historical_collector.py")
        return

    print(f"Loaded {len(trades):,} data points")
    if 'timestamp_dt' in trades.columns:
        print(f"Date range: {trades['timestamp_dt'].min()} to {trades['timestamp_dt'].max()}")

    # Run backtests
    methods = ['top_n', 'percentile', 'trade_size'] if args.method == 'all' else [args.method]

    results = []
    for method in methods:
        print(f"\nRunning {method} backtest...")
        result = run_backtest(trades, method, config)
        results.append(result)
        print_results(result, show_contracts=args.contracts)

    # Summary comparison
    if len(results) > 1:
        print(f"\n{'='*60}")
        print("METHOD COMPARISON")
        print(f"{'='*60}")
        print(f"{'Method':<15} {'Return':>10} {'Sharpe':>8} {'Win%':>8} {'Trades':>8}")
        print("-"*60)
        for r in sorted(results, key=lambda x: -x.total_return_pct):
            print(f"{r.method:<15} {r.total_return_pct:>+9.2f}% {r.sharpe_ratio:>8.2f} {r.win_rate:>7.1f}% {r.num_trades:>8}")

    # Save results
    output_dir = config.data.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    for result in results:
        # Save equity curve
        if result.equity_curve:
            eq_df = pd.DataFrame(result.equity_curve)
            suffix = '_xrp' if args.xrp else ''
            eq_df.to_csv(output_dir / f'equity_{result.method}{suffix}.csv', index=False)

        # Save contract results
        if result.contract_results:
            cr_df = pd.DataFrame([{
                'market': cr.market,
                'trades': cr.trades,
                'wins': cr.wins,
                'losses': cr.losses,
                'win_rate': cr.win_rate,
                'total_pnl': cr.total_pnl,
                'avg_pnl': cr.avg_pnl
            } for cr in result.contract_results])
            cr_df.to_csv(output_dir / f'contracts_{result.method}{suffix}.csv', index=False)

    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()

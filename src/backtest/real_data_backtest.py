"""
Backtest using real Polymarket data.
Adjusts whale detection thresholds to match actual Polymarket trading patterns.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import PARQUET_DIR, CONSENSUS_THRESHOLD

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Adjusted thresholds for real Polymarket data
REAL_WHALE_TOP_N = 10  # Top 10 by volume
REAL_WHALE_PERCENTILE = 95  # 95th percentile
REAL_WHALE_MIN_TRADE_SIZE = 100  # $100 per trade (Polymarket has smaller trades)
REAL_INITIAL_CAPITAL = 1000  # Start with $1000


def detect_whales_real_data(trades_df: pd.DataFrame, method: str = 'top_n') -> set:
    """
    Detect whales in real Polymarket data.

    Methods:
    - top_n: Top 10 traders by total volume
    - percentile: Traders above 95th percentile
    - trade_size: Traders with avg trade size > $100
    """
    if trades_df.empty:
        return set()

    # Aggregate by trader
    trader_stats = trades_df.groupby('proxyWallet').agg({
        'size': ['sum', 'mean', 'count']
    }).reset_index()
    trader_stats.columns = ['trader', 'total_volume', 'avg_trade_size', 'trade_count']

    if method == 'top_n':
        # Top N by total volume
        top_traders = trader_stats.nlargest(REAL_WHALE_TOP_N, 'total_volume')
        whales = set(top_traders['trader'].tolist())
        logger.info(f"Top {REAL_WHALE_TOP_N} method: {len(whales)} whales")

    elif method == 'percentile':
        # 95th percentile by volume
        threshold = np.percentile(trader_stats['total_volume'], REAL_WHALE_PERCENTILE)
        whales = set(trader_stats[trader_stats['total_volume'] >= threshold]['trader'].tolist())
        logger.info(f"Percentile method ({REAL_WHALE_PERCENTILE}%): {len(whales)} whales (threshold: ${threshold:.2f})")

    elif method == 'trade_size':
        # Average trade size > threshold
        whales = set(trader_stats[trader_stats['avg_trade_size'] >= REAL_WHALE_MIN_TRADE_SIZE]['trader'].tolist())
        logger.info(f"Trade size method (>${REAL_WHALE_MIN_TRADE_SIZE}): {len(whales)} whales")

    else:
        whales = set()

    return whales


def calculate_whale_consensus(trades_df: pd.DataFrame, whales: set) -> tuple:
    """
    Calculate consensus direction among whales.
    Returns (consensus_ratio, direction)
    """
    if not whales or trades_df.empty:
        return 0.0, 'neutral'

    # Filter to whale trades
    whale_trades = trades_df[trades_df['proxyWallet'].isin(whales)].copy()

    if whale_trades.empty:
        return 0.0, 'neutral'

    # Weight by volume
    buys = whale_trades[whale_trades['side'] == 'BUY']['size'].sum()
    sells = whale_trades[whale_trades['side'] == 'SELL']['size'].sum()
    total = buys + sells

    if total == 0:
        return 0.0, 'neutral'

    buy_ratio = buys / total
    sell_ratio = sells / total

    if buy_ratio > sell_ratio:
        return buy_ratio, 'buy'
    else:
        return sell_ratio, 'sell'


def run_real_backtest(trades_df: pd.DataFrame, method: str = 'top_n'):
    """
    Run backtest on real Polymarket data.

    Since we have limited time range, we'll:
    1. Split data into time windows
    2. Use earlier window to identify whales
    3. Trade in later window based on whale consensus
    """
    if trades_df.empty:
        logger.error("No trades data!")
        return None

    # Parse timestamp
    if 'ts' not in trades_df.columns:
        trades_df['ts'] = pd.to_datetime(trades_df['timestamp'], unit='s')

    trades_df = trades_df.sort_values('ts')

    # Get time range
    start_time = trades_df['ts'].min()
    end_time = trades_df['ts'].max()
    total_duration = (end_time - start_time).total_seconds()

    logger.info(f"Data range: {start_time} to {end_time} ({total_duration/60:.1f} minutes)")

    if total_duration < 300:  # Less than 5 minutes
        logger.warning("Very short time range - limited backtest value")

    # Split into windows (e.g., 1-minute windows)
    window_size = timedelta(minutes=1)
    current_time = start_time

    capital = REAL_INITIAL_CAPITAL
    position = None  # None, 'long', or 'short'
    entry_price = 0
    position_size = 0

    trades_log = []

    while current_time < end_time:
        window_end = current_time + window_size

        # Get trades in current window
        window_trades = trades_df[(trades_df['ts'] >= current_time) & (trades_df['ts'] < window_end)]

        if not window_trades.empty:
            # Detect whales from all prior trades
            prior_trades = trades_df[trades_df['ts'] < window_end]
            whales = detect_whales_real_data(prior_trades, method)

            # Calculate consensus
            consensus, direction = calculate_whale_consensus(window_trades, whales)

            # Current "price" is the average trade price in this window
            current_price = window_trades['price'].mean()

            # Trading logic
            if position is None:
                # Check for entry
                if consensus >= CONSENSUS_THRESHOLD:
                    if direction == 'buy':
                        # Enter long
                        position = 'long'
                        position_size = capital / current_price
                        entry_price = current_price
                        trades_log.append({
                            'time': current_time,
                            'action': 'buy',
                            'price': current_price,
                            'size': position_size,
                            'consensus': consensus,
                            'capital': capital
                        })
                    elif direction == 'sell':
                        # Enter short
                        position = 'short'
                        position_size = capital / (1 - current_price)
                        entry_price = current_price
                        trades_log.append({
                            'time': current_time,
                            'action': 'sell',
                            'price': current_price,
                            'size': position_size,
                            'consensus': consensus,
                            'capital': capital
                        })

            elif position == 'long':
                # Check for exit
                if consensus < CONSENSUS_THRESHOLD or direction == 'sell':
                    # Exit long
                    pnl = position_size * (current_price - entry_price)
                    capital += pnl
                    trades_log.append({
                        'time': current_time,
                        'action': 'close_long',
                        'price': current_price,
                        'size': position_size,
                        'consensus': consensus,
                        'capital': capital,
                        'pnl': pnl
                    })
                    position = None
                    position_size = 0
                    entry_price = 0

            elif position == 'short':
                # Check for exit
                if consensus < CONSENSUS_THRESHOLD or direction == 'buy':
                    # Exit short
                    entry_no_price = 1 - entry_price
                    exit_no_price = 1 - current_price
                    pnl = position_size * (exit_no_price - entry_no_price)
                    capital += pnl
                    trades_log.append({
                        'time': current_time,
                        'action': 'close_short',
                        'price': current_price,
                        'size': position_size,
                        'consensus': consensus,
                        'capital': capital,
                        'pnl': pnl
                    })
                    position = None
                    position_size = 0
                    entry_price = 0

        current_time = window_end

    # Close any open position at end
    if position is not None:
        final_trades = trades_df[trades_df['ts'] >= end_time - window_size]
        if not final_trades.empty:
            final_price = final_trades['price'].mean()
            if position == 'long':
                pnl = position_size * (final_price - entry_price)
            else:
                entry_no_price = 1 - entry_price
                exit_no_price = 1 - final_price
                pnl = position_size * (exit_no_price - entry_no_price)
            capital += pnl
            trades_log.append({
                'time': end_time,
                'action': f'close_{position}',
                'price': final_price,
                'size': position_size,
                'consensus': 0,
                'capital': capital,
                'pnl': pnl
            })

    return {
        'method': method,
        'initial_capital': REAL_INITIAL_CAPITAL,
        'final_capital': capital,
        'return_pct': (capital / REAL_INITIAL_CAPITAL - 1) * 100,
        'num_trades': len([t for t in trades_log if t['action'] in ['buy', 'sell']]),
        'trades': trades_log
    }


def main():
    """Run backtest on real Polymarket data."""
    print("\n" + "=" * 70)
    print("REAL DATA BACKTEST - POLYMARKET WHALE STRATEGY")
    print("=" * 70)

    # Load data
    try:
        all_trades = pd.read_parquet(PARQUET_DIR / "polymarket_trades_all.parquet")
        xrp_trades = pd.read_parquet(PARQUET_DIR / "polymarket_xrp_trades.parquet")
        crypto_trades = pd.read_parquet(PARQUET_DIR / "polymarket_crypto_trades.parquet")
    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}")
        logger.info("Run polymarket_fetcher.py first to collect data")
        return

    print(f"\nLoaded data:")
    print(f"  All trades: {len(all_trades):,}")
    print(f"  XRP trades: {len(xrp_trades):,}")
    print(f"  Crypto trades: {len(crypto_trades):,}")

    # Analyze whale distribution
    print("\n" + "=" * 70)
    print("WHALE ANALYSIS")
    print("=" * 70)

    for label, df in [("XRP", xrp_trades), ("All Crypto", crypto_trades)]:
        if df.empty:
            continue

        print(f"\n{label} Markets:")
        trader_vol = df.groupby('proxyWallet')['size'].agg(['sum', 'mean', 'count'])
        trader_vol.columns = ['total', 'avg', 'count']
        trader_vol = trader_vol.sort_values('total', ascending=False)

        print(f"  Unique traders: {len(trader_vol)}")
        print(f"  95th percentile volume: ${trader_vol['total'].quantile(0.95):,.2f}")
        print(f"  Top 10 control: {trader_vol.head(10)['total'].sum() / trader_vol['total'].sum() * 100:.1f}% of volume")

        # Show distribution
        print(f"\n  Volume distribution:")
        for pct in [50, 75, 90, 95, 99]:
            print(f"    {pct}th percentile: ${trader_vol['total'].quantile(pct/100):,.2f}")

    # Run backtest for each method
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS")
    print("=" * 70)

    results = {}
    for method in ['top_n', 'percentile', 'trade_size']:
        print(f"\n--- {method.upper()} ---")

        # Run on XRP if we have enough data, otherwise all crypto
        if len(xrp_trades) > 100:
            result = run_real_backtest(xrp_trades.copy(), method)
        else:
            result = run_real_backtest(crypto_trades.copy(), method)

        if result:
            results[method] = result
            print(f"Initial: ${result['initial_capital']:,.2f}")
            print(f"Final: ${result['final_capital']:,.2f}")
            print(f"Return: {result['return_pct']:.2f}%")
            print(f"Trades: {result['num_trades']}")

    # Save results
    with open(PARQUET_DIR / "real_backtest_results.json", "w") as f:
        json.dump({k: {key: val for key, val in v.items() if key != 'trades'}
                   for k, v in results.items()}, f, indent=2, default=str)

    print(f"\nResults saved to {PARQUET_DIR / 'real_backtest_results.json'}")

    return results


if __name__ == "__main__":
    main()

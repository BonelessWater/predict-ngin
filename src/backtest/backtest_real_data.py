"""
Backtest using real collected Polymarket data.
Requires running the data collector first to build historical database.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.config import Config, get_config
from src.data.collector import load_collected_data, print_collection_status
from src.backtest.proper_backtest import (
    detect_whales_no_lookahead,
    calculate_whale_consensus_no_lookahead,
    execute_trade,
    calculate_metrics,
    BacktestState,
    Position
)

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def prepare_real_data(trades_df: pd.DataFrame) -> tuple:
    """
    Prepare real Polymarket data for backtesting.

    Returns:
        (trades_df, prices_df) in format expected by backtest
    """
    if trades_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    df = trades_df.copy()

    # Standardize column names
    column_mapping = {
        'proxyWallet': 'trader_id',
        'timestamp': 'timestamp',
        'size': 'size',
        'side': 'side',
        'price': 'price',
        'conditionId': 'market_id',
        'title': 'market_title'
    }

    df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})

    # Parse timestamp
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
    elif 'timestamp_dt' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp_dt'])

    df = df.dropna(subset=['timestamp'])
    df = df.sort_values('timestamp')

    # Standardize side
    if 'side' in df.columns:
        df['side'] = df['side'].str.upper()

    # Identify whales (is_whale column based on trade size percentile)
    if 'size' in df.columns:
        size_95 = df['size'].quantile(0.95)
        df['is_whale'] = df['size'] >= size_95

    # Create price data from trades (VWAP per hour)
    if 'price' in df.columns and 'size' in df.columns:
        df['value'] = df['price'] * df['size']
        df['hour'] = df['timestamp'].dt.floor('h')

        prices_df = df.groupby('hour').agg({
            'price': 'mean',
            'size': 'sum',
            'value': 'sum'
        }).reset_index()

        prices_df.columns = ['date', 'price', 'volume', 'value']
        prices_df['vwap'] = prices_df['value'] / prices_df['volume']
        prices_df['price'] = prices_df['vwap']
    else:
        prices_df = pd.DataFrame()

    return df, prices_df


def run_backtest_on_real_data(
    market_filter: str = None,
    config: Config = None
):
    """
    Run backtest on real collected Polymarket data.

    Args:
        market_filter: Filter for specific markets (e.g., 'XRP', 'BTC')
        config: Configuration object
    """
    config = config or get_config()

    print("=" * 70)
    print("BACKTEST ON REAL POLYMARKET DATA")
    print("=" * 70)

    # Check collection status
    print("\nChecking collected data...")
    print_collection_status()

    # Load data - try market-specific directory first, then filter from general
    print("\nLoading collected trades...")
    raw_trades = load_collected_data(market_filter=market_filter)

    if raw_trades.empty and market_filter:
        # Fall back to loading all and filtering
        print(f"No {market_filter}-specific data found, checking general collection...")
        raw_trades = load_collected_data()
        if not raw_trades.empty and 'title' in raw_trades.columns:
            mask = raw_trades['title'].str.lower().str.contains(market_filter.lower(), na=False)
            raw_trades = raw_trades[mask]

    if raw_trades.empty:
        print("\nNo collected data found!")
        print("Please run the data collector first:")
        if market_filter:
            print(f"  python src/data/collector.py start --market {market_filter}")
        else:
            print("  python src/data/collector.py start")
        print("\nOr run a single collection:")
        if market_filter:
            print(f"  python src/data/collector.py once --market {market_filter}")
        else:
            print("  python src/data/collector.py once")
        return None

    print(f"Loaded {len(raw_trades):,} trades")

    if raw_trades.empty:
        print(f"No trades found for filter: {market_filter}")
        return None

    # Prepare data
    trades_df, prices_df = prepare_real_data(raw_trades)

    if prices_df.empty:
        print("Could not generate price data from trades")
        return None

    print(f"\nData prepared:")
    print(f"  Trades: {len(trades_df):,}")
    print(f"  Price points: {len(prices_df)}")
    print(f"  Date range: {trades_df['timestamp'].min()} to {trades_df['timestamp'].max()}")

    # Check if enough data
    days_of_data = (trades_df['timestamp'].max() - trades_df['timestamp'].min()).days
    if days_of_data < config.backtest.warmup_days:
        print(f"\nInsufficient data: {days_of_data} days (need {config.backtest.warmup_days}+)")
        print("Continue collecting data and try again later.")
        return None

    # Run backtest
    print(f"\nRunning backtest...")
    print(f"  Taker Fee: {config.costs.taker_fee:.2%}")
    print(f"  Base Slippage: {config.costs.base_slippage:.2%}")
    print(f"  Consensus Threshold: {config.strategy.consensus_threshold:.0%}")

    # Initialize state
    state = BacktestState(capital=config.strategy.initial_capital)

    # Convert prices to daily
    prices_df['date'] = pd.to_datetime(prices_df['date'])
    daily_prices = prices_df.set_index('date').resample('D').last().dropna().reset_index()

    if len(daily_prices) < config.backtest.warmup_days:
        print(f"Not enough daily price data")
        return None

    # Run simulation
    for i in range(config.backtest.warmup_days, len(daily_prices)):
        current_date = daily_prices.iloc[i]['date']
        current_price = daily_prices.iloc[i]['price']

        # Detect whales (no look-ahead)
        whales = detect_whales_no_lookahead(trades_df, current_date, config, 'top_n')

        if not whales:
            continue

        # Calculate consensus (no look-ahead)
        consensus, direction = calculate_whale_consensus_no_lookahead(
            trades_df, whales, current_date, lookback_hours=24
        )

        # Trading logic
        threshold = config.strategy.consensus_threshold

        if state.position == Position.NONE:
            if consensus >= threshold:
                if direction == 'buy':
                    state = execute_trade(state, 'buy', current_price, current_date,
                                         consensus, direction, config)
                elif direction == 'sell':
                    state = execute_trade(state, 'sell', current_price, current_date,
                                         consensus, direction, config)

        elif state.position == Position.LONG:
            if consensus < threshold or direction == 'sell':
                state = execute_trade(state, 'close_long', current_price, current_date,
                                     consensus, direction, config)

        elif state.position == Position.SHORT:
            if consensus < threshold or direction == 'buy':
                state = execute_trade(state, 'close_short', current_price, current_date,
                                     consensus, direction, config)

        # Track equity
        if state.position == Position.LONG:
            mtm = state.position_size * current_price
            equity = state.capital + mtm
        elif state.position == Position.SHORT:
            mtm = state.position_size * (1 - current_price)
            equity = state.capital + mtm
        else:
            equity = state.capital

        state.equity_curve.append({
            'date': current_date,
            'equity': equity,
            'price': current_price
        })

    # Close any open position
    if state.position != Position.NONE:
        final_price = daily_prices.iloc[-1]['price']
        final_date = daily_prices.iloc[-1]['date']
        if state.position == Position.LONG:
            state = execute_trade(state, 'close_long', final_price, final_date, 0, 'neutral', config)
        else:
            state = execute_trade(state, 'close_short', final_price, final_date, 0, 'neutral', config)

    # Calculate metrics
    metrics = calculate_metrics(state, config)

    # Print results
    print("\n" + "=" * 50)
    print("BACKTEST RESULTS (REAL DATA)")
    print("=" * 50)
    print(f"Initial Capital:   ${config.strategy.initial_capital:,.2f}")
    print(f"Final Capital:     ${metrics.get('final_capital', 0):,.2f}")
    print(f"Total Return:      {metrics.get('total_return_pct', 0):.2f}%")
    print(f"Sharpe Ratio:      {metrics.get('sharpe_ratio', 0):.2f}")
    print(f"Max Drawdown:      {metrics.get('max_drawdown_pct', 0):.2f}%")
    print(f"Win Rate:          {metrics.get('win_rate_pct', 0):.1f}%")
    print(f"Trades:            {metrics.get('num_trades', 0)}")
    print(f"Total Costs:       ${metrics.get('total_costs', 0):.2f}")

    # Save results
    output_dir = config.data.output_dir
    with open(output_dir / "real_data_backtest_results.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    if state.equity_curve:
        eq_df = pd.DataFrame(state.equity_curve)
        eq_df.to_csv(output_dir / "equity_curve_real_data.csv", index=False)

    print(f"\nResults saved to: {output_dir}")

    return metrics, state


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Backtest on real Polymarket data')
    parser.add_argument('--market', type=str, default=None,
                       help='Market filter (e.g., XRP, BTC, ETH)')
    parser.add_argument('--fee', type=float, default=None,
                       help='Override taker fee (e.g., 0.01 for 1%%)')

    args = parser.parse_args()

    config = get_config()
    if args.fee is not None:
        config.costs.taker_fee = args.fee

    run_backtest_on_real_data(market_filter=args.market, config=config)


if __name__ == "__main__":
    main()

"""
Main entry point for the Polymarket whale tracking system.
Orchestrates data loading, whale detection, and backtesting.
"""
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import PARQUET_DIR, INITIAL_CAPITAL, CONSENSUS_THRESHOLD
from src.data.loader import PolymarketDataLoader
from src.whales.detector import WhaleDetector, WhaleMethod
from src.backtest.engine import BacktestEngine

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_data_collection(force_refresh: bool = False):
    """Collect and cache market data."""
    logger.info("Starting data collection...")
    loader = PolymarketDataLoader()
    data = loader.load_xrp_market_data(force_refresh=force_refresh)

    print(f"\n{'='*60}")
    print("DATA COLLECTION SUMMARY")
    print(f"{'='*60}")
    print(f"XRP Markets found: {len(data['markets'])}")
    print(f"Trades collected: {len(data['trades'])}")
    print(f"Activity records: {len(data['activity'])}")
    print(f"Data cached in: {PARQUET_DIR}")

    if not data['markets'].empty and 'question' in data['markets'].columns:
        print(f"\nSample markets:")
        for _, m in data['markets'].head(5).iterrows():
            print(f"  - {m.get('question', 'Unknown')[:60]}...")

    return data


def run_whale_analysis(trades_df=None):
    """Analyze whale activity using all three methods."""
    if trades_df is None:
        loader = PolymarketDataLoader()
        data = loader.load_xrp_market_data()
        trades_df = data['trades']

    if trades_df.empty:
        logger.warning("No trade data available for whale analysis")
        return

    detector = WhaleDetector()
    results = detector.detect_all_methods(trades_df)

    print(f"\n{'='*60}")
    print("WHALE DETECTION RESULTS")
    print(f"{'='*60}")

    for method, whales in results.items():
        print(f"\n## {method.value.upper()} ({len(whales)} whales)")
        print("-" * 40)

        if whales:
            for i, w in enumerate(whales[:5], 1):
                print(f"{i}. {w.address[:20]}...")
                print(f"   Volume: ${w.total_volume:,.2f}")
                print(f"   Trades: {w.trade_count}")
                print(f"   Avg Size: ${w.avg_trade_size:,.2f}")
                print(f"   Direction Bias: {w.direction_bias:+.2f}")
                print()


def run_backtest(
    trades_df=None,
    prices_df=None,
    signal_interval_hours: int = 4
):
    """Run backtest for all whale detection methods."""
    loader = PolymarketDataLoader()

    if trades_df is None or prices_df is None:
        data = loader.load_xrp_market_data()
        trades_df = data['trades']

        # Try to load price data from first available token
        if not data['markets'].empty:
            for _, market in data['markets'].iterrows():
                token_ids = []
                if 'clobTokenIds' in market and market['clobTokenIds']:
                    token_ids = market['clobTokenIds'] if isinstance(market['clobTokenIds'], list) else [market['clobTokenIds']]

                for token_id in token_ids:
                    if token_id:
                        prices_df = loader.fetch_price_history(token_id)
                        if not prices_df.empty:
                            break
                if prices_df is not None and not prices_df.empty:
                    break

    if trades_df is None or trades_df.empty:
        logger.warning("No trade data available for backtest")
        return None

    if prices_df is None or prices_df.empty:
        logger.warning("No price data available for backtest")
        # Generate synthetic prices for demo
        logger.info("Generating synthetic price data for demonstration...")
        import numpy as np
        import pandas as pd

        n_prices = 168 * 4  # 4 weeks hourly
        base_time = datetime.now() - timedelta(days=28)
        base_price = 0.5

        prices_list = []
        for i in range(n_prices):
            base_price += np.random.randn() * 0.005
            base_price = max(0.05, min(0.95, base_price))
            prices_list.append({
                'timestamp': base_time + timedelta(hours=i),
                'price': base_price
            })
        prices_df = pd.DataFrame(prices_list)

    engine = BacktestEngine(
        initial_capital=INITIAL_CAPITAL,
        consensus_threshold=CONSENSUS_THRESHOLD
    )

    engine.run_all_methods(
        trades_df=trades_df,
        prices_df=prices_df,
        signal_interval_hours=signal_interval_hours
    )

    # Print report
    print(engine.generate_report())

    # Save results
    engine.save_results()

    return engine


def run_demo():
    """Run a full demo with synthetic data."""
    import numpy as np
    import pandas as pd

    logger.info("Running demo with synthetic data...")

    # Generate realistic synthetic data
    np.random.seed(42)
    n_trades = 2000
    n_prices = 168 * 4  # 4 weeks hourly

    # Create trader pool with whale concentration
    traders = [f"0x{i:040x}" for i in range(200)]
    whale_traders = traders[:15]  # Top 15 are potential whales

    trades_list = []
    base_time = datetime.now() - timedelta(days=28)

    # Simulate market phases
    # Phase 1 (week 1-2): Accumulation - whales buying
    # Phase 2 (week 3): Price rise
    # Phase 3 (week 4): Distribution - whales selling

    for i in range(n_trades):
        progress = i / n_trades
        is_whale = np.random.random() < 0.15

        if is_whale:
            trader = np.random.choice(whale_traders)
            base_size = np.random.exponential(20000)
        else:
            trader = np.random.choice(traders)
            base_size = np.random.exponential(500)

        # Phase-dependent whale behavior
        if progress < 0.5:  # First half: accumulation
            whale_buy_prob = 0.75
        else:  # Second half: distribution
            whale_buy_prob = 0.30

        retail_buy_prob = 0.5 + (progress - 0.5) * 0.4  # Retail follows price

        buy_prob = whale_buy_prob if is_whale else retail_buy_prob
        side = 'buy' if np.random.random() < buy_prob else 'sell'

        trades_list.append({
            'timestamp': base_time + timedelta(minutes=i * 20),
            'trader': trader,
            'size': base_size,
            'side': side
        })

    trades_df = pd.DataFrame(trades_list)

    # Generate prices that respond to whale activity
    prices_list = []
    base_price = 0.35

    for i in range(n_prices):
        progress = i / n_prices

        # Price trend follows whale accumulation/distribution
        if progress < 0.3:
            drift = 0.0005  # Slow rise during accumulation
        elif progress < 0.7:
            drift = 0.002  # Faster rise
        else:
            drift = -0.001  # Decline during distribution

        volatility = 0.008
        base_price += drift + np.random.randn() * volatility
        base_price = max(0.10, min(0.90, base_price))

        prices_list.append({
            'timestamp': base_time + timedelta(hours=i),
            'price': base_price
        })

    prices_df = pd.DataFrame(prices_list)

    # Run backtest
    engine = BacktestEngine()
    engine.run_all_methods(trades_df, prices_df, signal_interval_hours=4)

    print(engine.generate_report())
    engine.save_results()

    return engine


def main():
    parser = argparse.ArgumentParser(
        description='Polymarket Whale Tracking System'
    )
    parser.add_argument(
        'command',
        choices=['collect', 'analyze', 'backtest', 'demo', 'all'],
        help='Command to run'
    )
    parser.add_argument(
        '--force-refresh', '-f',
        action='store_true',
        help='Force refresh of cached data'
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=4,
        help='Signal check interval in hours (default: 4)'
    )

    args = parser.parse_args()

    if args.command == 'collect':
        run_data_collection(force_refresh=args.force_refresh)

    elif args.command == 'analyze':
        run_whale_analysis()

    elif args.command == 'backtest':
        run_backtest(signal_interval_hours=args.interval)

    elif args.command == 'demo':
        run_demo()

    elif args.command == 'all':
        data = run_data_collection(force_refresh=args.force_refresh)
        run_whale_analysis(data['trades'])
        run_backtest(signal_interval_hours=args.interval)


if __name__ == "__main__":
    main()

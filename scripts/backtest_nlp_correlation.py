#!/usr/bin/env python3
"""
Backtest NLP Correlation Strategy with random sample of markets.

Usage:
    python scripts/backtest_nlp_correlation.py --n-markets 500
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
import random

from src.trading.strategies.nlp_correlation import NLPCorrelationStrategy
from src.trading.data_modules.parquet_store import TradeStore, PriceStore
from src.trading.polymarket_backtest import (
    run_polymarket_backtest,
    PolymarketBacktestConfig,
    ClobPriceStore,
    print_polymarket_result,
)
from src.trading.data_modules.costs import DEFAULT_COST_MODEL
from src.trading.momentum_signals_from_trades import trades_to_price_history


def load_random_markets(
    n_markets: int = 500,
    parquet_markets_dir: str = "data/polymarket",
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load random sample of markets from parquet files.
    
    Returns:
        DataFrame with markets
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    markets_path = Path(parquet_markets_dir)
    if not markets_path.exists():
        raise FileNotFoundError(f"Markets directory not found: {parquet_markets_dir}")

    single = markets_path / "markets.parquet"
    markets_files = [single] if single.exists() else list(markets_path.glob("markets_*.parquet"))
    if not markets_files:
        raise FileNotFoundError(f"No markets parquet files found in {parquet_markets_dir}")
    
    print(f"Loading markets from {len(markets_files)} parquet files...")
    markets_list = []
    for f in markets_files:
        df = pd.read_parquet(f)
        markets_list.append(df)
    
    if not markets_list:
        raise ValueError("No markets found in parquet files")
    
    all_markets = pd.concat(markets_list, ignore_index=True)
    
    # Normalize column names
    if "question" not in all_markets.columns:
        if "Question" in all_markets.columns:
            all_markets["question"] = all_markets["Question"]
        elif "question_text" in all_markets.columns:
            all_markets["question"] = all_markets["question_text"]
        else:
            raise ValueError("No 'question' column found in markets")
    
    if "id" not in all_markets.columns:
        if "ID" in all_markets.columns:
            all_markets["id"] = all_markets["ID"]
        elif "market_id" in all_markets.columns:
            all_markets["id"] = all_markets["market_id"]
        else:
            raise ValueError("No 'id' column found in markets")
    
    # Filter markets with questions
    all_markets = all_markets[all_markets["question"].notna()]
    all_markets = all_markets[all_markets["question"].str.strip() != ""]
    
    print(f"  Found {len(all_markets):,} markets with questions")
    
    # Random sample
    if len(all_markets) > n_markets:
        sampled = all_markets.sample(n=n_markets, random_state=seed)
        print(f"  Randomly sampled {n_markets} markets")
    else:
        sampled = all_markets
        print(f"  Using all {len(sampled)} markets (less than requested {n_markets})")
    
    return sampled.reset_index(drop=True)


def generate_signals_for_backtest(
    markets_df: pd.DataFrame,
    prices_df: Optional[pd.DataFrame],
    trades_df: Optional[pd.DataFrame],
    strategy: NLPCorrelationStrategy,
    eval_freq_hours: int = 6,
) -> pd.DataFrame:
    """
    Generate signals for backtest.
    
    Returns:
        DataFrame with signals in format expected by run_polymarket_backtest
    """
    print("\nGenerating signals...")
    
    # Prepare timestamps
    if prices_df is not None and not prices_df.empty:
        prices_df["datetime"] = pd.to_datetime(prices_df["datetime"])
        start_time = prices_df["datetime"].min()
        end_time = prices_df["datetime"].max()
        timestamps = pd.date_range(start_time, end_time, freq=f"{eval_freq_hours}H").tolist()
    elif trades_df is not None and not trades_df.empty:
        trades_df["timestamp"] = pd.to_datetime(trades_df["timestamp"])
        start_time = trades_df["timestamp"].min()
        end_time = trades_df["timestamp"].max()
        timestamps = pd.date_range(start_time, end_time, freq=f"{eval_freq_hours}H").tolist()
    else:
        # Fallback: use current time
        timestamps = [datetime.now()]
    
    # Limit timestamps for performance
    if len(timestamps) > 1000:
        timestamps = timestamps[::len(timestamps)//1000 + 1][:1000]
    
    print(f"  Evaluating at {len(timestamps)} timestamps")
    
    # Convert markets to list of dicts
    markets = markets_df.to_dict("records")
    
    # Get current prices
    prices = {}
    if prices_df is not None and not prices_df.empty:
        for market_id in markets_df["id"]:
            market_prices = prices_df[prices_df["market_id"] == market_id]
            if not market_prices.empty:
                latest = market_prices.iloc[-1]
                prices[market_id] = {
                    "price": float(latest["price"]),
                    "datetime": latest["datetime"],
                }
    
    # Generate signals
    all_signals = []
    
    for i, timestamp in enumerate(timestamps):
        if i % 100 == 0:
            print(f"  Processing timestamp {i+1}/{len(timestamps)}...")
        
        market_data = {
            "markets": markets,
            "prices": prices,
        }
        
        if trades_df is not None:
            # Filter trades up to this timestamp
            trades_up_to_now = trades_df[
                pd.to_datetime(trades_df["timestamp"]) <= timestamp
            ]
            if not trades_up_to_now.empty:
                market_data["trades"] = trades_up_to_now
        
        signals = strategy.generate_signals(market_data, timestamp)
        
        for signal in signals:
            all_signals.append({
                "market_id": signal.market_id,
                "signal_time": signal.timestamp,
                "signal_ts": int(signal.timestamp.timestamp()),
                "outcome": signal.outcome,
                "side": signal.signal_type.value,
                "size": signal.size,
                "confidence": signal.confidence,
            })
    
    if not all_signals:
        print("  No signals generated")
        return pd.DataFrame()
    
    signals_df = pd.DataFrame(all_signals)
    print(f"  Generated {len(signals_df)} signals")
    
    return signals_df


def run_backtest(
    n_markets: int = 500,
    parquet_markets_dir: str = "data/polymarket",
    parquet_prices_dir: str = "data/polymarket/prices",
    parquet_trades_dir: str = "data/polymarket/trades",
    use_trades: bool = True,
    similarity_method: str = "hybrid",
    min_similarity: float = 0.4,
    z_score_threshold: float = 2.0,
    lookback_hours: int = 72,
    starting_capital: float = 10000,
    position_size: float = 100,
    seed: Optional[int] = None,
    output_dir: str = "data/output",
) -> None:
    """
    Run backtest with random sample of markets.
    """
    print("=" * 60)
    print("NLP CORRELATION STRATEGY BACKTEST")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Markets: {n_markets} (random sample)")
    print(f"  Similarity method: {similarity_method}")
    print(f"  Min similarity: {min_similarity}")
    print(f"  Z-score threshold: {z_score_threshold}")
    print(f"  Starting capital: ${starting_capital:,.0f}")
    print(f"  Position size: ${position_size:,.0f}")
    
    # Load random markets
    print(f"\n[1] Loading {n_markets} random markets...")
    markets_df = load_random_markets(
        n_markets=n_markets,
        parquet_markets_dir=parquet_markets_dir,
        seed=seed,
    )
    
    # Load prices/trades
    print(f"\n[2] Loading price/trade data...")
    prices_df = None
    trades_df = None
    
    if use_trades:
        print("  Loading trades from parquet...")
        try:
            trade_store = TradeStore(parquet_trades_dir)
            if trade_store.available():
                trades_df = trade_store.load_trades(min_usd=10)
                if not trades_df.empty:
                    print(f"  Loaded {len(trades_df):,} trades")
                    # Convert to price history
                    prices_df = trades_to_price_history(trades_df, outcome="YES")
                    print(f"  Converted to {len(prices_df):,} price records")
        except Exception as e:
            print(f"  Warning: Could not load trades: {e}")
            use_trades = False
    
    if prices_df is None or prices_df.empty:
        print("  Loading prices from parquet...")
        try:
            price_store = PriceStore(parquet_prices_dir)
            if price_store.available():
                market_ids = markets_df["id"].astype(str).tolist()
                prices_df = price_store.load_prices_for_markets(market_ids, outcome="YES")
                if not prices_df.empty:
                    print(f"  Loaded {len(prices_df):,} price records")
        except Exception as e:
            print(f"  Warning: Could not load prices: {e}")
    
    if prices_df is None or prices_df.empty:
        print("  ERROR: No price/trade data available!")
        print("  Cannot run backtest without price data.")
        return
    
    # Create strategy (with memory limits)
    print(f"\n[3] Creating NLP correlation strategy...")
    # Limit markets for similarity computation to avoid memory issues
    max_markets_for_sim = min(n_markets, args.max_markets_for_similarity)
    strategy = NLPCorrelationStrategy(
        similarity_method=similarity_method,
        min_similarity=min_similarity,
        z_score_threshold=z_score_threshold,
        lookback_hours=lookback_hours,
        max_markets_for_similarity=max_markets_for_sim,
        similarity_batch_size=50,  # Smaller batches for memory efficiency
        max_similarity_comparisons=args.max_similarity_comparisons,
    )
    
    # Generate signals
    print(f"\n[4] Generating signals...")
    signals_df = generate_signals_for_backtest(
        markets_df=markets_df,
        prices_df=prices_df,
        trades_df=trades_df,
        strategy=strategy,
    )
    
    if signals_df.empty:
        print("\n  No signals generated. Cannot run backtest.")
        print("  Try:")
        print("    - Lowering min_similarity threshold")
        print("    - Lowering z_score_threshold")
        print("    - Using more markets")
        return
    
    # Run backtest
    print(f"\n[5] Running backtest...")
    print(f"  Signals: {len(signals_df)}")
    
    # Create price store
    price_store = ClobPriceStore(parquet_dir=parquet_prices_dir)
    
    # Add market metadata to price store
    for _, market in markets_df.iterrows():
        market_id = str(market["id"])
        price_store._market_metadata[market_id] = {
            "id": market_id,
            "question": market.get("question", ""),
            "volume": float(market.get("volume", 0) or 0),
            "volume_24hr": float(market.get("volume_24hr", market.get("volume24hr", 0)) or 0),
            "liquidity": float(market.get("liquidity", 0) or 0),
            "end_date": market.get("end_date", market.get("endDate")),
        }
    
    # Backtest config
    config = PolymarketBacktestConfig(
        strategy_name="NLP Correlation",
        starting_capital=starting_capital,
        position_size=position_size,
        min_liquidity=100,
        min_volume=1000,
    )
    
    # Run backtest
    result = run_polymarket_backtest(
        signals=signals_df,
        price_store=price_store,
        config=config,
        cost_model=DEFAULT_COST_MODEL,
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    print_polymarket_result(result)
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    trades_file = output_path / f"nlp_correlation_backtest_{timestamp_str}.csv"
    signals_file = output_path / f"nlp_correlation_signals_{timestamp_str}.csv"
    
    result.trades_df.to_csv(trades_file, index=False)
    signals_df.to_csv(signals_file, index=False)
    
    print(f"\nResults saved:")
    print(f"  Trades: {trades_file}")
    print(f"  Signals: {signals_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Backtest NLP Correlation Strategy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--n-markets",
        type=int,
        default=500,
        help="Number of random markets to use",
    )
    parser.add_argument(
        "--parquet-markets-dir",
        type=str,
        default="data/polymarket",
        help="Directory with markets parquet files",
    )
    parser.add_argument(
        "--parquet-prices-dir",
        type=str,
        default="data/polymarket/prices",
        help="Directory with prices parquet files",
    )
    parser.add_argument(
        "--parquet-trades-dir",
        type=str,
        default="data/polymarket/trades",
        help="Directory with trades parquet files",
    )
    parser.add_argument(
        "--no-trades",
        action="store_true",
        help="Don't use trades, only prices",
    )
    parser.add_argument(
        "--similarity-method",
        type=str,
        default="hybrid",
        choices=["bag_of_words", "tfidf", "embedding", "hybrid"],
        help="Similarity method",
    )
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=0.4,
        help="Minimum similarity threshold",
    )
    parser.add_argument(
        "--z-threshold",
        type=float,
        default=2.0,
        help="Z-score threshold",
    )
    parser.add_argument(
        "--lookback-hours",
        type=int,
        default=72,
        help="Lookback window in hours",
    )
    parser.add_argument(
        "--starting-capital",
        type=float,
        default=10000,
        help="Starting capital",
    )
    parser.add_argument(
        "--position-size",
        type=float,
        default=100,
        help="Position size",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--max-markets-for-similarity",
        type=int,
        default=500,
        help="Maximum markets to process for similarity (memory limit)",
    )
    parser.add_argument(
        "--max-similarity-comparisons",
        type=int,
        default=100000,
        help="Maximum similarity comparisons (memory limit)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/output",
        help="Output directory",
    )
    
    args = parser.parse_args()
    
    run_backtest(
        n_markets=args.n_markets,
        parquet_markets_dir=args.parquet_markets_dir,
        parquet_prices_dir=args.parquet_prices_dir,
        parquet_trades_dir=args.parquet_trades_dir,
        use_trades=not args.no_trades,
        similarity_method=args.similarity_method,
        min_similarity=args.min_similarity,
        z_score_threshold=args.z_threshold,
        lookback_hours=args.lookback_hours,
        starting_capital=args.starting_capital,
        position_size=args.position_size,
        seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()

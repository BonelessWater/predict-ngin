#!/usr/bin/env python3
"""
Run NLP Correlation Strategy

Generates signals using NLP-based market similarity and correlation trading.

Usage:
    # Basic run with default settings
    python scripts/run_nlp_correlation.py

    # Use trades instead of prices
    python scripts/run_nlp_correlation.py --use-trades

    # Custom parameters
    python scripts/run_nlp_correlation.py --min-similarity 0.5 --z-threshold 2.5

    # Optimize parameters first
    python scripts/optimize_nlp_correlation.py --method random_search --n-iterations 50
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

from src.trading.strategies.nlp_correlation import NLPCorrelationStrategy
from src.trading.data_modules import PredictionMarketDB, DEFAULT_DB_PATH
from src.trading.data_modules.parquet_store import TradeStore, PriceStore
from src.trading.momentum_signals_from_trades import trades_to_price_history


def load_data(
    db_path: Optional[str] = None,
    use_trades: bool = False,
    market_limit: Optional[int] = None,
    parquet_markets_dir: Optional[str] = None,
    parquet_prices_dir: Optional[str] = None,
    parquet_trades_dir: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Load market and price/trade data from parquet files (preferred) or database.
    
    Returns:
        (markets_df, prices_df, trades_df)
    """
    markets_df = pd.DataFrame()
    prices_df = None
    trades_df = None
    
    # Try parquet first (preferred)
    parquet_markets_path = Path(parquet_markets_dir or "data/polymarket")
    parquet_prices_path = Path(parquet_prices_dir or "data/polymarket/prices")
    parquet_trades_path = Path(parquet_trades_dir or "data/polymarket/trades")

    # Load markets from parquet if available
    if parquet_markets_path.exists():
        single = parquet_markets_path / "markets.parquet"
        markets_files = [single] if single.exists() else list(parquet_markets_path.glob("markets_*.parquet"))
        if markets_files:
            print(f"Loading markets from parquet ({len(markets_files)} files)...")
            markets_list = []
            for f in markets_files:
                df = pd.read_parquet(f)
                markets_list.append(df)
            if markets_list:
                markets_df = pd.concat(markets_list, ignore_index=True)
                # Normalize column names
                if "question" not in markets_df.columns and "Question" in markets_df.columns:
                    markets_df["question"] = markets_df["Question"]
                if "id" not in markets_df.columns and "ID" in markets_df.columns:
                    markets_df["id"] = markets_df["ID"]
                print(f"  Loaded {len(markets_df):,} markets from parquet")
    
    # Fallback to database if no parquet markets
    if markets_df.empty and db_path:
        print(f"Loading markets from database {db_path}...")
        try:
            db = PredictionMarketDB(db_path)
            markets_df = db.get_all_markets()
            db.close()
            print(f"  Loaded {len(markets_df)} markets from database")
        except Exception as e:
            print(f"  Warning: Could not load from database: {e}")
    
    if markets_df.empty:
        print("  ERROR: No markets found!")
        return markets_df, None, None
    
    if market_limit:
        markets_df = markets_df.head(market_limit)
    
    # Load trades or prices
    if use_trades:
        # Load trades from parquet
        print("Loading trades from parquet...")
        try:
            trade_store = TradeStore(str(parquet_trades_path))
            if trade_store.available():
                trades_df = trade_store.load_trades(min_usd=10)
                if not trades_df.empty:
                    print(f"  Loaded {len(trades_df):,} trades from parquet")
                    # Convert to price history format
                    prices_df = trades_to_price_history(trades_df, outcome="YES")
                    print(f"  Converted to {len(prices_df):,} price records")
            else:
                print("  No parquet trades found, trying database...")
                if db_path:
                    db = PredictionMarketDB(db_path)
                    # Try to load trades from DB
                    trades_df = db.query("SELECT * FROM polymarket_trades LIMIT 10000")
                    db.close()
                    if not trades_df.empty:
                        print(f"  Loaded {len(trades_df):,} trades from database")
                        prices_df = trades_to_price_history(trades_df, outcome="YES")
        except Exception as e:
            print(f"  Warning: Could not load trades: {e}")
            use_trades = False
    
    if not use_trades or prices_df is None or prices_df.empty:
        # Load prices from parquet
        print("Loading prices from parquet...")
        try:
            price_store = PriceStore(str(parquet_prices_path))
            if price_store.available():
                # Get prices for markets we have
                market_ids = markets_df["id"].astype(str).tolist()[:100]  # Limit for performance
                prices_df = price_store.load_prices_for_markets(market_ids, outcome="YES")
                if not prices_df.empty:
                    print(f"  Loaded {len(prices_df):,} price records from parquet")
            else:
                print("  No parquet prices found, trying database...")
                if db_path:
                    db = PredictionMarketDB(db_path)
                    market_ids = markets_df["id"].tolist()[:100]
                    prices_list = []
                    for market_id in market_ids:
                        prices = db.get_price_history(market_id, outcome="YES")
                        if not prices.empty:
                            prices_list.append(prices)
                    db.close()
                    if prices_list:
                        prices_df = pd.concat(prices_list, ignore_index=True)
                        print(f"  Loaded {len(prices_df):,} price records from database")
        except Exception as e:
            print(f"  Warning: Could not load prices: {e}")
    
    return markets_df, prices_df, trades_df


def run_strategy(
    markets_df: pd.DataFrame,
    prices_df: Optional[pd.DataFrame],
    trades_df: Optional[pd.DataFrame],
    similarity_method: str = "hybrid",
    min_similarity: float = 0.4,
    z_score_threshold: float = 2.0,
    lookback_hours: int = 72,
    min_spread_pct: float = 0.05,
    max_spread_pct: float = 0.30,
    use_price_correlation: bool = True,
    output_path: Optional[str] = None,
):
    """
    Run NLP correlation strategy and generate signals.
    """
    print("\n" + "=" * 60)
    print("NLP CORRELATION STRATEGY")
    print("=" * 60)
    
    # Create strategy
    print(f"\nCreating strategy...")
    print(f"  Similarity method: {similarity_method}")
    print(f"  Min similarity: {min_similarity}")
    print(f"  Z-score threshold: {z_score_threshold}")
    print(f"  Lookback hours: {lookback_hours}")
    
    # Limit markets for similarity to avoid memory issues
    max_markets_for_sim = min(len(markets_df), args.max_markets_for_similarity)
    
    strategy = NLPCorrelationStrategy(
        similarity_method=similarity_method,
        min_similarity=min_similarity,
        z_score_threshold=z_score_threshold,
        lookback_hours=lookback_hours,
        min_spread_pct=min_spread_pct,
        max_spread_pct=max_spread_pct,
        use_price_correlation=use_price_correlation,
        max_markets_for_similarity=max_markets_for_sim,
        similarity_batch_size=50,  # Smaller batches for memory
        max_similarity_comparisons=args.max_similarity_comparisons,
    )
    
    # Prepare market data
    print("\nPreparing market data...")
    markets = markets_df.to_dict("records")
    
    # Get current prices from prices_df or trades_df
    prices = {}
    if prices_df is not None and not prices_df.empty:
        # Get latest price for each market
        for market_id in markets_df["id"]:
            market_prices = prices_df[prices_df["market_id"] == market_id]
            if not market_prices.empty:
                latest = market_prices.iloc[-1]
                prices[market_id] = {
                    "price": float(latest["price"]),
                    "datetime": latest["datetime"],
                }
    
    # Prepare timestamps for signal generation
    if prices_df is not None and not prices_df.empty:
        prices_df["datetime"] = pd.to_datetime(prices_df["datetime"])
        start_time = prices_df["datetime"].min()
        end_time = prices_df["datetime"].max()
        timestamps = pd.date_range(start_time, end_time, freq="6H").tolist()
        # Limit to reasonable number
        if len(timestamps) > 50:
            timestamps = timestamps[::len(timestamps)//50 + 1][:50]
    else:
        timestamps = [datetime.now()]
    
    print(f"  Markets: {len(markets)}")
    print(f"  Prices: {len(prices)}")
    print(f"  Timestamps: {len(timestamps)}")
    
    # Generate signals
    print("\nGenerating signals...")
    all_signals = []
    
    for i, timestamp in enumerate(timestamps):
        if i % 10 == 0:
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
        all_signals.extend(signals)
    
    print(f"\nGenerated {len(all_signals)} signals")
    
    # Display results
    if all_signals:
        print("\n" + "=" * 60)
        print("SIGNAL SUMMARY")
        print("=" * 60)
        
        buy_signals = [s for s in all_signals if s.signal_type.value == "BUY"]
        sell_signals = [s for s in all_signals if s.signal_type.value == "SELL"]
        
        print(f"\nBuy signals: {len(buy_signals)}")
        print(f"Sell signals: {len(sell_signals)}")
        
        # Show top signals by confidence
        all_signals_sorted = sorted(all_signals, key=lambda s: s.confidence, reverse=True)
        print(f"\nTop 10 signals by confidence:")
        for i, signal in enumerate(all_signals_sorted[:10], 1):
            pair_market = signal.features.get("pair_market", "N/A")
            z_score = signal.features.get("z_score", 0)
            spread = signal.features.get("spread", 0)
            print(f"  {i}. {signal.market_id[:30]}...")
            print(f"     Type: {signal.signal_type.value}, Confidence: {signal.confidence:.3f}")
            print(f"     Pair: {pair_market[:30]}..., Z-score: {z_score:.2f}, Spread: {spread:.3f}")
        
        # Save to CSV if requested
        if output_path:
            print(f"\nSaving signals to {output_path}...")
            signals_data = []
            for signal in all_signals:
                signals_data.append({
                    "timestamp": signal.timestamp,
                    "market_id": signal.market_id,
                    "signal_type": signal.signal_type.value,
                    "confidence": signal.confidence,
                    "price": signal.price,
                    "pair_market": signal.features.get("pair_market", ""),
                    "z_score": signal.features.get("z_score", 0),
                    "spread": signal.features.get("spread", 0),
                    "correlation": signal.features.get("correlation", 0),
                })
            
            signals_df = pd.DataFrame(signals_data)
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            signals_df.to_csv(output_path, index=False)
            print(f"  Saved {len(signals_df)} signals")
    else:
        print("\nNo signals generated. Try:")
        print("  - Lowering min_similarity threshold")
        print("  - Lowering z_score_threshold")
        print("  - Checking if you have enough market/price data")
    
    return all_signals


def main():
    parser = argparse.ArgumentParser(
        description="Run NLP Correlation Strategy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic run
    python scripts/run_nlp_correlation.py

    # Use trades instead of prices
    python scripts/run_nlp_correlation.py --use-trades

    # Custom parameters
    python scripts/run_nlp_correlation.py \\
        --min-similarity 0.5 \\
        --z-threshold 2.5 \\
        --lookback-hours 96

    # Save signals to CSV
    python scripts/run_nlp_correlation.py --output data/output/nlp_signals.csv
        """
    )
    
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Database path (optional, uses parquet if available)",
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
        "--use-trades",
        action="store_true",
        help="Use trades instead of price history",
    )
    parser.add_argument(
        "--market-limit",
        type=int,
        default=None,
        help="Limit number of markets (for faster testing)",
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
        help="Z-score threshold for signals",
    )
    parser.add_argument(
        "--lookback-hours",
        type=int,
        default=72,
        help="Lookback window in hours",
    )
    parser.add_argument(
        "--min-spread-pct",
        type=float,
        default=0.05,
        help="Minimum spread percentage",
    )
    parser.add_argument(
        "--max-spread-pct",
        type=float,
        default=0.30,
        help="Maximum spread percentage",
    )
    parser.add_argument(
        "--no-price-correlation",
        action="store_true",
        help="Disable price correlation (text-only similarity)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path for signals",
    )
    parser.add_argument(
        "--max-markets-for-similarity",
        type=int,
        default=500,
        help="Maximum markets to process for similarity (memory limit, default: 500)",
    )
    parser.add_argument(
        "--max-similarity-comparisons",
        type=int,
        default=100000,
        help="Maximum similarity comparisons (memory limit, default: 100000)",
    )
    
    args = parser.parse_args()
    
    # Load data (prefers parquet over database)
    markets_df, prices_df, trades_df = load_data(
        db_path=args.db_path or DEFAULT_DB_PATH,
        use_trades=args.use_trades,
        market_limit=args.market_limit,
        parquet_markets_dir=args.parquet_markets_dir,
        parquet_prices_dir=args.parquet_prices_dir,
        parquet_trades_dir=args.parquet_trades_dir,
    )
    
    if markets_df.empty:
        print("ERROR: No markets loaded")
        return 1
    
    # Run strategy
    signals = run_strategy(
        markets_df=markets_df,
        prices_df=prices_df,
        trades_df=trades_df,
        similarity_method=args.similarity_method,
        min_similarity=args.min_similarity,
        z_score_threshold=args.z_threshold,
        lookback_hours=args.lookback_hours,
        min_spread_pct=args.min_spread_pct,
        max_spread_pct=args.max_spread_pct,
        use_price_correlation=not args.no_price_correlation,
        output_path=args.output,
    )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

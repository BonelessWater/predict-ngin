"""
Optimize NLP Correlation Strategy Parameters

Uses parameter tuning to find optimal parameters for the NLP correlation strategy.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime

from src.trading.parameter_tuning import (
    ParameterTuner,
    ParameterSpace,
    TuningResult,
    save_tuning_result,
)
from src.trading.strategies.nlp_correlation import NLPCorrelationStrategy
from src.trading.data_modules import PredictionMarketDB, DEFAULT_DB_PATH
from src.trading.engine import TradingEngine, EngineConfig


def load_market_data(
    db_path: str = DEFAULT_DB_PATH,
    limit: Optional[int] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load market data for optimization.
    
    Returns:
        (markets_df, prices_df)
    """
    db = PredictionMarketDB(db_path)
    
    # Load markets
    markets_df = db.get_all_markets()
    if limit:
        markets_df = markets_df.head(limit)
    
    # Load price history for these markets
    market_ids = markets_df["id"].tolist()
    prices_list = []
    
    for market_id in market_ids:
        prices = db.get_price_history(market_id, outcome="YES")
        if not prices.empty:
            prices_list.append(prices)
    
    if prices_list:
        prices_df = pd.concat(prices_list, ignore_index=True)
    else:
        prices_df = pd.DataFrame(columns=["market_id", "datetime", "price"])
    
    db.close()
    
    return markets_df, prices_df


def prepare_market_data_for_strategy(
    markets_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    timestamp: datetime,
) -> Dict[str, Any]:
    """
    Prepare market data in format expected by strategy.
    """
    # Convert markets to list of dicts
    markets = markets_df.to_dict("records")
    
    # Get current prices
    prices = {}
    for market_id in markets_df["id"]:
        market_prices = prices_df[prices_df["market_id"] == market_id]
        market_prices = market_prices[market_prices["datetime"] <= timestamp]
        if not market_prices.empty:
            latest = market_prices.iloc[-1]
            prices[market_id] = {
                "price": float(latest["price"]),
                "datetime": latest["datetime"],
            }
    
    return {
        "markets": markets,
        "prices": prices,
    }


def run_backtest_with_params(
    params: Dict[str, Any],
    markets_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    timestamps: list[datetime],
) -> float:
    """
    Run backtest with given parameters and return Sharpe ratio.
    
    Args:
        params: Strategy parameters
        markets_df: Market metadata
        prices_df: Price history
        timestamps: Timestamps to evaluate
        
    Returns:
        Sharpe ratio (or negative if error)
    """
    try:
        # Create strategy with parameters
        strategy = NLPCorrelationStrategy(**params)
        
        # Track returns
        returns = []
        positions = {}  # market_id -> entry_price
        
        for timestamp in timestamps:
            # Prepare market data
            market_data = prepare_market_data_for_strategy(
                markets_df, prices_df, timestamp
            )
            
            # Generate signals
            signals = strategy.generate_signals(market_data, timestamp)
            
            # Simple execution: track entry/exit
            for signal in signals:
                market_id = signal.market_id
                current_price = market_data["prices"].get(market_id, {}).get("price", 0.5)
                
                if signal.signal_type.value == "BUY":
                    if market_id not in positions:
                        positions[market_id] = current_price
                elif signal.signal_type.value == "SELL":
                    if market_id in positions:
                        entry_price = positions.pop(market_id)
                        # Calculate return
                        ret = (current_price - entry_price) / entry_price
                        returns.append(ret)
        
        # Close remaining positions at final price
        final_timestamp = timestamps[-1]
        final_market_data = prepare_market_data_for_strategy(
            markets_df, prices_df, final_timestamp
        )
        for market_id, entry_price in positions.items():
            final_price = final_market_data["prices"].get(market_id, {}).get("price", 0.5)
            ret = (final_price - entry_price) / entry_price
            returns.append(ret)
        
        if not returns:
            return -1.0  # No trades
        
        returns_array = np.array(returns)
        
        # Calculate Sharpe ratio (annualized)
        mean_return = returns_array.mean()
        std_return = returns_array.std()
        
        if std_return == 0:
            return 0.0
        
        # Assume daily returns, annualize
        sharpe = (mean_return / std_return) * np.sqrt(252)
        
        return float(sharpe)
        
    except Exception as e:
        print(f"Error in backtest: {e}")
        import traceback
        traceback.print_exc()
        return -10.0  # Penalty for errors


def optimize_nlp_correlation_strategy(
    db_path: str = DEFAULT_DB_PATH,
    method: str = "random_search",
    n_iterations: int = 50,
    market_limit: Optional[int] = 100,
    output_path: str = "data/output/nlp_correlation_tuning.json",
):
    """
    Optimize NLP correlation strategy parameters.
    
    Args:
        db_path: Database path
        method: Optimization method ("grid_search", "random_search", "bayesian_optimization")
        n_iterations: Number of iterations
        market_limit: Limit number of markets for faster optimization
        output_path: Where to save results
    """
    print("=" * 60)
    print("NLP CORRELATION STRATEGY OPTIMIZATION")
    print("=" * 60)
    
    # Load data
    print("\n[1] Loading market data...")
    markets_df, prices_df = load_market_data(db_path, limit=market_limit)
    print(f"  Loaded {len(markets_df)} markets")
    print(f"  Loaded {len(prices_df)} price records")
    
    if markets_df.empty or prices_df.empty:
        print("  ERROR: No data loaded")
        return
    
    # Prepare timestamps (sample hourly for a period)
    print("\n[2] Preparing timestamps...")
    if "datetime" in prices_df.columns:
        prices_df["datetime"] = pd.to_datetime(prices_df["datetime"])
        start_time = prices_df["datetime"].min()
        end_time = prices_df["datetime"].max()
        timestamps = pd.date_range(start_time, end_time, freq="6H").tolist()
        # Limit to reasonable number
        if len(timestamps) > 100:
            timestamps = timestamps[::len(timestamps)//100 + 1][:100]
    else:
        # Fallback: use current time
        timestamps = [datetime.now()]
    
    print(f"  Using {len(timestamps)} timestamps")
    
    # Define parameter space
    print("\n[3] Setting up parameter space...")
    parameter_spaces = {
        "similarity_method": ParameterSpace(
            name="similarity_method",
            param_type="categorical",
            values=["bag_of_words", "tfidf", "hybrid"],
        ),
        "min_similarity": ParameterSpace(
            name="min_similarity",
            param_type="continuous",
            bounds=(0.2, 0.6),
        ),
        "z_score_threshold": ParameterSpace(
            name="z_score_threshold",
            param_type="continuous",
            bounds=(1.5, 3.0),
        ),
        "lookback_hours": ParameterSpace(
            name="lookback_hours",
            param_type="discrete",
            values=[24, 48, 72, 96, 120],
        ),
        "min_spread_pct": ParameterSpace(
            name="min_spread_pct",
            param_type="continuous",
            bounds=(0.02, 0.10),
        ),
        "max_spread_pct": ParameterSpace(
            name="max_spread_pct",
            param_type="continuous",
            bounds=(0.20, 0.40),
        ),
        "price_correlation_weight": ParameterSpace(
            name="price_correlation_weight",
            param_type="continuous",
            bounds=(0.1, 0.5),
        ),
    }
    
    # Create objective function
    def objective(params: Dict[str, Any]) -> float:
        """Objective: maximize Sharpe ratio."""
        return run_backtest_with_params(
            params, markets_df, prices_df, timestamps
        )
    
    # Create tuner
    tuner = ParameterTuner(
        objective_function=objective,
        parameter_spaces=parameter_spaces,
        maximize=True,
    )
    
    # Run optimization
    print(f"\n[4] Running optimization ({method}, {n_iterations} iterations)...")
    result = tuner.optimize(method=method, n_iterations=n_iterations)
    
    # Print results
    print("\n" + "=" * 60)
    print("OPTIMIZATION RESULTS")
    print("=" * 60)
    print(f"\nBest Parameters:")
    for key, value in result.best_params.items():
        print(f"  {key}: {value}")
    print(f"\nBest Score (Sharpe Ratio): {result.best_score:.4f}")
    print(f"Method: {result.optimization_method}")
    print(f"Iterations: {result.n_iterations}")
    
    # Save results
    print(f"\n[5] Saving results to {output_path}...")
    save_tuning_result(result, output_path)
    print("  Done!")
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Optimize NLP Correlation Strategy"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=DEFAULT_DB_PATH,
        help="Database path",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="random_search",
        choices=["grid_search", "random_search", "bayesian_optimization"],
        help="Optimization method",
    )
    parser.add_argument(
        "--n-iterations",
        type=int,
        default=50,
        help="Number of iterations",
    )
    parser.add_argument(
        "--market-limit",
        type=int,
        default=100,
        help="Limit number of markets",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/output/nlp_correlation_tuning.json",
        help="Output path for results",
    )
    
    args = parser.parse_args()
    
    optimize_nlp_correlation_strategy(
        db_path=args.db_path,
        method=args.method,
        n_iterations=args.n_iterations,
        market_limit=args.market_limit,
        output_path=args.output,
    )

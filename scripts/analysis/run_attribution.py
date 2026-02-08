#!/usr/bin/env python3
"""
Run Performance Attribution Analysis

Analyzes backtest results and breaks down performance by various dimensions.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))

from src.trading.attribution import (
    generate_attribution_report,
    attribution_report_to_dataframe,
)
from src.backtest.storage import load_backtest_result
from src.backtest.catalog import BacktestCatalog
import pandas as pd


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run performance attribution analysis on backtest results"
    )
    parser.add_argument(
        "--run-id",
        type=str,
        help="Backtest run ID to analyze"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        help="Strategy name (if run-id not provided, uses latest)"
    )
    parser.add_argument(
        "--trades-csv",
        type=str,
        help="Path to trades CSV file (alternative to run-id)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/output/attribution_analysis.csv",
        help="Output path for attribution report"
    )
    parser.add_argument(
        "--time-period",
        type=str,
        default="month",
        choices=["day", "week", "month", "quarter", "year"],
        help="Time period for breakdown"
    )
    
    args = parser.parse_args()
    
    # Load trades data
    if args.trades_csv:
        print(f"Loading trades from {args.trades_csv}...")
        trades_df = pd.read_csv(args.trades_csv)
    elif args.run_id:
        print(f"Loading backtest result {args.run_id}...")
        result = load_backtest_result(args.run_id, args.strategy)
        trades_df = result.get("trades_df")
        if trades_df is None:
            print("Error: No trades_df found in backtest result")
            return 1
    else:
        # Use catalog to find latest
        catalog = BacktestCatalog()
        if args.strategy:
            runs = catalog.search(strategy_name=args.strategy, limit=1)
        else:
            runs = catalog.search(limit=1)
        
        if not runs:
            print("Error: No backtest results found")
            return 1
        
        run = runs[0]
        print(f"Using latest run: {run.run_id}")
        result = load_backtest_result(run.run_id, run.strategy_name)
        trades_df = result.get("trades_df")
        if trades_df is None:
            print("Error: No trades_df found in backtest result")
            return 1
    
    print(f"Loaded {len(trades_df)} trades")
    
    # Generate attribution report
    print("Generating attribution report...")
    report = generate_attribution_report(
        trades_df=trades_df,
        time_period=args.time_period,
    )
    
    # Convert to DataFrame
    df = attribution_report_to_dataframe(report)
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"\nAttribution report saved to {output_path}")
    print("\nSummary:")
    print(df.groupby("dimension")[["trade_count", "total_pnl", "win_rate", "sharpe_ratio"]].agg({
        "trade_count": "sum",
        "total_pnl": "sum",
        "win_rate": "mean",
        "sharpe_ratio": "mean",
    }))
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

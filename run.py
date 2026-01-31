#!/usr/bin/env python3
"""
Whale Following Strategy - Main Entry Point

Run the complete whale following strategy analysis on Manifold Markets data.

Usage:
    python run.py                    # Full analysis with default settings
    python run.py --method win_rate_60pct  # Specific whale method
    python run.py --position-size 500      # Custom position size
    python run.py --help             # Show all options

Output:
    data/output/summary.csv          # Strategy comparison
    data/output/trades_*.csv         # Individual trade logs
    data/output/quantstats_*.html    # Interactive QuantStats reports
"""

import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Ensure src is importable
import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.whale_strategy import (
    load_manifold_data,
    load_markets,
    build_resolution_map,
    identify_whales,
    run_backtest,
    print_result,
    generate_all_reports,
    categorize_market,
    CostModel,
    DataFetcher,
    WHALE_METHODS,
    COST_ASSUMPTIONS,
)
from src.whale_strategy.data import train_test_split


def main():
    parser = argparse.ArgumentParser(
        description="Whale Following Strategy Backtest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run.py
    python run.py --method win_rate_60pct --position-size 250
    python run.py --all-methods
    python run.py --category ai_tech
        """,
    )

    parser.add_argument(
        "--method",
        choices=list(WHALE_METHODS.keys()),
        default="volume_pct95",
        help="Whale identification method (default: volume_pct95)",
    )
    parser.add_argument(
        "--all-methods",
        action="store_true",
        help="Run all whale identification methods",
    )
    parser.add_argument(
        "--position-size",
        type=float,
        default=100,
        help="Position size in dollars (default: 100)",
    )
    parser.add_argument(
        "--cost-model",
        choices=list(COST_ASSUMPTIONS.keys()),
        default="small",
        help="Cost model assumptions (default: small)",
    )
    parser.add_argument(
        "--category",
        help="Filter to specific market category",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.3,
        help="Train/test split ratio (default: 0.3)",
    )
    parser.add_argument(
        "--data-dir",
        default="data/manifold",
        help="Directory with Manifold data",
    )
    parser.add_argument(
        "--output-dir",
        default="data/output",
        help="Output directory for reports",
    )
    parser.add_argument(
        "--no-reports",
        action="store_true",
        help="Skip generating HTML reports",
    )
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="Fetch fresh data from APIs before running",
    )
    parser.add_argument(
        "--fetch-only",
        action="store_true",
        help="Only fetch data, don't run backtest",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("WHALE FOLLOWING STRATEGY BACKTEST")
    print("=" * 70)

    # Handle fetch options
    if args.fetch or args.fetch_only:
        print("\n[0] Fetching data from APIs...")
        fetcher = DataFetcher(str(Path(args.data_dir).parent))
        fetcher.fetch_all()
        if args.fetch_only:
            print("\nFetch complete. Use 'python run.py' to run backtest.")
            return

    # Load data
    print("\n[1] Loading data...")
    df = load_manifold_data(args.data_dir)
    print(f"    Loaded {len(df):,} bets")
    print(f"    Date range: {df['datetime'].min()} to {df['datetime'].max()}")

    markets_df = load_markets(args.data_dir)
    print(f"    Loaded {len(markets_df):,} markets")

    # Build resolution map
    print("\n[2] Building resolution map...")
    resolution_data = build_resolution_map(markets_df)
    print(f"    Resolved YES/NO markets: {len(resolution_data):,}")

    # Add categories
    for mid, data in resolution_data.items():
        data["category"] = categorize_market(data.get("question", ""))

    # Train/test split
    print("\n[3] Train/test split...")
    train_df, test_df = train_test_split(df, args.train_ratio)
    print(f"    Training: {len(train_df):,} bets ({args.train_ratio*100:.0f}%)")
    print(f"    Testing:  {len(test_df):,} bets ({(1-args.train_ratio)*100:.0f}%)")

    # Cost model
    cost_model = CostModel.from_assumptions(args.cost_model)
    print(f"\n[4] Cost model: {COST_ASSUMPTIONS[args.cost_model].name}")
    print(f"    Base spread: {cost_model.base_spread*100:.1f}%")
    print(f"    Base slippage: {cost_model.base_slippage*100:.1f}%")

    # Determine methods to run
    methods = list(WHALE_METHODS.keys()) if args.all_methods else [args.method]

    # Run backtests
    print(f"\n[5] Running backtests...")
    results = {}

    for method in methods:
        print(f"\n    Method: {method} - {WHALE_METHODS[method]}")

        # Identify whales
        whales = identify_whales(train_df, resolution_data, method)
        print(f"    Identified {len(whales)} whales")

        # Run backtest
        result = run_backtest(
            test_df=test_df,
            whale_set=whales,
            resolution_data=resolution_data,
            strategy_name=f"{method}",
            cost_model=cost_model,
            position_size=args.position_size,
            category_filter=args.category,
        )

        if result:
            results[method] = result
            print_result(result)
        else:
            print("    No valid trades")
            results[method] = None

    # Generate reports
    if not args.no_reports:
        print(f"\n[6] Generating reports...")
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        generate_all_reports(results, args.output_dir)

    print("\n" + "=" * 70)
    print("BACKTEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

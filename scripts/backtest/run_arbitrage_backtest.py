#!/usr/bin/env python3
"""
Cross-platform arbitrage backtest: Polymarket vs Kalshi.

Matches equivalent markets across both platforms, identifies spread
opportunities, and backtests a convergence-based arbitrage strategy.

Usage:
    python scripts/backtest/run_arbitrage_backtest.py
    python scripts/backtest/run_arbitrage_backtest.py --entry-spread 0.08 --position-size 200
    python scripts/backtest/run_arbitrage_backtest.py --min-similarity 0.3 --show-matches
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Project root setup
_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root / "src"))

import pandas as pd


def _load_config():
    """Load configuration (mirrors pattern from other backtest scripts)."""
    try:
        from config import get_config
        return get_config()
    except Exception:
        return None


def _load_markets(platform: str, data_dir: Path) -> pd.DataFrame:
    """Load market metadata from parquet."""
    if platform == "polymarket":
        markets_path = data_dir / "polymarket" / "markets.parquet"
        if not markets_path.exists():
            # Check alternative location
            markets_path = data_dir / "parquet" / "markets" / "markets.parquet"
        if not markets_path.exists():
            # Try to find any markets parquet
            for p in (data_dir / "polymarket").glob("*.parquet"):
                if "market" in p.stem.lower():
                    markets_path = p
                    break
    elif platform == "kalshi":
        markets_path = data_dir / "kalshi" / "markets.parquet"
    else:
        raise ValueError(f"Unknown platform: {platform}")

    if not markets_path.exists():
        return pd.DataFrame()

    return pd.read_parquet(markets_path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cross-platform arbitrage backtest: Polymarket vs Kalshi",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic run with defaults
    python scripts/backtest/run_arbitrage_backtest.py

    # Tighter spreads, bigger positions
    python scripts/backtest/run_arbitrage_backtest.py --entry-spread 0.03 --position-size 500

    # Show matched markets without running backtest
    python scripts/backtest/run_arbitrage_backtest.py --show-matches --min-confidence 0.5

    # Custom data directories
    python scripts/backtest/run_arbitrage_backtest.py --data-dir ./my_data
        """,
    )

    # Data paths
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Root data directory (default: data)",
    )

    # Market matching parameters
    match_group = parser.add_argument_group("Market Matching")
    match_group.add_argument(
        "--min-similarity",
        type=float,
        default=0.25,
        help="Minimum TF-IDF cosine similarity for matching (default: 0.25)",
    )
    match_group.add_argument(
        "--min-confidence",
        type=float,
        default=0.40,
        help="Minimum overall match confidence (default: 0.40)",
    )
    match_group.add_argument(
        "--max-pairs",
        type=int,
        default=200,
        help="Maximum number of matched pairs (default: 200)",
    )
    match_group.add_argument(
        "--max-kalshi",
        type=int,
        default=50000,
        help="Cap Kalshi markets by volume to avoid OOM (default: 50000)",
    )
    match_group.add_argument(
        "--max-poly",
        type=int,
        default=None,
        help="Cap Polymarket markets by volume (default: None)",
    )
    match_group.add_argument(
        "--show-matches",
        action="store_true",
        help="Show matched markets and exit (no backtest)",
    )

    # Strategy parameters
    strat_group = parser.add_argument_group("Strategy")
    strat_group.add_argument(
        "--entry-spread",
        type=float,
        default=0.05,
        help="Minimum spread to enter arb position (default: 0.05 = 5 cents)",
    )
    strat_group.add_argument(
        "--exit-spread",
        type=float,
        default=0.01,
        help="Close when spread narrows to this (default: 0.01)",
    )
    strat_group.add_argument(
        "--min-profit",
        type=float,
        default=0.02,
        help="Minimum expected profit after fees (default: 0.02)",
    )
    strat_group.add_argument(
        "--z-threshold",
        type=float,
        default=1.5,
        help="Z-score threshold for entry signals (default: 1.5)",
    )
    strat_group.add_argument(
        "--lookback",
        type=int,
        default=30,
        help="Rolling lookback periods for spread stats (default: 30)",
    )
    strat_group.add_argument(
        "--max-hold",
        type=int,
        default=None,
        help="Maximum holding periods before forced exit (default: None)",
    )
    strat_group.add_argument(
        "--resample-freq",
        type=str,
        default="1D",
        help="Price resampling frequency (default: 1D = daily)",
    )

    # Backtest parameters
    bt_group = parser.add_argument_group("Backtest")
    bt_group.add_argument(
        "--position-size",
        type=float,
        default=100.0,
        help="Position size per side in USD (default: 100)",
    )
    bt_group.add_argument(
        "--starting-capital",
        type=float,
        default=10000.0,
        help="Starting capital in USD (default: 10000)",
    )
    bt_group.add_argument(
        "--max-positions",
        type=int,
        default=20,
        help="Maximum concurrent arb positions (default: 20)",
    )
    bt_group.add_argument(
        "--slippage-bps",
        type=float,
        default=50.0,
        help="Slippage in basis points per side (default: 50 = 0.5%%)",
    )

    # Fee overrides
    fee_group = parser.add_argument_group("Fees")
    fee_group.add_argument(
        "--poly-profit-fee",
        type=float,
        default=0.0,
        help="Polymarket profit fee rate (default: 0.0)",
    )
    fee_group.add_argument(
        "--kalshi-profit-fee",
        type=float,
        default=0.07,
        help="Kalshi profit fee rate (default: 0.07 = 7%%)",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="backtests/arbitrage",
        help="Output directory for results (default: backtests/arbitrage)",
    )
    parser.add_argument(
        "--save-matches",
        type=str,
        default=None,
        help="Save matched pairs to CSV (optional)",
    )

    args = parser.parse_args()
    t0 = time.time()
    data_dir = Path(args.data_dir)

    print("=" * 65)
    print("CROSS-PLATFORM ARBITRAGE BACKTEST")
    print("Polymarket vs Kalshi")
    print("=" * 65)

    # ====================================================================
    # Step 1: Load market metadata
    # ====================================================================
    print("\n[1/4] Loading market metadata...")

    poly_markets = _load_markets("polymarket", data_dir)
    kalshi_markets = _load_markets("kalshi", data_dir)

    # Cap by volume to avoid OOM (17M Kalshi markets = huge TF-IDF)
    if args.max_poly is not None and len(poly_markets) > args.max_poly:
        vol_col = "volume" if "volume" in poly_markets.columns else "volume24hr"
        if vol_col in poly_markets.columns:
            vol = pd.to_numeric(poly_markets[vol_col], errors="coerce").fillna(0)
            poly_markets = poly_markets.loc[vol.nlargest(args.max_poly).index].reset_index(drop=True)
        else:
            poly_markets = poly_markets.head(args.max_poly)
        print(f"  Polymarket capped to top {args.max_poly:,} by volume")
    if args.max_kalshi and len(kalshi_markets) > args.max_kalshi:
        vol_col = "volume" if "volume" in kalshi_markets.columns else "volume_24h"
        if vol_col in kalshi_markets.columns:
            vol = pd.to_numeric(kalshi_markets[vol_col], errors="coerce").fillna(0)
            kalshi_markets = kalshi_markets.loc[vol.nlargest(args.max_kalshi).index].reset_index(drop=True)
        else:
            kalshi_markets = kalshi_markets.head(args.max_kalshi)
        print(f"  Kalshi capped to top {args.max_kalshi:,} by volume")

    if poly_markets.empty:
        print("  ERROR: No Polymarket markets found.")
        print(f"  Expected: {data_dir / 'polymarket' / 'markets.parquet'}")
        print("  Run: python run.py --fetch-clob")
        return 1

    if kalshi_markets.empty:
        print("  ERROR: No Kalshi markets found.")
        print(f"  Expected: {data_dir / 'kalshi' / 'markets.parquet'}")
        print("  Run: python run.py --fetch-kalshi --kalshi-min-volume 100000")
        return 1

    print(f"  Polymarket: {len(poly_markets):,} markets")
    print(f"  Kalshi:     {len(kalshi_markets):,} markets")

    # ====================================================================
    # Step 2: Match markets across platforms
    # ====================================================================
    print("\n[2/4] Matching markets across platforms...")

    from trading.arbitrage.market_matcher import MarketMatcher

    matcher = MarketMatcher(
        min_similarity=args.min_similarity,
        min_confidence=args.min_confidence,
    )

    pairs = matcher.match(poly_markets, kalshi_markets, max_pairs=args.max_pairs)
    print(f"  Found {len(pairs):,} matched pairs")

    if not pairs:
        print("  No matches found. Try lowering --min-similarity or --min-confidence.")
        return 1

    # Display top matches
    print(f"\n  Top matches:")
    for p in pairs[:10]:
        print(f"    [{p.confidence:.3f}] {p.polymarket_question[:40]}")
        print(f"             = {p.kalshi_title[:40]}")

    if args.save_matches:
        matches_df = matcher.match_to_dataframe(
            poly_markets, kalshi_markets, max_pairs=args.max_pairs
        )
        matches_df.to_csv(args.save_matches, index=False)
        print(f"\n  Saved {len(matches_df)} matches to {args.save_matches}")

    if args.show_matches:
        print("\n  --show-matches: exiting without backtest")
        return 0

    # ====================================================================
    # Step 3: Generate arbitrage signals
    # ====================================================================
    print("\n[3/4] Generating arbitrage signals...")

    from trading.arbitrage.cross_platform_price_store import CrossPlatformPriceStore
    from trading.arbitrage.arbitrage_strategy import (
        ArbitrageStrategy,
        PlatformFees,
    )

    price_store = CrossPlatformPriceStore(
        polymarket_prices_dir=str(data_dir / "polymarket" / "prices"),
        kalshi_prices_dir=str(data_dir / "kalshi" / "prices"),
    )

    poly_avail, kalshi_avail = price_store.available()
    print(f"  Polymarket prices: {'available' if poly_avail else 'NOT FOUND'}")
    print(f"  Kalshi prices:     {'available' if kalshi_avail else 'NOT FOUND'}")

    if not poly_avail or not kalshi_avail:
        print("\n  ERROR: Price data missing for one or both platforms.")
        if not poly_avail:
            print("  Run: python run.py --fetch-clob")
        if not kalshi_avail:
            print("  Run: python run.py --fetch-kalshi --kalshi-min-volume 100000")
        return 1

    poly_fees = PlatformFees(
        name="Polymarket",
        profit_fee_rate=args.poly_profit_fee,
        maker_fee_rate=0.0,
        taker_fee_rate=0.0,
        withdrawal_fee=0.0,
    )
    kalshi_fees = PlatformFees(
        name="Kalshi",
        profit_fee_rate=args.kalshi_profit_fee,
        maker_fee_rate=0.0,
        taker_fee_rate=0.0,
        withdrawal_fee=0.0,
    )

    strategy = ArbitrageStrategy(
        entry_spread=args.entry_spread,
        exit_spread=args.exit_spread,
        min_expected_profit=args.min_profit,
        lookback_periods=args.lookback,
        z_score_threshold=args.z_threshold,
        max_holding_periods=args.max_hold,
        poly_fees=poly_fees,
        kalshi_fees=kalshi_fees,
    )

    print(f"  Strategy parameters:")
    for k, v in strategy.get_parameters().items():
        if not isinstance(v, dict):
            print(f"    {k}: {v}")

    signals = strategy.generate_signals(
        pairs=pairs,
        price_store=price_store,
        resample_freq=args.resample_freq,
    )

    entry_signals = [s for s in signals if s.signal_type == "entry"]
    exit_signals = [s for s in signals if s.signal_type == "exit"]
    print(f"\n  Generated {len(signals):,} signals ({len(entry_signals)} entries, {len(exit_signals)} exits)")

    if not signals:
        print("  No arbitrage opportunities found.")
        print("  Try: lower --entry-spread, lower --z-threshold, or more data.")
        return 0

    # ====================================================================
    # Step 4: Run backtest
    # ====================================================================
    print("\n[4/4] Running arbitrage backtest...")

    from trading.arbitrage.cross_platform_backtest import (
        ArbitrageBacktestConfig,
        run_arbitrage_backtest,
        print_arbitrage_result,
    )

    bt_config = ArbitrageBacktestConfig(
        strategy_name=f"Poly-Kalshi Arb (entry={args.entry_spread})",
        position_size=args.position_size,
        starting_capital=args.starting_capital,
        max_concurrent_positions=args.max_positions,
        slippage_bps=args.slippage_bps,
        poly_fees=poly_fees,
        kalshi_fees=kalshi_fees,
    )

    result = run_arbitrage_backtest(
        signals=signals,
        price_store=price_store,
        config=bt_config,
    )
    result.pairs_analyzed = len(pairs)

    # Print results
    print_arbitrage_result(result)

    # ====================================================================
    # Save results
    # ====================================================================
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp_str = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"run_{timestamp_str}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save trades
    if not result.trades_df.empty:
        trades_path = run_dir / "trades.parquet"
        result.trades_df.to_parquet(trades_path, index=False)
        print(f"\n  Trades saved: {trades_path}")

        # Also save CSV for easy inspection
        csv_path = run_dir / "trades.csv"
        result.trades_df.to_csv(csv_path, index=False)

    # Save config
    config_data = {
        "strategy": strategy.get_parameters(),
        "backtest": {
            "position_size": args.position_size,
            "starting_capital": args.starting_capital,
            "max_positions": args.max_positions,
            "slippage_bps": args.slippage_bps,
        },
        "matching": matcher.get_parameters(),
        "pairs_matched": len(pairs),
        "signals_generated": len(signals),
        "elapsed_seconds": time.time() - t0,
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config_data, f, indent=2, default=str)

    # Save matched pairs
    matches_df = matcher.match_to_dataframe(
        poly_markets, kalshi_markets, max_pairs=args.max_pairs,
    )
    if not matches_df.empty:
        matches_df.to_csv(run_dir / "matched_pairs.csv", index=False)

    elapsed = time.time() - t0
    print(f"\n  Results saved: {run_dir}")
    print(f"  Total time: {elapsed:.1f}s")

    return 0


if __name__ == "__main__":
    sys.exit(main())

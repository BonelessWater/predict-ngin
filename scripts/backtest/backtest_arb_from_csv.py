#!/usr/bin/env python3
"""
Backtest arbitrage strategy from scan CSV (snapshot opportunities).

Reads opportunities from data/arb/opportunities.csv, filters for edge >= 10%,
allocates $500 total per position with proper hedge sizing based on contract
prices (cost per hedged unit = sum of YES + NO prices = 1 - edge).

Usage:
    python scripts/backtest/backtest_arb_from_csv.py
    python scripts/backtest/backtest_arb_from_csv.py --csv data/arb/opportunities.csv
    python scripts/backtest/backtest_arb_from_csv.py --min-edge 0.15 --position-size 1000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Project root
_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))

import numpy as np
import pandas as pd

DEFAULT_CSV = "data/arb/opportunities.csv"
KALSHI_PROFIT_FEE = 0.07  # ~7% on profits


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backtest arb strategy from scan CSV: edge>=10%%, $500/position",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/backtest/backtest_arb_from_csv.py
    python scripts/backtest/backtest_arb_from_csv.py --min-edge 0.15 --position-size 1000
        """,
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=DEFAULT_CSV,
        help=f"Input CSV path (default: {DEFAULT_CSV})",
    )
    parser.add_argument(
        "--min-edge",
        type=float,
        default=0.0,
        help="Minimum edge (return) to take position (default: 0 = all positive)",
    )
    parser.add_argument(
        "--position-size",
        type=float,
        default=500.0,
        help="Total capital per position in USD (default: 500). Split across both legs.",
    )
    parser.add_argument(
        "--kalshi-fee",
        type=float,
        default=KALSHI_PROFIT_FEE,
        help=f"Kalshi profit fee rate (default: {KALSHI_PROFIT_FEE})",
    )
    args = parser.parse_args()

    # Resolve path
    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = _project_root / csv_path

    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}")
        print("  Run: python scripts/analysis/scan_arbitrage_opportunities.py -o data/arb/opportunities.csv")
        return 1

    df = pd.read_csv(csv_path)
    if "edge" not in df.columns:
        print("ERROR: CSV must have 'edge' column")
        return 1

    # Filter for min edge (strictly positive when min_edge=0)
    eligible = df[df["edge"] > args.min_edge].copy()
    eligible = eligible.sort_values("edge", ascending=False).reset_index(drop=True)

    if eligible.empty:
        thresh = "> 0" if args.min_edge == 0 else f">= {args.min_edge:.0%}"
        print(f"No opportunities with edge {thresh}")
        print(f"  Total in CSV: {len(df)}")
        print(f"  Max edge: {df['edge'].max():.2%}" if not df.empty else "")
        return 0

    # Hedged arb: cost per unit = sum of (YES on cheap side) + (NO on expensive side)
    # = 1 - edge. N = position_size / cost_per_unit for $position_size total.
    poly_yes = pd.to_numeric(eligible["poly_yes_price"], errors="coerce").fillna(0.5)
    kalshi_yes = pd.to_numeric(eligible["kalshi_yes_price"], errors="coerce").fillna(0.5)
    edge = eligible["edge"].values

    # cost_per_unit = 1 - edge (valid for both buy_kalshi and buy_poly)
    cost_per_unit = np.maximum(1.0 - edge, 0.01)  # avoid div by zero
    n_contracts = args.position_size / cost_per_unit

    # Allocations per leg (spend on each side)
    # buy_kalshi: Kalshi YES @ kalshi_yes, Poly NO @ (1 - poly_yes)
    # buy_poly:   Poly YES @ poly_yes, Kalshi NO @ (1 - kalshi_yes)
    is_buy_kalshi = (eligible["direction"] == "buy_kalshi").values
    spend_kalshi = pd.Series(
        np.where(is_buy_kalshi, n_contracts * kalshi_yes, n_contracts * (1 - kalshi_yes))
    )
    spend_poly = pd.Series(
        np.where(is_buy_kalshi, n_contracts * (1 - poly_yes), n_contracts * poly_yes)
    )
    eligible["n_contracts"] = n_contracts
    eligible["spend_kalshi"] = spend_kalshi
    eligible["spend_poly"] = spend_poly
    eligible["capital"] = spend_kalshi + spend_poly  # should equal position_size

    # Gross profit at resolution: receive N per contract, paid position_size
    gross_profit = n_contracts - args.position_size
    # Equivalently: position_size * edge / (1 - edge)
    eligible["gross_profit"] = gross_profit
    eligible["kalshi_fee"] = eligible["gross_profit"] * args.kalshi_fee
    eligible["net_profit"] = eligible["gross_profit"] * (1 - args.kalshi_fee)

    total_net_profit = eligible["net_profit"].sum()
    total_gross_profit = eligible["gross_profit"].sum()
    total_fees = eligible["kalshi_fee"].sum()

    n_positions = len(eligible)
    max_capital = n_positions * args.position_size

    # Print results
    print("=" * 65)
    print("ARBITRAGE BACKTEST (Snapshot Strategy)")
    print("Hold-to-resolution from scan CSV (hedged positions)")
    print("=" * 65)
    print(f"  Input:       {csv_path}")
    edge_str = "> 0 (all positive)" if args.min_edge == 0 else f">= {args.min_edge:.0%}"
    print(f"  Min edge:    {edge_str}")
    print(f"  Position:    ${args.position_size:,.0f} total per pair (hedged)")
    print()
    print("--- Capital ---")
    print(f"  Positions:   {n_positions:,}")
    print(f"  Max capital: ${max_capital:,.0f}")
    print()
    print("--- P&L (simulated hold-to-resolution) ---")
    print(f"  Gross profit: ${total_gross_profit:,.2f}")
    print(f"  Kalshi fees:  ${total_fees:,.2f}")
    print(f"  Net profit:   ${total_net_profit:,.2f}")
    if max_capital > 0:
        print(f"  Return:       {100 * total_net_profit / max_capital:.2f}%")
    print()
    print("--- Top 10 positions ---")
    for i, row in eligible.head(10).iterrows():
        q = str(row.get("polymarket_question", ""))[:45]
        print(f"  [{row['edge']:.2%}] ${row['net_profit']:.0f} | {q}...")
    print("=" * 65)

    # Save backtest positions to data/arb/
    out_dir = _project_root / "data" / "arb"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "backtest_positions.csv"
    cols = [
        "edge", "direction", "polymarket_question", "kalshi_title",
        "poly_yes_price", "kalshi_yes_price", "n_contracts", "spend_poly",
        "spend_kalshi", "capital", "gross_profit", "kalshi_fee", "net_profit",
        "polymarket_url", "kalshi_url",
    ]
    out_cols = [c for c in cols if c in eligible.columns]
    eligible[out_cols].to_csv(out_path, index=False)
    print(f"\n  Positions saved: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

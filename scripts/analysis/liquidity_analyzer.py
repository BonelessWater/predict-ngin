#!/usr/bin/env python3
"""
Liquidity Analyzer and Visualization Tool

Analyzes liquidity data from the database and generates reports/visualizations.
"""

import argparse
import sqlite3
from pathlib import Path
from typing import Optional, Dict, List
import pandas as pd
import numpy as np
import sys

# Add project root to path
_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))

from src.config import get_config


def load_liquidity_snapshots(
    db_path: str,
    market_id: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Load liquidity snapshots from database."""
    conn = sqlite3.connect(db_path)
    
    query = "SELECT * FROM liquidity_snapshots WHERE 1=1"
    params = []
    
    if market_id:
        query += " AND market_id = ?"
        params.append(market_id)
    
    if start_date:
        query += " AND datetime >= ?"
        params.append(start_date)
    
    if end_date:
        query += " AND datetime <= ?"
        params.append(end_date)
    
    query += " ORDER BY datetime"
    
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    
    if not df.empty:
        df["datetime"] = pd.to_datetime(df["datetime"])
    
    return df


def load_liquidity_estimates(
    db_path: str,
    market_id: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Load liquidity estimates from database."""
    conn = sqlite3.connect(db_path)
    
    query = "SELECT * FROM liquidity_estimates WHERE 1=1"
    params = []
    
    if market_id:
        query += " AND market_id = ?"
        params.append(market_id)
    
    if start_date:
        query += " AND date >= ?"
        params.append(start_date)
    
    if end_date:
        query += " AND date <= ?"
        params.append(end_date)
    
    query += " ORDER BY date"
    
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    
    return df


def analyze_liquidity_trends(snapshots_df: pd.DataFrame) -> Dict:
    """Analyze liquidity trends over time."""
    if snapshots_df.empty:
        return {}
    
    # Group by market and date
    snapshots_df = snapshots_df.copy()
    snapshots_df["date"] = snapshots_df["datetime"].dt.date
    
    trends = {}
    
    for market_id in snapshots_df["market_id"].unique():
        market_data = snapshots_df[snapshots_df["market_id"] == market_id].copy()
        market_data = market_data.sort_values("datetime")
        
        # Calculate average depth over time
        avg_bid_depth = market_data.groupby("date")["bid_depth_10pct"].mean()
        avg_ask_depth = market_data.groupby("date")["ask_depth_10pct"].mean()
        avg_spread = market_data.groupby("date")["spread"].mean()
        
        trends[market_id] = {
            "avg_bid_depth": avg_bid_depth.to_dict(),
            "avg_ask_depth": avg_ask_depth.to_dict(),
            "avg_spread": avg_spread.to_dict(),
            "snapshot_count": len(market_data),
            "date_range": (market_data["date"].min(), market_data["date"].max()),
        }
    
    return trends


def analyze_market_liquidity(
    snapshots_df: pd.DataFrame,
    estimates_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Analyze liquidity by market."""
    if snapshots_df.empty:
        return pd.DataFrame()
    
    market_stats = []
    
    for market_id in snapshots_df["market_id"].unique():
        market_data = snapshots_df[snapshots_df["market_id"] == market_id].copy()
        
        stats = {
            "market_id": market_id,
            "snapshot_count": len(market_data),
            "avg_bid_depth_1pct": market_data["bid_depth_1pct"].mean(),
            "avg_bid_depth_5pct": market_data["bid_depth_5pct"].mean(),
            "avg_bid_depth_10pct": market_data["bid_depth_10pct"].mean(),
            "avg_ask_depth_1pct": market_data["ask_depth_1pct"].mean(),
            "avg_ask_depth_5pct": market_data["ask_depth_5pct"].mean(),
            "avg_ask_depth_10pct": market_data["ask_depth_10pct"].mean(),
            "avg_spread": market_data["spread"].mean(),
            "min_spread": market_data["spread"].min(),
            "max_spread": market_data["spread"].max(),
            "avg_midpoint": market_data["midpoint"].mean(),
            "first_snapshot": market_data["datetime"].min(),
            "last_snapshot": market_data["datetime"].max(),
        }
        
        # Add estimated liquidity if available
        if estimates_df is not None and not estimates_df.empty:
            market_estimates = estimates_df[estimates_df["market_id"] == market_id]
            if not market_estimates.empty:
                stats["avg_estimated_liquidity"] = market_estimates["estimated_liquidity"].mean()
                stats["avg_trade_impact"] = market_estimates["avg_trade_impact"].mean()
        
        market_stats.append(stats)
    
    return pd.DataFrame(market_stats)


def identify_liquidity_issues(
    snapshots_df: pd.DataFrame,
    min_depth_threshold: float = 1000.0,
    max_spread_threshold: float = 0.05,
) -> pd.DataFrame:
    """Identify markets with liquidity issues."""
    if snapshots_df.empty:
        return pd.DataFrame()
    
    issues = []
    
    for market_id in snapshots_df["market_id"].unique():
        market_data = snapshots_df[snapshots_df["market_id"] == market_id].copy()
        
        # Check for low depth
        low_depth_count = (
            (market_data["bid_depth_10pct"] < min_depth_threshold) |
            (market_data["ask_depth_10pct"] < min_depth_threshold)
        ).sum()
        
        # Check for wide spreads
        wide_spread_count = (market_data["spread"] > max_spread_threshold).sum()
        
        if low_depth_count > 0 or wide_spread_count > 0:
            issues.append({
                "market_id": market_id,
                "low_depth_snapshots": low_depth_count,
                "wide_spread_snapshots": wide_spread_count,
                "pct_low_depth": low_depth_count / len(market_data) * 100,
                "pct_wide_spread": wide_spread_count / len(market_data) * 100,
                "avg_spread": market_data["spread"].mean(),
                "min_depth": min(
                    market_data["bid_depth_10pct"].min(),
                    market_data["ask_depth_10pct"].min()
                ),
            })
    
    return pd.DataFrame(issues)


def generate_liquidity_report(
    db_path: str,
    output_path: Optional[str] = None,
    market_id: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> None:
    """Generate comprehensive liquidity report."""
    
    print("Loading liquidity data...")
    snapshots_df = load_liquidity_snapshots(db_path, market_id, start_date, end_date)
    estimates_df = load_liquidity_estimates(db_path, market_id, start_date, end_date)
    
    if snapshots_df.empty:
        print("No liquidity snapshots found in database.")
        return
    
    print(f"Loaded {len(snapshots_df)} liquidity snapshots")
    if not estimates_df.empty:
        print(f"Loaded {len(estimates_df)} liquidity estimates")
    
    # Analyze by market
    print("\n=== Market Liquidity Analysis ===")
    market_analysis = analyze_market_liquidity(snapshots_df, estimates_df)
    if not market_analysis.empty:
        print(market_analysis.to_string(index=False))
    
    # Identify issues
    print("\n=== Liquidity Issues ===")
    issues = identify_liquidity_issues(snapshots_df)
    if not issues.empty:
        print(issues.to_string(index=False))
    else:
        print("No liquidity issues detected.")
    
    # Trends
    print("\n=== Liquidity Trends ===")
    trends = analyze_liquidity_trends(snapshots_df)
    print(f"Analyzed {len(trends)} markets")
    
    # Save report
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            f.write("Liquidity Analysis Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Market Liquidity Analysis:\n")
            f.write(market_analysis.to_string(index=False))
            f.write("\n\n")
            
            f.write("Liquidity Issues:\n")
            f.write(issues.to_string(index=False))
            f.write("\n")
        
        print(f"\nReport saved to {output_path}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze liquidity data from the database"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Path to SQLite database (default: from config)"
    )
    parser.add_argument(
        "--market-id",
        type=str,
        default=None,
        help="Filter by specific market ID"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for report (default: print to stdout)"
    )
    
    args = parser.parse_args()
    
    # Get database path from config if not provided
    if args.db_path is None:
        config = get_config()
        args.db_path = config.database.path
    
    if not Path(args.db_path).exists():
        print(f"Database not found: {args.db_path}")
        return 1
    
    generate_liquidity_report(
        db_path=args.db_path,
        output_path=args.output,
        market_id=args.market_id,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

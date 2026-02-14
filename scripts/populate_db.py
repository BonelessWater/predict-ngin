#!/usr/bin/env python3
"""
Quick script to populate the database for NLP correlation strategy.

DEPRECATED: Use data/research and fetch_research_trades_and_prices.py instead.
This script will be removed.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.trading.data_modules.fetcher import DataFetcher
from src.trading.data_modules.database import build_database, DEFAULT_DB_PATH


def main():
    print("⚠️  DEPRECATED: Use data/research and fetch_research_trades_and_prices.py instead.")
    print("=" * 60)
    print("POPULATING DATABASE FOR NLP CORRELATION STRATEGY")
    print("=" * 60)
    
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Step 1: Fetch Polymarket markets
    print("\n[1] Fetching Polymarket markets from API...")
    fetcher = DataFetcher(str(data_dir))
    
    # Fetch markets (limit to reasonable number for testing)
    print("  Fetching active markets...")
    num_markets = fetcher.fetch_polymarket_markets(
        limit=500,  # Start with 500 markets
        batch_size=100,
    )
    print(f"  Fetched {num_markets} markets")
    
    # Step 2: Fetch CLOB price data for those markets
    print("\n[2] Fetching CLOB price data...")
    print("  This may take a while...")
    fetcher.fetch_polymarket_clob_data(
        min_volume=1000,  # Minimum volume filter
        max_markets=500,  # Match the markets we fetched
    )
    
    # Step 3: Build database from fetched data (Polymarket only, no Manifold)
    print("\n[3] Building database from fetched data (Polymarket only)...")
    db = build_database(
        db_path=DEFAULT_DB_PATH,
        import_polymarket=True,
        import_manifold=False,  # Skip Manifold entirely
        import_manifold_markets=False,  # Skip Manifold markets
        import_polymarket_trades=False,  # Skip trades for now
    )
    
    # Show stats
    print("\n" + "=" * 60)
    print("DATABASE STATS")
    print("=" * 60)
    
    markets_count = db.query("SELECT COUNT(*) as count FROM polymarket_markets").iloc[0]["count"]
    prices_count = db.query("SELECT COUNT(*) as count FROM polymarket_prices").iloc[0]["count"]
    
    print(f"Markets: {markets_count:,}")
    print(f"Price records: {prices_count:,}")
    
    if markets_count == 0:
        print("\n⚠️  WARNING: No markets found!")
        print("   Make sure data/polymarket_clob/ contains JSON files")
        print("   Try running: python run.py --fetch-clob --clob-min-volume 1000")
    elif prices_count == 0:
        print("\n⚠️  WARNING: No price data found!")
        print("   Price data should be in data/polymarket_clob/")
        print("   Try running: python run.py --fetch-clob")
    else:
        print("\n✅ Database populated successfully!")
        print("   Prefer: python scripts/data/fetch_research_trades_and_prices.py --research-dir data/research")
    
    db.close()


if __name__ == "__main__":
    main()

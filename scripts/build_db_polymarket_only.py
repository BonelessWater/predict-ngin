#!/usr/bin/env python3
"""
Build database with Polymarket data only (no Manifold).
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.trading.data_modules.database import build_database, DEFAULT_DB_PATH


def main():
    print("Building database with Polymarket data only (no Manifold)...")
    
    db = build_database(
        db_path=DEFAULT_DB_PATH,
        import_polymarket=True,
        import_manifold=False,  # Skip Manifold bets
        import_manifold_markets=False,  # Skip Manifold markets
        import_polymarket_trades=False,  # Optional: set to True if you have trades
    )
    
    # Show stats
    markets_count = db.query("SELECT COUNT(*) as count FROM polymarket_markets").iloc[0]["count"]
    prices_count = db.query("SELECT COUNT(*) as count FROM polymarket_prices").iloc[0]["count"]
    
    print(f"\nDatabase built:")
    print(f"  Polymarket markets: {markets_count:,}")
    print(f"  Polymarket prices: {prices_count:,}")
    
    db.close()


if __name__ == "__main__":
    main()

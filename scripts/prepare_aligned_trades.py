#!/usr/bin/env python3
"""
Prepare aligned trade data by joining new trades to markets via slug.

This creates a table with properly aligned market_ids that can be
joined to the prices table for accurate backtesting.

Usage:
    python scripts/prepare_aligned_trades.py
"""

import sqlite3
from pathlib import Path


def prepare_aligned_trades(db_path: str = "data/prediction_markets.db") -> None:
    """Create aligned trades table with proper market IDs."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    print("=" * 60)
    print("PREPARING ALIGNED TRADE DATA")
    print("=" * 60)

    # Drop existing table if it exists
    print("\nDropping existing aligned trades table...")
    cur.execute("DROP TABLE IF EXISTS polymarket_trades_aligned")

    # Create aligned trades table by joining on slug
    print("Creating aligned trades table (join via slug)...")
    cur.execute("""
        CREATE TABLE polymarket_trades_aligned AS
        SELECT
            t.id as original_id,
            m.id as market_id,
            t.slug,
            m.question,
            t.timestamp,
            t.proxy_wallet,
            t.side,
            t.outcome,
            t.outcome_index,
            t.price,
            t.size,
            t.transaction_hash
        FROM polymarket_trades_new t
        INNER JOIN polymarket_markets m ON t.slug = m.slug
        WHERE t.proxy_wallet IS NOT NULL
    """)

    # Create indexes
    print("Creating indexes...")
    cur.execute("CREATE INDEX idx_aligned_market ON polymarket_trades_aligned(market_id)")
    cur.execute("CREATE INDEX idx_aligned_wallet ON polymarket_trades_aligned(proxy_wallet)")
    cur.execute("CREATE INDEX idx_aligned_timestamp ON polymarket_trades_aligned(timestamp)")
    cur.execute("CREATE INDEX idx_aligned_slug ON polymarket_trades_aligned(slug)")

    conn.commit()

    # Get stats
    print("\n--- Aligned Trades Stats ---")
    cur.execute("SELECT COUNT(*) FROM polymarket_trades_aligned")
    total = cur.fetchone()[0]
    print(f"Total aligned trades: {total:,}")

    cur.execute("SELECT COUNT(DISTINCT market_id) FROM polymarket_trades_aligned")
    markets = cur.fetchone()[0]
    print(f"Unique markets: {markets:,}")

    cur.execute("SELECT COUNT(DISTINCT proxy_wallet) FROM polymarket_trades_aligned")
    wallets = cur.fetchone()[0]
    print(f"Unique wallets: {wallets:,}")

    cur.execute("SELECT MIN(timestamp), MAX(timestamp) FROM polymarket_trades_aligned")
    row = cur.fetchone()
    print(f"Date range: {row[0]} to {row[1]}")

    # Verify price data exists for these markets
    print("\n--- Price Data Verification ---")
    cur.execute("""
        SELECT COUNT(DISTINCT t.market_id)
        FROM polymarket_trades_aligned t
        WHERE EXISTS (
            SELECT 1 FROM polymarket_prices p
            WHERE p.market_id = t.market_id
            LIMIT 1
        )
    """)
    markets_with_prices = cur.fetchone()[0]
    print(f"Markets with price history: {markets_with_prices} / {markets}")

    conn.close()

    print("\n" + "=" * 60)
    print("ALIGNED DATA READY")
    print("=" * 60)
    print(f"\nTable 'polymarket_trades_aligned' created with {total:,} trades")
    print("These trades can now be joined to polymarket_prices via market_id")


if __name__ == "__main__":
    prepare_aligned_trades()

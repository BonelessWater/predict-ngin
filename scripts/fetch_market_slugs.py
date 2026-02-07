#!/usr/bin/env python3
"""
Fetch historical market slugs from Polymarket Gamma API.

This script fetches all markets (not just closed ones) to get slug mappings
that can be used to link trades to resolution data.

Usage:
    python scripts/fetch_market_slugs.py
"""

import json
import time
import sqlite3
import requests
from pathlib import Path
from typing import Optional, Dict, Any, List

GAMMA_API = "https://gamma-api.polymarket.com"
REQUEST_DELAY = 0.1


def fetch_markets_batch(
    limit: int = 100,
    offset: int = 0,
    closed: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Fetch markets from Gamma API."""
    session = requests.Session()
    session.headers.update({"User-Agent": "PredictionMarketResearch/1.0"})

    params = {"limit": limit, "offset": offset}
    if closed is not None:
        params["closed"] = closed

    try:
        response = session.get(
            f"{GAMMA_API}/markets",
            params=params,
            timeout=30,
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching markets: {e}")
        return []


def fetch_all_markets(max_markets: int = 50000) -> List[Dict[str, Any]]:
    """Fetch all markets from Gamma API."""
    all_markets = []
    offset = 0
    batch_size = 100

    print("Fetching all markets from Polymarket Gamma API...")

    while len(all_markets) < max_markets:
        time.sleep(REQUEST_DELAY)
        batch = fetch_markets_batch(limit=batch_size, offset=offset)

        if not batch:
            break

        all_markets.extend(batch)
        offset += len(batch)

        if len(all_markets) % 1000 == 0:
            print(f"  Fetched {len(all_markets):,} markets...")

        if len(batch) < batch_size:
            break

    print(f"  Total markets fetched: {len(all_markets):,}")
    return all_markets


def save_slug_mapping(
    markets: List[Dict[str, Any]],
    db_path: str = "data/prediction_markets.db",
) -> int:
    """Save market ID to slug mapping to database."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Create mapping table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS market_slug_map (
            gamma_id TEXT PRIMARY KEY,
            slug TEXT,
            question TEXT,
            condition_id TEXT,
            clob_token_ids TEXT
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_slug_map_slug ON market_slug_map(slug)")

    inserted = 0
    for m in markets:
        gamma_id = m.get("id")
        slug = m.get("slug")
        question = m.get("question", "")[:200]
        condition_id = m.get("conditionId")
        clob_token_ids = json.dumps(m.get("clobTokenIds", []))

        if gamma_id and slug:
            try:
                cur.execute("""
                    INSERT OR REPLACE INTO market_slug_map
                    (gamma_id, slug, question, condition_id, clob_token_ids)
                    VALUES (?, ?, ?, ?, ?)
                """, (gamma_id, slug, question, condition_id, clob_token_ids))
                inserted += 1
            except Exception as e:
                pass

    conn.commit()
    conn.close()

    return inserted


def update_trades_with_slugs(db_path: str = "data/prediction_markets.db") -> int:
    """
    Add slug column to trades table and populate from mapping.

    Note: This may not work if trade market_id doesn't match gamma_id.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Check if slug column exists
    cur.execute("PRAGMA table_info(polymarket_trades)")
    cols = {row[1] for row in cur.fetchall()}

    if "slug" not in cols:
        print("Adding slug column to polymarket_trades...")
        cur.execute("ALTER TABLE polymarket_trades ADD COLUMN slug TEXT")
        conn.commit()

    # Try to update from mapping
    print("Attempting to update trades with slugs from mapping...")
    cur.execute("""
        UPDATE polymarket_trades
        SET slug = (
            SELECT slug FROM market_slug_map
            WHERE market_slug_map.gamma_id = polymarket_trades.market_id
        )
        WHERE slug IS NULL
    """)
    updated = cur.rowcount
    conn.commit()
    conn.close()

    return updated


def analyze_market_id_formats(db_path: str = "data/prediction_markets.db"):
    """Analyze the different market ID formats to understand the mapping."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    print("\n=== Market ID Format Analysis ===\n")

    # Trade market IDs
    print("Trade market IDs (polymarket_trades):")
    cur.execute("SELECT DISTINCT market_id FROM polymarket_trades LIMIT 10")
    for row in cur.fetchall():
        print(f"  {row[0]}")

    # Gamma API IDs
    print("\nGamma API IDs (market_slug_map):")
    cur.execute("SELECT gamma_id, slug FROM market_slug_map LIMIT 10")
    for row in cur.fetchall():
        print(f"  {row[0]} -> {row[1]}")

    # Condition IDs
    print("\nCondition IDs (market_slug_map):")
    cur.execute("SELECT condition_id FROM market_slug_map WHERE condition_id IS NOT NULL LIMIT 5")
    for row in cur.fetchall():
        cid = row[0]
        if cid:
            print(f"  {cid[:50]}...")

    # CLOB token IDs
    print("\nCLOB Token IDs (market_slug_map):")
    cur.execute("SELECT clob_token_ids FROM market_slug_map WHERE clob_token_ids != '[]' LIMIT 3")
    for row in cur.fetchall():
        print(f"  {row[0]}")

    # Check if trade market_id matches any format
    print("\n=== Checking ID overlap ===")

    # Direct gamma_id match
    cur.execute("""
        SELECT COUNT(DISTINCT t.market_id)
        FROM polymarket_trades t
        INNER JOIN market_slug_map m ON CAST(t.market_id AS TEXT) = m.gamma_id
    """)
    gamma_match = cur.fetchone()[0]
    print(f"Trade IDs matching Gamma IDs: {gamma_match}")

    conn.close()


def save_to_json(markets: List[Dict[str, Any]], path: str = "data/market_slugs.json"):
    """Save markets to JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    # Extract relevant fields
    mapping = []
    for m in markets:
        mapping.append({
            "gamma_id": m.get("id"),
            "slug": m.get("slug"),
            "question": m.get("question", "")[:200] if m.get("question") else None,
            "condition_id": m.get("conditionId"),
            "clob_token_ids": m.get("clobTokenIds"),
        })

    with open(path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)

    print(f"  Saved {len(mapping):,} markets to {path}")


def main():
    print("=" * 60)
    print("MARKET SLUG FETCHER")
    print("=" * 60)

    # Fetch all markets
    markets = fetch_all_markets()

    if not markets:
        print("No markets fetched")
        return

    # Save to JSON first (always works)
    print("\nSaving to JSON...")
    save_to_json(markets)

    # Try to save to database
    print("\nSaving slug mapping to database...")
    try:
        inserted = save_slug_mapping(markets)
        print(f"  Saved {inserted:,} market slug mappings")

        # Analyze ID formats
        analyze_market_id_formats()

        # Try to update trades
        updated = update_trades_with_slugs()
        print(f"\nUpdated {updated:,} trades with slugs")
    except Exception as e:
        print(f"  Database error: {e}")
        print("  Slug mapping saved to JSON - import later with:")
        print("  python scripts/fetch_market_slugs.py --import-json")

    print("\n" + "=" * 60)
    print("SLUG FETCH COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

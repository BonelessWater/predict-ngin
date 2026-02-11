#!/usr/bin/env python3
"""
Scan currently open Polymarket and Kalshi markets for cross-platform arbitrage opportunities.

Fetches live data from both APIs, filters by:
  - Currently open markets only
  - Optional: --min-days-open to require established liquidity
  - Volume > 50k (USD for Polymarket, contracts for Kalshi)
  - Category: tech (ai_tech) by default

Matches equivalent markets using TF-IDF + cosine similarity (NLP), then computes
current spread and ranks by arbitrage opportunity. Outputs CSV sorted by edge (best first).

Typical runtime: 2-5 min (Poly ~2-3 min, Kalshi ~30s, matching ~10s).
Use --quick for ~30s scan (limit 500 per platform).

Usage:
    python scripts/analysis/scan_arbitrage_opportunities.py
    python scripts/analysis/scan_arbitrage_opportunities.py --quick  # Faster, fewer markets
    python scripts/analysis/scan_arbitrage_opportunities.py --min-volume 100000
    python scripts/analysis/scan_arbitrage_opportunities.py --no-category-filter -o best_trades.csv
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Project root
_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))

import pandas as pd
import requests


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

POLYMARKET_GAMMA_API = "https://gamma-api.polymarket.com"
KALSHI_API = "https://api.elections.kalshi.com/trade-api/v2"
DEFAULT_MIN_VOLUME_POLY = 10000.0   # USD (more liquid)
DEFAULT_MIN_VOLUME_KALSHI = 10000   # contracts (more liquid)
DEFAULT_MIN_DAYS_OPEN = 0  # No minimum; include all by default
DEFAULT_MAX_DAYS_OPEN: float | None = None  # None = no max
REQUEST_DELAY = 0.15


# ---------------------------------------------------------------------------
# Polymarket fetcher
# ---------------------------------------------------------------------------

def fetch_polymarket_open_markets(
    min_volume: float = DEFAULT_MIN_VOLUME_POLY,
    min_days_open: float = DEFAULT_MIN_DAYS_OPEN,
    max_days_open: float | None = DEFAULT_MAX_DAYS_OPEN,
    limit: int = 5000,
    max_scanned: int | None = None,
    max_requests: int = 100,
    debug: bool = False,
) -> pd.DataFrame:
    """
    Fetch currently open Polymarket markets with volume and age filters.

    Scanning = downloading: each API request downloads a page of ~100 markets.
    No server-side volume filter, so we must paginate until we have enough.
    Stops after max_requests to avoid 10+ minute scans.

    Returns DataFrame with: id, question, slug, volume, volume24hr, yes_price,
    start_date, days_open, outcomes, outcomePrices.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=min_days_open)
    cutoff_ts = cutoff.timestamp()

    session = requests.Session()
    session.headers.update({"User-Agent": "PredictionMarketResearch/1.0"})

    rows = []
    offset = 0
    batch_size = 100
    request_count = 0

    while len(rows) < limit:
        request_count += 1
        if request_count > max_requests:
            print(f"    [Poly] stopping at {max_requests} requests (kept {len(rows):,})", flush=True)
            break
        if max_scanned is not None and offset >= max_scanned:
            break
        try:
            resp = session.get(
                f"{POLYMARKET_GAMMA_API}/markets",
                params={
                    "limit": batch_size,
                    "offset": offset,
                    "closed": "false",
                    "active": "true",
                },
                timeout=60,
            )
            if resp.status_code == 429:
                time.sleep(2)
                continue
            resp.raise_for_status()
            batch = resp.json()
        except requests.RequestException as e:
            print(f"  Polymarket API error: {e}")
            break

        if not batch:
            break

        batch_vol_pass = 0
        batch_days_min_pass = 0
        batch_days_max_pass = 0
        for m in batch:
            vol = float(m.get("volume", 0) or m.get("volumeNum", 0) or 0)
            if vol < min_volume:
                continue
            batch_vol_pass += 1

            # Parse start date (use startDate or createdAt)
            start_str = m.get("startDate") or m.get("createdAt") or m.get("created_at")
            if not start_str:
                continue
            try:
                if start_str.endswith("Z"):
                    start_dt = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
                else:
                    start_dt = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
                start_ts = start_dt.timestamp()
            except (ValueError, TypeError):
                continue

            if start_ts > cutoff_ts:
                continue  # Opened less than min_days_open ago
            batch_days_min_pass += 1

            days_open = (datetime.now(timezone.utc).timestamp() - start_ts) / 86400
            if max_days_open is not None and days_open > max_days_open:
                continue  # Opened more than max_days_open ago
            batch_days_max_pass += 1

            # Parse YES price
            prices_str = m.get("outcomePrices", "")
            if isinstance(prices_str, str) and prices_str:
                try:
                    prices = json.loads(prices_str)
                    yes_price = float(prices[0]) if isinstance(prices, list) and len(prices) >= 1 else 0.5
                except json.JSONDecodeError:
                    yes_price = float(m.get("bestBid", 0) or m.get("lastTradePrice", 0.5) or 0.5)
            else:
                yes_price = float(m.get("bestBid", 0) or m.get("lastTradePrice", 0.5) or 0.5)

            rows.append({
                "id": str(m.get("id", "")),
                "question": str(m.get("question", ""))[:500],
                "slug": m.get("slug", ""),
                "volume": vol,
                "volume24hr": float(m.get("volume24hr", 0) or m.get("volume24hrClob", 0) or 0),
                "yes_price": yes_price,
                "start_date": start_str,
                "days_open": days_open,
                "outcomes": m.get("outcomes"),
                "outcomePrices": m.get("outcomePrices"),
                "endDate": m.get("endDate"),
            })

        offset += len(batch)
        if debug and len(batch) > 0:
            print(f"    [Poly] req {request_count}: batch={len(batch)}, vol>={min_volume:,.0f}->{batch_vol_pass}, "
                  f"days_ok->{batch_days_max_pass}, kept={len(rows):,}", flush=True)
        elif offset % 500 == 0 or request_count % 5 == 0:
            print(f"    [Poly] req {request_count}, scanned {offset:,}, kept {len(rows):,}...", flush=True)
        if len(rows) >= limit:
            print(f"    [Poly] got {len(rows):,} (target {limit}), stopping", flush=True)
            break
        if len(batch) < batch_size:
            break
        time.sleep(REQUEST_DELAY)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Kalshi fetcher
# ---------------------------------------------------------------------------

def fetch_kalshi_open_markets(
    min_volume: float = DEFAULT_MIN_VOLUME_KALSHI,
    min_days_open: float = DEFAULT_MIN_DAYS_OPEN,
    max_days_open: float | None = DEFAULT_MAX_DAYS_OPEN,
    limit: int = 5000,
) -> pd.DataFrame:
    """
    Fetch currently open Kalshi markets with volume and age filters.

    Returns DataFrame with: ticker, title, volume, yes_price, open_time,
    days_open, status, event_ticker.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=min_days_open)
    cutoff_ts = cutoff.timestamp()

    session = requests.Session()
    session.headers.update({"User-Agent": "PredictionMarketResearch/1.0"})

    rows = []
    cursor = ""
    page = 0

    while len(rows) < limit:
        page += 1
        try:
            params = {"limit": 200, "status": "open"}
            if cursor:
                params["cursor"] = cursor

            resp = session.get(
                f"{KALSHI_API}/markets",
                params=params,
                timeout=60,
            )
            if resp.status_code == 429:
                time.sleep(2)
                continue
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            print(f"  Kalshi API error: {e}")
            break

        markets = data.get("markets", [])
        if not markets:
            break

        for m in markets:
            status = (m.get("status") or "").lower()
            if status not in ("open", "active"):
                continue
            vol = int(m.get("volume", 0) or 0)
            if vol < min_volume:
                continue

            open_str = m.get("open_time") or m.get("created_time", "")
            if not open_str:
                continue
            try:
                # Kalshi uses ISO format
                if "T" in open_str:
                    open_dt = datetime.fromisoformat(open_str.replace("Z", "+00:00"))
                else:
                    open_dt = datetime.fromisoformat(open_str.replace("Z", "+00:00"))
                open_ts = open_dt.timestamp()
            except (ValueError, TypeError):
                continue

            if open_ts > cutoff_ts:
                continue

            days_open = (datetime.now(timezone.utc).timestamp() - open_ts) / 86400
            if max_days_open is not None and days_open > max_days_open:
                continue

            # Kalshi prices are in cents (0-100)
            last = m.get("last_price") or m.get("yes_bid") or m.get("yes_ask")
            if last is not None:
                yes_price = float(last) / 100.0
            else:
                yes_price = 0.5

            rows.append({
                "ticker": str(m.get("ticker", "")),
                "title": str(m.get("title", "") or m.get("subtitle", ""))[:500],
                "volume": vol,
                "volume_24h": int(m.get("volume_24h", 0) or 0),
                "yes_price": yes_price,
                "open_time": open_str,
                "days_open": days_open,
                "status": m.get("status", ""),
                "event_ticker": m.get("event_ticker", ""),
                "category": str(m.get("category", "")),
            })

        cursor = data.get("cursor", "")
        if page % 5 == 0 or len(rows) >= limit:
            print(f"    [Kalshi] page {page}, {len(rows):,} kept...", flush=True)
        if not cursor:
            break
        time.sleep(REQUEST_DELAY)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def fetch_kalshi_via_events(
    min_volume: float = DEFAULT_MIN_VOLUME_KALSHI,
    min_days_open: float = DEFAULT_MIN_DAYS_OPEN,
    max_days_open: float | None = DEFAULT_MAX_DAYS_OPEN,
    limit: int = 5000,
    max_pages: int = 100,
    debug: bool = False,
) -> pd.DataFrame:
    """
    Fetch open Kalshi markets via /events endpoint (high-volume first, ~10x faster).
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=min_days_open)
    cutoff_ts = cutoff.timestamp()

    session = requests.Session()
    session.headers.update({"User-Agent": "PredictionMarketResearch/1.0"})

    rows = []
    cursor = ""
    page = 0

    while len(rows) < limit and page < max_pages:
        page += 1
        try:
            params = {
                "limit": 200,
                "with_nested_markets": "true",
                "status": "open",
            }
            if cursor:
                params["cursor"] = cursor

            resp = session.get(
                f"{KALSHI_API}/events",
                params=params,
                timeout=60,
            )
            if resp.status_code == 429:
                time.sleep(2)
                continue
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            print(f"  Kalshi API error: {e}")
            break

        events = data.get("events", [])
        if not events:
            break

        page_vol_pass = 0
        for ev in events:
            for m in ev.get("markets", []):
                status = (m.get("status") or "").lower()
                if status not in ("open", "active"):
                    continue
                vol = int(m.get("volume", 0) or 0)
                if vol == 0:
                    vol = int(float(m.get("volume_fp", 0) or 0))
                if vol < min_volume:
                    continue
                page_vol_pass += 1

                open_str = m.get("open_time") or m.get("created_time", "") or ev.get("created_time", "")
                if not open_str:
                    continue
                try:
                    open_dt = datetime.fromisoformat(open_str.replace("Z", "+00:00"))
                    open_ts = open_dt.timestamp()
                except (ValueError, TypeError):
                    continue

                if open_ts > cutoff_ts:
                    continue

                days_open = (datetime.now(timezone.utc).timestamp() - open_ts) / 86400
                if max_days_open is not None and days_open > max_days_open:
                    continue

                last = m.get("last_price") or m.get("yes_bid") or m.get("yes_ask")
                yes_price = float(last) / 100.0 if last is not None else 0.5

                rows.append({
                    "ticker": str(m.get("ticker", "")),
                    "title": str(m.get("title", "") or m.get("subtitle", ""))[:500],
                    "volume": vol,
                    "volume_24h": int(m.get("volume_24h", 0) or 0),
                    "yes_price": yes_price,
                    "open_time": open_str,
                    "status": m.get("status", ""),
                    "event_ticker": ev.get("event_ticker", ""),
                    "category": str(ev.get("category", "") or m.get("category", "")),
                })
                if len(rows) >= limit:
                    break

        if debug:
            n_events = len(events)
            n_markets = sum(len(ev.get("markets", [])) for ev in events)
            print(f"    [Kalshi] page {page}: {n_events} events, {n_markets} mkts, vol>={min_volume:,.0f}->{page_vol_pass}, total_kept={len(rows):,}", flush=True)
        else:
            print(f"    [Kalshi] page {page}, {len(rows):,} kept...", flush=True)
        cursor = data.get("cursor", "")
        if not cursor or len(rows) >= limit:
            break
        time.sleep(REQUEST_DELAY)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Load from local parquet (fallback)
# ---------------------------------------------------------------------------

def load_polymarket_from_parquet(data_dir: Path) -> pd.DataFrame:
    """Load Polymarket markets from parquet, filter for open + volume."""
    candidates: list[Path] = []
    for p in [
        data_dir / "polymarket" / "markets.parquet",
        data_dir / "parquet" / "markets" / "markets.parquet",
    ]:
        if p.exists():
            candidates.append(p)
    for base in [data_dir / "polymarket", data_dir / "parquet" / "markets"]:
        if base.exists():
            candidates.extend(base.glob("markets_*.parquet"))
    dfs = []
    for p in sorted(set(candidates)):
        try:
            df = pd.read_parquet(p)
        except Exception:
            continue
        if df.empty:
            continue
        dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    if "id" in df.columns:
        df = df.drop_duplicates(subset=["id"], keep="last")
    # Normalize columns
    if "id" not in df.columns and "market_id" in df.columns:
        df["id"] = df["market_id"].astype(str)
    if "id" not in df.columns and "conditionId" in df.columns:
        df["id"] = df["conditionId"].astype(str)
    if "question" not in df.columns and "title" in df.columns:
        df["question"] = df["title"]
    if "closed" in df.columns:
        df = df[df["closed"] == False]
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
    return df


def load_kalshi_from_parquet(data_dir: Path) -> pd.DataFrame:
    """Load Kalshi markets from parquet, filter for open + volume."""
    p = data_dir / "kalshi" / "markets.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    if "status" in df.columns:
        df = df[df["status"] == "open"]
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scan Polymarket vs Kalshi for arbitrage opportunities (output CSV)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/analysis/scan_arbitrage_opportunities.py
    python scripts/analysis/scan_arbitrage_opportunities.py --min-volume 100000
    python scripts/analysis/scan_arbitrage_opportunities.py -o data/arb/tech_arb.csv
    python scripts/analysis/scan_arbitrage_opportunities.py --no-category-filter
        """,
    )
    parser.add_argument(
        "--min-volume",
        type=float,
        default=100_000,
        help="Minimum volume (USD for Poly, contracts for Kalshi) (default: 100000)",
    )
    parser.add_argument(
        "--min-days-open",
        type=float,
        default=0,
        help="Markets must be open at least this many days (default: 0 = no min)",
    )
    parser.add_argument(
        "--max-days-open",
        type=float,
        default=None,
        metavar="N",
        help="Markets must be open at most N days (default: None = no max)",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.40,
        help="Minimum match confidence for NLP (default: 0.40)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="data/arb/opportunities.csv",
        help="Output CSV path (default: data/arb/opportunities.csv)",
    )
    parser.add_argument(
        "--use-local-data",
        action="store_true",
        help="Use local parquet files instead of live API (no 5-day filter applied)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Data directory for local parquet (default: data)",
    )
    parser.add_argument(
        "--category",
        type=str,
        default="ai_tech",
        help="Filter to category (default: ai_tech). Use 'tech' as alias. Set to '' for no filter.",
    )
    parser.add_argument(
        "--no-category-filter",
        action="store_true",
        help="Disable category filter (include all categories)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5000,
        help="Max markets to fetch per platform from live API (default: 5000)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Fast run: limit 500 per platform (~30s total)",
    )
    parser.add_argument(
        "--max-poly-requests",
        type=int,
        default=200,
        help="Max API requests for Polymarket (default: 200)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print per-batch volume/days filtering stats",
    )
    args = parser.parse_args()

    if args.quick:
        args.limit = 500

    # Normalize category: "tech" -> "ai_tech"
    category_filter = None if args.no_category_filter else (args.category or "ai_tech")
    if category_filter and category_filter.lower() == "tech":
        category_filter = "ai_tech"

    data_dir = _project_root / args.data_dir

    print("=" * 65)
    print("ARBITRAGE OPPORTUNITY SCANNER")
    print("Polymarket vs Kalshi — Fresh data from APIs, open markets only")
    max_str = str(args.max_days_open) if args.max_days_open is not None else "inf"
    filter_str = f"volume >= {args.min_volume:,.0f}, open {args.min_days_open}-{max_str} days"
    print(f"  Filters: {filter_str}")
    if category_filter:
        print(f"  Category filter: {category_filter}")
    print("=" * 65)

    # ---- Step 1: Load markets ----
    if args.use_local_data:
        print("\n[1/4] Loading from local parquet...")
        poly_df = load_polymarket_from_parquet(data_dir)
        kalshi_df = load_kalshi_from_parquet(data_dir)

        if not poly_df.empty and "volume" in poly_df.columns:
            poly_df = poly_df[pd.to_numeric(poly_df["volume"], errors="coerce") >= args.min_volume]
        if not kalshi_df.empty and "volume" in kalshi_df.columns:
            kalshi_df = kalshi_df[kalshi_df["volume"] >= args.min_volume]

        # Normalize for matcher
        if not poly_df.empty:
            if "question" not in poly_df.columns and "title" in poly_df.columns:
                poly_df["question"] = poly_df["title"]
        if not kalshi_df.empty:
            if "title" not in kalshi_df.columns and "subtitle" in kalshi_df.columns:
                kalshi_df["title"] = kalshi_df["subtitle"]
    else:
        print("\n[1/4] Fetching live markets from APIs (open only, fresh data)...", flush=True)
        print("  (Scanning = downloading each page; Poly has no volume filter, so we paginate)", flush=True)
        fetch_start = time.time()

        print("  [Poly] Starting...", flush=True)
        poly_max_scanned = 3000 if args.quick else None
        poly_df = fetch_polymarket_open_markets(
            args.min_volume,
            args.min_days_open,
            args.max_days_open,
            args.limit,
            poly_max_scanned,
            args.max_poly_requests,
            args.debug,
        )
        print(f"  [Poly] Done: {len(poly_df):,} markets", flush=True)

        print("  [Kalshi] Starting Kalshi fetch (via /events)...", flush=True)
        kalshi_max_pages = 50 if args.quick else 100
        kalshi_df = fetch_kalshi_via_events(
            args.min_volume,
            args.min_days_open,
            args.max_days_open,
            args.limit,
            kalshi_max_pages,
            args.debug,
        )
        if kalshi_df.empty:
            print("  [Kalshi] 0 from /events, fallback to /markets...", flush=True)
            kalshi_df = fetch_kalshi_open_markets(
                args.min_volume,
                args.min_days_open,
                args.max_days_open,
                args.limit,
            )
        if kalshi_df.empty and args.min_days_open > 0:
            print("  [Kalshi] 0 from /markets, retrying with min_days_open=0...", flush=True)
            kalshi_df = fetch_kalshi_open_markets(
                args.min_volume,
                min_days_open=0,
                max_days_open=args.max_days_open,
                limit=args.limit,
            )
        print(f"  [Kalshi] Done: {len(kalshi_df):,} markets", flush=True)

        print(f"  Fetch complete in {time.time() - fetch_start:.1f}s", flush=True)

    print(f"  Polymarket: {len(poly_df):,} markets (open, vol >= {args.min_volume:,.0f})")
    print(f"  Kalshi:     {len(kalshi_df):,} markets (open, vol >= {args.min_volume:,.0f})")

    # ---- Category filter ----
    if category_filter:
        from src.taxonomy.markets import categorize_market

        def _is_tech(q: str) -> bool:
            return categorize_market(str(q) if pd.notna(q) else "") == category_filter

        if not poly_df.empty and "question" in poly_df.columns:
            before = len(poly_df)
            poly_df = poly_df[poly_df["question"].apply(_is_tech)]
            print(f"  After {category_filter} filter: Polymarket {before} -> {len(poly_df)}")
        if not kalshi_df.empty:
            before = len(kalshi_df)
            # Kalshi: use category column if present and non-empty, else categorize from title
            if "category" in kalshi_df.columns and kalshi_df["category"].notna().any():
                tech_mask = (
                    kalshi_df["category"].fillna("").str.lower().str.contains("tech", regex=False)
                    | kalshi_df["category"].fillna("").str.lower().str.contains("technology", regex=False)
                )
                kalshi_df = kalshi_df[tech_mask]
            else:
                title_col = "title" if "title" in kalshi_df.columns else "subtitle"
                kalshi_df = kalshi_df[kalshi_df[title_col].fillna("").apply(_is_tech)]
            print(f"  After {category_filter} filter: Kalshi {before} -> {len(kalshi_df)}")

    if poly_df.empty or kalshi_df.empty:
        print("\n  No markets found. Try --min-volume, --no-category-filter, or --use-local-data.")
        return 1

    # ---- Step 2: Match markets ----
    print("\n[2/4] Matching markets (TF-IDF similarity, may take 10-30s)...", flush=True)

    from src.trading.arbitrage.market_matcher import MarketMatcher

    matcher = MarketMatcher(
        min_similarity=0.25,
        min_confidence=args.min_confidence,
    )
    pairs = matcher.match(poly_df, kalshi_df, max_pairs=500)
    print(f"  Matched pairs: {len(pairs)}")

    if not pairs:
        print("  No pairs matched. Try lowering --min-confidence.")
        return 1

    # ---- Step 3: Compute spread, stream output ----
    print("\n[3/4] Computing spread and streaming results...")
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = _project_root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    COLUMNS = [
        "edge", "abs_spread", "spread", "direction", "estimated_profit_after_fees",
        "polymarket_id", "kalshi_ticker", "polymarket_question", "kalshi_title",
        "poly_yes_price", "kalshi_yes_price", "match_confidence", "similarity_score",
        "poly_volume", "kalshi_volume", "polymarket_url", "kalshi_url",
    ]

    def _poly_yes_price(row: dict) -> float | None:
        """Extract YES price from Polymarket row (0-1 scale)."""
        if row.get("yes_price") is not None:
            return float(row["yes_price"])
        op = row.get("outcomePrices") or row.get("outcome_prices")
        if op is not None:
            try:
                p = json.loads(op) if isinstance(op, str) else op
                return float(p[0]) if isinstance(p, (list, tuple)) and p else None
            except (json.JSONDecodeError, TypeError, IndexError):
                pass
        return None

    def _kalshi_yes_price(row: dict) -> float | None:
        """Extract YES price from Kalshi row. API returns cents (0-100), parquet may be same."""
        v = row.get("yes_price") or row.get("last_price") or row.get("yes_bid") or row.get("yes_ask")
        if v is None:
            return None
        v = float(v)
        return v / 100.0 if v > 1.0 else v  # Convert cents to 0-1 if needed

    # Build lookup for prices
    poly_by_id = poly_df.set_index("id").to_dict("index") if "id" in poly_df.columns else {}
    kalshi_by_ticker = kalshi_df.set_index("ticker").to_dict("index") if "ticker" in kalshi_df.columns else {}

    rows = []
    for p in pairs:
        poly_row = poly_by_id.get(p.polymarket_id, {})
        kalshi_row = kalshi_by_ticker.get(p.kalshi_ticker, {})

        poly_price = _poly_yes_price(poly_row) if poly_row else None
        kalshi_price = _kalshi_yes_price(kalshi_row) if kalshi_row else None

        if poly_price is None or kalshi_price is None:
            continue

        spread = float(poly_price) - float(kalshi_price)
        abs_spread = abs(spread)

        # Direction: which side is cheap
        if spread > 0:
            direction = "buy_kalshi"  # Poly expensive
        else:
            direction = "buy_poly"   # Kalshi expensive

        # Kalshi ~7% profit fee
        kalshi_fee = 0.07
        estimated_profit = abs_spread * (1 - kalshi_fee)

        # URLs for manual verification
        poly_url = f"https://polymarket.com/event/{poly_row.get('slug', p.polymarket_id)}" if poly_row else ""
        kalshi_url = f"https://kalshi.com/markets/{p.kalshi_ticker}" if p.kalshi_ticker else ""

        rows.append({
            "edge": abs_spread,
            "abs_spread": abs_spread,
            "spread": spread,
            "direction": direction,
            "estimated_profit_after_fees": estimated_profit,
            "polymarket_id": p.polymarket_id,
            "kalshi_ticker": p.kalshi_ticker,
            "polymarket_question": p.polymarket_question[:200],
            "kalshi_title": p.kalshi_title[:200],
            "poly_yes_price": poly_price,
            "kalshi_yes_price": kalshi_price,
            "match_confidence": p.confidence,
            "similarity_score": p.similarity_score,
            "poly_volume": p.polymarket_volume,
            "kalshi_volume": p.kalshi_volume,
            "polymarket_url": poly_url,
            "kalshi_url": kalshi_url,
        })

    if not rows:
        print("  No pairs with valid prices (use live API for current prices).")
        return 1

    # Sort by edge (best trades first)
    rows.sort(key=lambda r: r["edge"], reverse=True)

    with open(out_path, "w", newline="", encoding="utf-8") as csv_handle:
        import csv as csv_module
        writer = csv_module.DictWriter(csv_handle, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    for i, r in enumerate(rows[:10]):
        print(f"  [{i+1}] edge={r['edge']:.3f} {r['direction']} | {r['polymarket_question'][:50]}...")

    print(f"\n[4/4] Done. {len(rows)} opportunities written to {out_path} (sorted by edge descending)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

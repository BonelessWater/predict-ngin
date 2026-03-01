#!/usr/bin/env python3
"""
Fetch Polymarket markets from the Gamma API and save markets_filtered.csv
per category into data/research/{Category}/.

Sources (in order):
  1. GET /events  – has structured `tags` for accurate classification
  2. GET /markets – catches any orphan markets not nested in an event

Categories produced:
  Art_and_Culture, Climate_and_Science, Economy, Finance,
  Geopolitics, Politics, Sports, Tech, Other

Usage:
  python scripts/data/fetch_markets_by_category.py
  python scripts/data/fetch_markets_by_category.py --include-closed
  python scripts/data/fetch_markets_by_category.py --min-volume 10000
  python scripts/data/fetch_markets_by_category.py --dry-run
"""

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

GAMMA_API = "https://gamma-api.polymarket.com"
REQUEST_DELAY = 0.15   # seconds between pages

# ── Category priority (earlier = wins when multiple tags match) ──────────────
CATEGORY_PRIORITY = [
    "Politics",
    "Sports",
    "Geopolitics",
    "Finance",
    "Economy",
    "Tech",
    "Climate_and_Science",
    "Art_and_Culture",
]

# ── Tag label / slug → folder category ──────────────────────────────────────
TAG_TO_CATEGORY: Dict[str, str] = {
    # Politics
    "politics": "Politics",
    "election": "Politics",
    "elections": "Politics",
    "us-elections": "Politics",
    "us elections": "Politics",
    "us-current-affairs": "Politics",
    "government": "Politics",
    "congress": "Politics",
    "senate": "Politics",
    "president": "Politics",
    "presidential": "Politics",
    "democratic": "Politics",
    "republican": "Politics",
    "political": "Politics",
    "voting": "Politics",
    "ballot": "Politics",
    "legislation": "Politics",
    "impeachment": "Politics",
    "2024 us elections": "Politics",
    "trump": "Politics",
    "biden": "Politics",
    "harris": "Politics",
    "doge": "Politics",          # Dept of Govt Efficiency (not the crypto)
    "white house": "Politics",
    "supreme court": "Politics",
    "midterms": "Politics",

    # Sports
    "sports": "Sports",
    "nba": "Sports",
    "nfl": "Sports",
    "nhl": "Sports",
    "mlb": "Sports",
    "soccer": "Sports",
    "tennis": "Sports",
    "golf": "Sports",
    "mma": "Sports",
    "boxing": "Sports",
    "olympics": "Sports",
    "chess": "Sports",
    "poker": "Sports",
    "f1": "Sports",
    "formula 1": "Sports",
    "formula1": "Sports",
    "racing": "Sports",
    "basketball": "Sports",
    "football": "Sports",
    "baseball": "Sports",
    "hockey": "Sports",
    "esports": "Sports",
    "cricket": "Sports",
    "rugby": "Sports",
    "volleyball": "Sports",
    "swimming": "Sports",
    "ncaa": "Sports",
    "premier league": "Sports",
    "bundesliga": "Sports",
    "la liga": "Sports",
    "serie a": "Sports",
    "champions league": "Sports",
    "ufc": "Sports",
    "wwe": "Sports",
    "nascar": "Sports",
    "pga": "Sports",
    "wimbledon": "Sports",
    "nba playoffs": "Sports",
    "world cup": "Sports",
    "super bowl": "Sports",
    "march madness": "Sports",
    "mlb playoffs": "Sports",
    "conmebol": "Sports",

    # Geopolitics
    "geopolitics": "Geopolitics",
    "global politics": "Geopolitics",
    "international": "Geopolitics",
    "war": "Geopolitics",
    "russia": "Geopolitics",
    "ukraine": "Geopolitics",
    "middle east": "Geopolitics",
    "china": "Geopolitics",
    "nato": "Geopolitics",
    "united nations": "Geopolitics",
    "foreign policy": "Geopolitics",
    "diplomacy": "Geopolitics",
    "conflict": "Geopolitics",
    "sanctions": "Geopolitics",
    "terrorism": "Geopolitics",
    "north korea": "Geopolitics",
    "iran": "Geopolitics",
    "israel": "Geopolitics",
    "palestine": "Geopolitics",
    "taiwan": "Geopolitics",
    "india": "Geopolitics",
    "europe": "Geopolitics",
    "eu": "Geopolitics",
    "european union": "Geopolitics",

    # Finance
    "finance": "Finance",
    "stocks": "Finance",
    "stock market": "Finance",
    "ipo": "Finance",
    "ipos": "Finance",
    "forex": "Finance",
    "bonds": "Finance",
    "derivatives": "Finance",
    "commodities": "Finance",
    "gold": "Finance",
    "oil": "Finance",
    "wall street": "Finance",
    "etf": "Finance",
    "interest rates": "Finance",
    "fed": "Finance",
    "federal reserve": "Finance",
    "inflation": "Finance",
    "s&p": "Finance",
    "nasdaq": "Finance",
    "dow": "Finance",
    "sp500": "Finance",
    "hedge fund": "Finance",
    "private equity": "Finance",
    "m&a": "Finance",

    # Economy
    "economy": "Economy",
    "business": "Economy",
    "trade": "Economy",
    "gdp": "Economy",
    "unemployment": "Economy",
    "recession": "Economy",
    "economic": "Economy",
    "commerce": "Economy",
    "tariffs": "Economy",
    "labor": "Economy",
    "supply chain": "Economy",
    "retail": "Economy",
    "housing market": "Economy",
    "real estate": "Economy",
    "jobs": "Economy",

    # Tech  (note: crypto goes here; if user later wants Finance for crypto, change)
    "tech": "Tech",
    "crypto": "Tech",
    "ai": "Tech",
    "nfts": "Tech",
    "web3": "Tech",
    "defi": "Tech",
    "blockchain": "Tech",
    "cryptocurrency": "Tech",
    "software": "Tech",
    "hardware": "Tech",
    "semiconductor": "Tech",
    "artificial intelligence": "Tech",
    "machine learning": "Tech",
    "bitcoin": "Tech",
    "ethereum": "Tech",
    "solana": "Tech",
    "chatgpt": "Tech",
    "openai": "Tech",
    "elon musk": "Tech",

    # Climate_and_Science
    "science": "Climate_and_Science",
    "climate": "Climate_and_Science",
    "environment": "Climate_and_Science",
    "coronavirus": "Climate_and_Science",
    "covid": "Climate_and_Science",
    "pandemic": "Climate_and_Science",
    "space": "Climate_and_Science",
    "health": "Climate_and_Science",
    "biology": "Climate_and_Science",
    "physics": "Climate_and_Science",
    "chemistry": "Climate_and_Science",
    "energy": "Climate_and_Science",
    "renewable": "Climate_and_Science",
    "nuclear": "Climate_and_Science",
    "weather": "Climate_and_Science",
    "astronomy": "Climate_and_Science",
    "nasa": "Climate_and_Science",

    # Art_and_Culture
    "art": "Art_and_Culture",
    "culture": "Art_and_Culture",
    "entertainment": "Art_and_Culture",
    "pop-culture": "Art_and_Culture",
    "pop culture": "Art_and_Culture",
    "movies": "Art_and_Culture",
    "music": "Art_and_Culture",
    "tv": "Art_and_Culture",
    "television": "Art_and_Culture",
    "film": "Art_and_Culture",
    "celebrity": "Art_and_Culture",
    "awards": "Art_and_Culture",
    "oscars": "Art_and_Culture",
    "grammys": "Art_and_Culture",
    "emmys": "Art_and_Culture",
    "fashion": "Art_and_Culture",
    "food": "Art_and_Culture",
    "anime": "Art_and_Culture",
    "books": "Art_and_Culture",
    "literature": "Art_and_Culture",
    "gaming": "Art_and_Culture",
    "nba (culture)": "Art_and_Culture",
}

# ── Fallback: API market.category string → folder category ──────────────────
API_CATEGORY_MAP: Dict[str, str] = {
    "us-current-affairs": "Politics",
    "global politics":    "Geopolitics",
    "sports":             "Sports",
    "nba playoffs":       "Sports",
    "olympics":           "Sports",
    "chess":              "Sports",
    "poker":              "Sports",
    "crypto":             "Tech",
    "nfts":               "Tech",
    "tech":               "Tech",
    "space":              "Climate_and_Science",
    "science":            "Climate_and_Science",
    "coronavirus":        "Climate_and_Science",
    "business":           "Economy",
    "finance":            "Finance",
    "art":                "Art_and_Culture",
    "pop-culture":        "Art_and_Culture",
    "pop culture":        "Art_and_Culture",
}


def _make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "predict-ngin/1.0 (research)"})
    retry = Retry(total=5, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s


def classify_by_tags(tags: List[dict]) -> str:
    """Return the highest-priority category matched by any tag label/slug."""
    if not tags:
        return "Other"
    matched: Set[str] = set()
    for tag in tags:
        for key in (tag.get("label", "").lower().strip(),
                    tag.get("slug", "").lower().strip()):
            if key in TAG_TO_CATEGORY:
                matched.add(TAG_TO_CATEGORY[key])
    if not matched:
        return "Other"
    for cat in CATEGORY_PRIORITY:
        if cat in matched:
            return cat
    return next(iter(matched))


def classify_by_api_category(api_cat: Optional[str]) -> str:
    """Map the raw market.category string from the API to a folder category."""
    if not api_cat:
        return "Other"
    return API_CATEGORY_MAP.get(api_cat.lower().strip(), "Other")


def fetch_events(
    session: requests.Session,
    include_closed: bool = False,
    min_volume: float = 0.0,
) -> List[dict]:
    """Paginate /events and return list of event dicts (with nested markets)."""
    events: List[dict] = []
    offset = 0
    limit = 100

    params: dict = {"limit": limit, "active": "true"}
    if not include_closed:
        params["closed"] = "false"

    print("  Fetching events …", end="", flush=True)
    while True:
        params["offset"] = offset
        try:
            r = session.get(f"{GAMMA_API}/events", params=params, timeout=30)
            r.raise_for_status()
            batch = r.json()
        except Exception as exc:
            print(f"\n  WARNING: events fetch failed at offset {offset}: {exc}")
            break

        if not batch:
            break

        for evt in batch:
            vol = float(evt.get("volume", 0) or 0)
            if vol < min_volume:
                continue
            events.append(evt)

        print(".", end="", flush=True)
        offset += len(batch)
        if len(batch) < limit:
            break
        time.sleep(REQUEST_DELAY)

    print(f" {len(events):,} events")
    return events


def fetch_all_markets(
    session: requests.Session,
    include_closed: bool = False,
    min_volume: float = 0.0,
) -> List[dict]:
    """Paginate /markets and return flat list of market dicts."""
    markets: List[dict] = []
    offset = 0
    limit = 100

    params: dict = {"limit": limit}
    if not include_closed:
        params["closed"] = "false"

    print("  Fetching markets …", end="", flush=True)
    while True:
        params["offset"] = offset
        try:
            r = session.get(f"{GAMMA_API}/markets", params=params, timeout=30)
            r.raise_for_status()
            batch = r.json()
        except Exception as exc:
            print(f"\n  WARNING: markets fetch failed at offset {offset}: {exc}")
            break

        if not batch:
            break

        for m in batch:
            vol = float(m.get("volumeNum", 0) or m.get("volume", 0) or 0)
            if vol < min_volume:
                continue
            markets.append(m)

        print(".", end="", flush=True)
        offset += len(batch)
        if len(batch) < limit:
            break
        time.sleep(REQUEST_DELAY)

    print(f" {len(markets):,} markets")
    return markets


def _normalize_market(m: dict, category: str) -> dict:
    """Return a flat dict ready for the CSV row."""
    row = dict(m)
    row["category"] = category
    # Ensure conditionId exists (needed by research_data_loader)
    if "conditionId" not in row or not row["conditionId"]:
        row["conditionId"] = str(row.get("id", ""))
    # Normalise liquidity field names for downstream compatibility
    if "liquidityNum" not in row:
        row["liquidityNum"] = row.get("liquidity", 0) or 0
    if "volumeNum" not in row:
        row["volumeNum"] = row.get("volume", 0) or 0
    # Drop nested objects (events array, etc.) that can't go in CSV cleanly
    for key in list(row.keys()):
        if isinstance(row[key], (list, dict)):
            row[key] = str(row[key])
    return row


def build_category_dataframes(
    events: List[dict],
    orphan_markets: List[dict],
    seen_ids: Optional[Set[str]] = None,
) -> Dict[str, List[dict]]:
    """
    Combine events (classified by tags) and orphan markets (classified by
    API category string) into a dict: category → list of market dicts.
    """
    if seen_ids is None:
        seen_ids = set()

    buckets: Dict[str, List[dict]] = defaultdict(list)

    # 1. Markets nested inside events
    for evt in events:
        tags = evt.get("tags") or []
        cat = classify_by_tags(tags)
        for m in evt.get("markets") or []:
            cid = str(m.get("conditionId") or m.get("id") or "").strip()
            if not cid or cid in seen_ids:
                continue
            seen_ids.add(cid)
            buckets[cat].append(_normalize_market(m, cat))

    # 2. Orphan markets not in any event
    for m in orphan_markets:
        cid = str(m.get("conditionId") or m.get("id") or "").strip()
        if not cid or cid in seen_ids:
            continue
        seen_ids.add(cid)
        cat = classify_by_api_category(m.get("category"))
        buckets[cat].append(_normalize_market(m, cat))

    return buckets


def write_category_csvs(
    buckets: Dict[str, List[dict]],
    output_dir: Path,
    dry_run: bool = False,
) -> None:
    all_cats = CATEGORY_PRIORITY + ["Other"]

    print()
    print(f"{'Category':<22}  {'Markets':>8}  {'Output'}")
    print("-" * 70)

    for cat in all_cats:
        rows = buckets.get(cat, [])
        if not rows:
            print(f"  {cat:<20}  {'0':>8}  (skipped – no markets)")
            continue

        df = pd.DataFrame(rows)

        # Ensure conditionId is always first for readability
        cols = list(df.columns)
        priority_cols = ["conditionId", "id", "question", "category",
                         "slug", "active", "closed", "volumeNum", "liquidityNum",
                         "endDateIso", "endDate", "closedTime"]
        ordered = [c for c in priority_cols if c in cols]
        rest = [c for c in cols if c not in ordered]
        df = df[ordered + rest]

        cat_dir = output_dir / cat
        out_path = cat_dir / "markets_filtered.csv"

        if dry_run:
            print(f"  {cat:<20}  {len(df):>8}  [dry-run] {out_path}")
        else:
            cat_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_path, index=False)
            print(f"  {cat:<20}  {len(df):>8}  {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch Polymarket markets from the Gamma API, split by category"
    )
    parser.add_argument(
        "--output-dir",
        default="data/research",
        help="Base output directory (default: data/research)",
    )
    parser.add_argument(
        "--include-closed",
        action="store_true",
        help="Also include closed / resolved markets (default: active only)",
    )
    parser.add_argument(
        "--min-volume",
        type=float,
        default=0.0,
        help="Minimum market volume in USD (default: 0 = all markets)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be written without creating files",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    print("=" * 70)
    print("FETCH POLYMARKET MARKETS BY CATEGORY")
    print("=" * 70)
    print(f"Output dir    : {output_dir}")
    print(f"Include closed: {args.include_closed}")
    print(f"Min volume    : ${args.min_volume:,.0f}")
    print(f"Dry run       : {args.dry_run}")
    print()

    session = _make_session()

    print("Fetching from Gamma API …")
    events = fetch_events(session, args.include_closed, args.min_volume)
    orphan_markets = fetch_all_markets(session, args.include_closed, args.min_volume)

    print()
    print("Classifying …")
    buckets = build_category_dataframes(events, orphan_markets)
    total = sum(len(v) for v in buckets.values())
    print(f"  Total unique markets: {total:,}")

    print()
    print("Writing CSVs …")
    write_category_csvs(buckets, output_dir, dry_run=args.dry_run)

    print()
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

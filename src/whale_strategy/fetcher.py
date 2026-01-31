"""
Data fetching from prediction market APIs.

Supports:
- Manifold Markets (free, no auth required)
- Polymarket (free market metadata, no auth required)
"""

import json
import time
import requests
from pathlib import Path
from typing import Optional, List
from datetime import datetime


# API endpoints
MANIFOLD_API = "https://api.manifold.markets/v0"
POLYMARKET_API = "https://gamma-api.polymarket.com"

# Rate limiting
REQUEST_DELAY = 0.1  # seconds between requests


class DataFetcher:
    """Fetches and caches prediction market data."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "PredictionMarketResearch/1.0"
        })

    def _rate_limit(self):
        time.sleep(REQUEST_DELAY)

    def _save_json(self, data: list, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    # =========================================================================
    # MANIFOLD MARKETS
    # =========================================================================

    def fetch_manifold_bets(
        self,
        limit: int = 5000000,
        batch_size: int = 1000,
        before_id: Optional[str] = None,
    ) -> int:
        """
        Fetch bets from Manifold Markets API.

        Args:
            limit: Maximum total bets to fetch
            batch_size: Bets per API request (max 1000)
            before_id: Start fetching before this bet ID

        Returns:
            Total bets fetched
        """
        output_dir = self.data_dir / "manifold"
        output_dir.mkdir(parents=True, exist_ok=True)

        total_fetched = 0
        file_num = 1
        current_batch = []

        print(f"Fetching Manifold bets (target: {limit:,})...")

        while total_fetched < limit:
            self._rate_limit()

            params = {"limit": min(batch_size, 1000)}
            if before_id:
                params["before"] = before_id

            try:
                response = self.session.get(
                    f"{MANIFOLD_API}/bets",
                    params=params
                )
                response.raise_for_status()
                bets = response.json()

                if not bets:
                    break

                current_batch.extend(bets)
                total_fetched += len(bets)
                before_id = bets[-1]["id"]

                # Save every 5000 bets
                if len(current_batch) >= 5000:
                    self._save_json(
                        current_batch,
                        output_dir / f"bets_{file_num}.json"
                    )
                    print(f"  Saved bets_{file_num}.json ({len(current_batch):,} bets, total: {total_fetched:,})")
                    current_batch = []
                    file_num += 1

                if len(bets) < batch_size:
                    break

            except requests.RequestException as e:
                print(f"  Error fetching bets: {e}")
                break

        # Save remaining
        if current_batch:
            self._save_json(
                current_batch,
                output_dir / f"bets_{file_num}.json"
            )
            print(f"  Saved bets_{file_num}.json ({len(current_batch):,} bets)")

        print(f"  Total: {total_fetched:,} bets")
        return total_fetched

    def fetch_manifold_markets(
        self,
        limit: int = 200000,
        batch_size: int = 1000,
    ) -> int:
        """
        Fetch markets from Manifold Markets API.

        Args:
            limit: Maximum markets to fetch
            batch_size: Markets per request (max 1000)

        Returns:
            Total markets fetched
        """
        output_dir = self.data_dir / "manifold"
        output_dir.mkdir(parents=True, exist_ok=True)

        total_fetched = 0
        file_num = 1
        current_batch = []
        before_id = None

        print(f"Fetching Manifold markets (target: {limit:,})...")

        while total_fetched < limit:
            self._rate_limit()

            params = {"limit": min(batch_size, 1000)}
            if before_id:
                params["before"] = before_id

            try:
                response = self.session.get(
                    f"{MANIFOLD_API}/markets",
                    params=params
                )
                response.raise_for_status()
                markets = response.json()

                if not markets:
                    break

                current_batch.extend(markets)
                total_fetched += len(markets)
                before_id = markets[-1]["id"]

                # Save every 5000 markets
                if len(current_batch) >= 5000:
                    self._save_json(
                        current_batch,
                        output_dir / f"markets_{file_num}.json"
                    )
                    print(f"  Saved markets_{file_num}.json ({len(current_batch):,} markets)")
                    current_batch = []
                    file_num += 1

                if len(markets) < batch_size:
                    break

            except requests.RequestException as e:
                print(f"  Error fetching markets: {e}")
                break

        # Save remaining
        if current_batch:
            self._save_json(
                current_batch,
                output_dir / f"markets_{file_num}.json"
            )
            print(f"  Saved markets_{file_num}.json ({len(current_batch):,} markets)")

        print(f"  Total: {total_fetched:,} markets")
        return total_fetched

    # =========================================================================
    # POLYMARKET
    # =========================================================================

    def fetch_polymarket_markets(
        self,
        limit: int = 200000,
        batch_size: int = 100,
    ) -> int:
        """
        Fetch markets from Polymarket Gamma API.

        Args:
            limit: Maximum markets to fetch
            batch_size: Markets per request (max 100)

        Returns:
            Total markets fetched
        """
        output_dir = self.data_dir / "polymarket"
        output_dir.mkdir(parents=True, exist_ok=True)

        total_fetched = 0
        file_num = 1
        current_batch = []
        offset = 0

        print(f"Fetching Polymarket markets (target: {limit:,})...")

        while total_fetched < limit:
            self._rate_limit()

            try:
                response = self.session.get(
                    f"{POLYMARKET_API}/markets",
                    params={"limit": batch_size, "offset": offset}
                )
                response.raise_for_status()
                markets = response.json()

                if not markets:
                    break

                current_batch.extend(markets)
                total_fetched += len(markets)
                offset += len(markets)

                # Save every 5000 markets
                if len(current_batch) >= 5000:
                    self._save_json(
                        current_batch,
                        output_dir / f"markets_{file_num}.json"
                    )
                    print(f"  Saved markets_{file_num}.json ({len(current_batch):,} markets)")
                    current_batch = []
                    file_num += 1

                if len(markets) < batch_size:
                    break

            except requests.RequestException as e:
                print(f"  Error fetching Polymarket markets: {e}")
                break

        # Save remaining
        if current_batch:
            self._save_json(
                current_batch,
                output_dir / f"markets_{file_num}.json"
            )
            print(f"  Saved markets_{file_num}.json ({len(current_batch):,} markets)")

        print(f"  Total: {total_fetched:,} markets")
        return total_fetched

    # =========================================================================
    # CONVENIENCE METHODS
    # =========================================================================

    def fetch_all(
        self,
        manifold_bets: int = 5000000,
        manifold_markets: int = 200000,
        polymarket_markets: int = 200000,
    ) -> dict:
        """
        Fetch all data from all sources.

        Args:
            manifold_bets: Max Manifold bets to fetch
            manifold_markets: Max Manifold markets to fetch
            polymarket_markets: Max Polymarket markets to fetch

        Returns:
            Dictionary with counts of fetched data
        """
        print("=" * 60)
        print("FETCHING PREDICTION MARKET DATA")
        print("=" * 60)

        results = {}

        results["manifold_bets"] = self.fetch_manifold_bets(manifold_bets)
        results["manifold_markets"] = self.fetch_manifold_markets(manifold_markets)
        results["polymarket_markets"] = self.fetch_polymarket_markets(polymarket_markets)

        print("\n" + "=" * 60)
        print("FETCH COMPLETE")
        print("=" * 60)
        print(f"  Manifold bets: {results['manifold_bets']:,}")
        print(f"  Manifold markets: {results['manifold_markets']:,}")
        print(f"  Polymarket markets: {results['polymarket_markets']:,}")

        return results

    def data_exists(self) -> dict:
        """Check what data already exists locally."""
        manifold_dir = self.data_dir / "manifold"
        polymarket_dir = self.data_dir / "polymarket"

        return {
            "manifold_bets": list(manifold_dir.glob("bets_*.json")) if manifold_dir.exists() else [],
            "manifold_markets": list(manifold_dir.glob("markets_*.json")) if manifold_dir.exists() else [],
            "polymarket_markets": list(polymarket_dir.glob("markets_*.json")) if polymarket_dir.exists() else [],
        }


def ensure_data_exists(data_dir: str = "data", min_bets: int = 10000) -> bool:
    """
    Ensure data exists, fetching if necessary.

    Args:
        data_dir: Data directory
        min_bets: Minimum bets required before fetching

    Returns:
        True if data is available
    """
    fetcher = DataFetcher(data_dir)
    existing = fetcher.data_exists()

    if not existing["manifold_bets"]:
        print("No Manifold bets found. Fetching data...")
        fetcher.fetch_all()
        return True

    # Count existing bets
    total_bets = 0
    for f in existing["manifold_bets"]:
        try:
            with open(f, encoding="utf-8") as file:
                total_bets += len(json.load(file))
        except:
            pass

    if total_bets < min_bets:
        print(f"Only {total_bets:,} bets found (need {min_bets:,}). Fetching more...")
        fetcher.fetch_all()

    return True


if __name__ == "__main__":
    # Quick test
    fetcher = DataFetcher()
    existing = fetcher.data_exists()
    print("Existing data:")
    for key, files in existing.items():
        print(f"  {key}: {len(files)} files")

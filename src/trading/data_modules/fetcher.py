"""
Data fetching from prediction market APIs.

Supports:
- Manifold Markets (free, no auth required)
- Polymarket Gamma API (market metadata, no auth required)
- Polymarket CLOB API (historical price data, no auth required)
"""

import json
import time
import requests
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Set
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


# API endpoints
MANIFOLD_API = "https://api.manifold.markets/v0"
POLYMARKET_API = "https://gamma-api.polymarket.com"
POLYMARKET_CLOB_API = "https://clob.polymarket.com"
POLYMARKET_ORDERBOOK_API = (
    "https://api.goldsky.com/api/public/project_cl6mb8i9h0003e201j6li0diw/"
    "subgraphs/orderbook-subgraph/0.0.1/gn"
)

# CLOB API limits - max ~7 days per request
CLOB_MAX_RANGE_SECONDS = 7 * 24 * 60 * 60

# Concurrency settings
MAX_WORKERS = 20  # Concurrent requests
CLOB_REQUEST_DELAY = 0.05  # Delay between requests per thread

# Rate limiting
REQUEST_DELAY = 0.1  # seconds between requests

# Default order-filled fields from the Goldsky orderbook subgraph
DEFAULT_ORDER_FILLED_FIELDS = [
    "id",
    "timestamp",
    "maker",
    "makerAssetId",
    "makerAmountFilled",
    "taker",
    "takerAssetId",
    "takerAmountFilled",
    "transactionHash",
    "orderHash",
    "fee",
]


class DataFetcher:
    """Fetches and caches prediction market data."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "PredictionMarketResearch/1.0"
        })
        self._local = threading.local()

    def _get_session(self) -> requests.Session:
        """Get thread-local session for concurrent requests."""
        if not hasattr(self._local, "session"):
            self._local.session = requests.Session()
            self._local.session.headers.update({
                "User-Agent": "PredictionMarketResearch/1.0"
            })
        return self._local.session

    def _rate_limit(self):
        time.sleep(REQUEST_DELAY)

    def _save_json(self, data: list, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def _require_pyarrow(self):
        try:
            import pyarrow as pa  # noqa: F401
            import pyarrow.parquet as pq  # noqa: F401
        except Exception as exc:
            raise RuntimeError("pyarrow is required for parquet output") from exc

    def _load_state(
        self,
        state_path: Path,
        defaults: Dict[str, Any],
        reset: bool = False,
    ) -> Dict[str, Any]:
        if reset and state_path.exists():
            state_path.unlink()
        if state_path.exists():
            with state_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        return defaults

    def _save_state(self, state_path: Path, state: Dict[str, Any]) -> None:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = state_path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(state, f)
        tmp.replace(state_path)

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
        output_format: str = "json",
        parquet_dir: Optional[str] = None,
    ) -> int:
        """
        Fetch markets from Polymarket Gamma API.

        Args:
            limit: Maximum markets to fetch
            batch_size: Markets per request (max 100)
            output_format: "json" or "parquet"
            parquet_dir: Output dir for parquet (default: data/parquet/markets)

        Returns:
            Total markets fetched
        """
        output_format = (output_format or "json").strip().lower()
        if output_format not in {"json", "parquet"}:
            raise ValueError(f"Unsupported output_format: {output_format}")

        output_dir = self.data_dir / "polymarket"
        output_dir.mkdir(parents=True, exist_ok=True)

        if output_format == "parquet":
            self._require_pyarrow()
            import pyarrow as pa
            import pyarrow.parquet as pq

            parquet_output_dir = Path(parquet_dir) if parquet_dir else Path("data/parquet/markets")
            parquet_output_dir.mkdir(parents=True, exist_ok=True)

            def _save_parquet_batch(rows: List[Dict[str, Any]], file_num: int) -> None:
                if not rows:
                    return
                normalized = [self._normalize_market_row(r) for r in rows]
                table = pa.Table.from_pylist(normalized)
                pq.write_table(
                    table,
                    parquet_output_dir / f"markets_{file_num}.parquet",
                    compression="zstd",
                    compression_level=3,
                    row_group_size=500_000,  # Optimize row groups for predicate pushdown
                )

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
                    if output_format == "json":
                        self._save_json(
                            current_batch,
                            output_dir / f"markets_{file_num}.json"
                        )
                        print(f"  Saved markets_{file_num}.json ({len(current_batch):,} markets)")
                    else:
                        _save_parquet_batch(current_batch, file_num)
                        print(f"  Saved markets_{file_num}.parquet ({len(current_batch):,} markets)")
                    current_batch = []
                    file_num += 1

                if len(markets) < batch_size:
                    break

            except requests.RequestException as e:
                print(f"  Error fetching Polymarket markets: {e}")
                break

        # Save remaining
        if current_batch:
            if output_format == "json":
                self._save_json(
                    current_batch,
                    output_dir / f"markets_{file_num}.json"
                )
                print(f"  Saved markets_{file_num}.json ({len(current_batch):,} markets)")
            else:
                _save_parquet_batch(current_batch, file_num)
                print(f"  Saved markets_{file_num}.parquet ({len(current_batch):,} markets)")

        print(f"  Total: {total_fetched:,} markets")
        return total_fetched

    def _normalize_market_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in row.items():
            if isinstance(v, (list, dict)):
                out[k] = json.dumps(v)
            else:
                out[k] = v
        return out

    def _parse_token_list(self, tokens_val: Any) -> List[str]:
        if isinstance(tokens_val, str):
            try:
                tokens_val = json.loads(tokens_val)
            except json.JSONDecodeError:
                return []
        if isinstance(tokens_val, (list, tuple)):
            return [str(t) for t in tokens_val if t is not None]
        return []

    def _load_polymarket_token_map(self, markets_dir: Path) -> Dict[str, Tuple[str, str]]:
        """
        Build token_id -> (market_id, side) map from markets parquet files.
        """
        token_map: Dict[str, Tuple[str, str]] = {}
        if not markets_dir.exists():
            return token_map

        files = sorted(markets_dir.glob("*.parquet"))
        if not files:
            return token_map

        try:
            import pandas as pd
        except Exception as exc:
            raise RuntimeError("pandas is required to load parquet markets") from exc

        for f in files:
            try:
                df = pd.read_parquet(f)
            except Exception:
                continue

            cols = set(df.columns)
            token_col = None
            for c in ("clobTokenIds", "clob_token_ids"):
                if c in cols:
                    token_col = c
                    break
            if token_col is None:
                continue

            if "id" in cols:
                id_col = "id"
            elif "gamma_id" in cols:
                id_col = "gamma_id"
            else:
                continue

            for _, row in df[[id_col, token_col]].iterrows():
                market_id = str(row.get(id_col))
                tokens = self._parse_token_list(row.get(token_col))
                if len(tokens) < 2 or not market_id:
                    continue
                token_map[tokens[0]] = (market_id, "token1")
                token_map[tokens[1]] = (market_id, "token2")

        return token_map

    def _fetch_market_by_token(self, token_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a single market by token_id (Gamma API)."""
        try:
            response = self.session.get(
                f"{POLYMARKET_API}/markets",
                params={"clob_token_ids": token_id},
                timeout=30,
            )
            response.raise_for_status()
            markets = response.json()
            if markets:
                return markets[0]
        except requests.RequestException:
            return None
        return None

    def _append_markets_parquet(self, markets: List[Dict[str, Any]], markets_dir: Path) -> int:
        if not markets:
            return 0
        self._require_pyarrow()
        import pyarrow as pa
        import pyarrow.parquet as pq

        markets_dir.mkdir(parents=True, exist_ok=True)
        normalized = [self._normalize_market_row(m) for m in markets]
        table = pa.Table.from_pylist(normalized)

        # Align schema with existing markets parquet if present.
        existing = sorted(markets_dir.glob("*.parquet"))
        if existing:
            schema = pq.read_schema(existing[0])
            if table.schema != schema:
                table = table.cast(schema)

        filename = markets_dir / f"markets_missing_{int(time.time())}.parquet"
        pq.write_table(
            table,
            filename,
            compression="zstd",
            compression_level=3,
            row_group_size=500_000,  # Optimize row groups for predicate pushdown
        )
        return len(markets)

    def fetch_polymarket_price_history(
        self,
        token_id: str,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
        max_days: int = 365,
    ) -> List[Dict[str, Any]]:
        """
        Fetch historical price data for a Polymarket token from CLOB API.

        Args:
            token_id: The CLOB token ID
            start_ts: Start timestamp (default: max_days ago)
            end_ts: End timestamp (default: now)
            max_days: Maximum days of history to fetch

        Returns:
            List of {t: timestamp, p: price} entries
        """
        session = self._get_session()
        now = int(time.time())
        if end_ts is None:
            end_ts = now
        if start_ts is None:
            start_ts = now - (max_days * 24 * 60 * 60)

        all_history = []
        current_start = start_ts

        while current_start < end_ts:
            current_end = min(current_start + CLOB_MAX_RANGE_SECONDS, end_ts)
            time.sleep(CLOB_REQUEST_DELAY)  # Lighter rate limit for CLOB

            try:
                response = session.get(
                    f"{POLYMARKET_CLOB_API}/prices-history",
                    params={
                        "market": token_id,
                        "startTs": current_start,
                        "endTs": current_end,
                    },
                    timeout=30,
                )

                if response.status_code == 200:
                    data = response.json()
                    history = data.get("history", [])
                    all_history.extend(history)
                elif response.status_code == 429:
                    # Rate limited, wait and retry
                    time.sleep(1)
                    continue

            except requests.RequestException:
                pass

            current_start = current_end

        return all_history

    def _fetch_single_market_history(
        self,
        market: Dict[str, Any],
        output_dir: Path,
        max_days: int,
        progress: Dict[str, int],
        lock: threading.Lock,
    ) -> Tuple[bool, int]:
        """Fetch price history for a single market (thread-safe)."""
        try:
            market_data = {
                "market_info": market,
                "price_history": {},
            }

            total_points = 0

            # Fetch history for each token (YES/NO outcomes)
            for j, token_id in enumerate(market["token_ids"][:2]):
                history = self.fetch_polymarket_price_history(
                    token_id,
                    max_days=max_days,
                )
                outcome = "YES" if j == 0 else "NO"
                market_data["price_history"][outcome] = {
                    "token_id": token_id,
                    "history": history,
                    "data_points": len(history),
                }
                total_points += len(history)

            # Save individual market file
            safe_slug = market["slug"][:50] if market["slug"] else str(market["id"])
            safe_slug = "".join(c if c.isalnum() or c == "-" else "_" for c in safe_slug)
            self._save_json(
                market_data,
                output_dir / f"market_{safe_slug}.json"
            )

            with lock:
                progress["fetched"] += 1
                progress["total_points"] += total_points
                if progress["fetched"] % 20 == 0:
                    print(f"  Progress: {progress['fetched']}/{progress['total']} markets, {progress['total_points']:,} data points")

            return True, total_points

        except Exception as e:
            with lock:
                progress["errors"] += 1
            return False, 0

    def fetch_polymarket_clob_data(
        self,
        min_volume: float = 10000,  # Default: $10k minimum volume
        max_markets: Optional[int] = None,
        max_days: int = 1180,  # Default: ~3.2 years (matches existing trades data: Nov 21, 2022 - Dec 30, 2025)
        workers: int = MAX_WORKERS,
        include_closed: bool = True,  # Default: include closed markets for maximum historical data
    ) -> int:
        """
        Fetch historical price data for Polymarket markets (concurrent).

        Args:
            min_volume: Minimum 24hr volume to include market (default: 10000 = $10k)
            max_markets: Maximum number of markets to fetch (default: None = unlimited)
            max_days: Days of history per market (default: 1180 = ~3.2 years, matches existing trades data: Nov 21, 2022 - Dec 30, 2025)
            workers: Number of concurrent workers (default: 20)
            include_closed: Include closed markets in the scan (default: True for maximum historical data)

        Returns:
            Total markets fetched
        """
        output_dir = self.data_dir / "polymarket_clob"
        output_dir.mkdir(parents=True, exist_ok=True)

        # First, get list of markets with volume
        print("Fetching Polymarket markets...")
        markets_by_id: Dict[str, Dict[str, Any]] = {}
        # Use smaller batch size for closed markets to avoid memory issues
        batch_size = 50 if include_closed else 100
        closed_flags = ["false", "true"] if include_closed else ["false"]

        for closed_flag in closed_flags:
            offset = 0
            consecutive_empty_batches = 0
            max_consecutive_empty = 3  # Stop after 3 empty batches in a row
            # Fetch enough markets: if max_markets specified, fetch 2x to filter; otherwise fetch all
            target_count = (max_markets * 2) if max_markets else float('inf')
            while len(markets_by_id) < target_count:
                self._rate_limit()
                try:
                    response = self.session.get(
                        f"{POLYMARKET_API}/markets",
                        params={
                            "limit": batch_size,
                            "offset": offset,
                            "closed": closed_flag,
                        },
                        timeout=60,  # Increase timeout for large responses
                    )
                    response.raise_for_status()
                    
                    # Check response size before parsing (warn if > 50MB)
                    content_length = response.headers.get('content-length')
                    if content_length:
                        size_mb = int(content_length) / (1024 * 1024)
                        if size_mb > 50:
                            print(f"  Warning: Large response ({size_mb:.1f}MB) for closed={closed_flag}, offset={offset}")
                    
                    # Try to parse JSON with memory error handling
                    try:
                        batch = response.json()
                    except MemoryError:
                        print(f"  MemoryError parsing response (closed={closed_flag}, offset={offset})")
                        print(f"  Response size: {len(response.content) / (1024*1024):.1f}MB")
                        print(f"  Reducing batch size and retrying...")
                        # Reduce batch size and retry
                        batch_size = max(10, batch_size // 2)
                        print(f"  New batch size: {batch_size}")
                        continue
                    except json.JSONDecodeError as e:
                        print(f"  JSON decode error (closed={closed_flag}, offset={offset}): {e}")
                        print(f"  Response preview: {response.text[:500]}")
                        break

                    if not batch:
                        consecutive_empty_batches += 1
                        if consecutive_empty_batches >= max_consecutive_empty:
                            print(f"  Stopping after {max_consecutive_empty} consecutive empty batches")
                            break
                        offset += batch_size  # Still increment offset to try next batch
                        continue
                    else:
                        consecutive_empty_batches = 0  # Reset counter on successful batch

                    for m in batch:
                        vol = float(m.get("volume24hr", 0) or 0)
                        tokens = m.get("clobTokenIds", "[]")
                        if isinstance(tokens, str):
                            try:
                                tokens = json.loads(tokens)
                            except json.JSONDecodeError:
                                tokens = []

                        if vol >= min_volume and tokens:
                            market_id = str(m.get("id"))
                            markets_by_id[market_id] = {
                                "id": m.get("id"),
                                "question": m.get("question", "")[:100],
                                "slug": m.get("slug"),
                                "token_ids": tokens,
                                "volume24hr": vol,
                                "volume": float(m.get("volume", 0) or 0),
                                "liquidity": float(m.get("liquidity", 0) or 0),
                                "outcomes": m.get("outcomes"),
                                "outcomePrices": m.get("outcomePrices"),
                                "endDate": m.get("endDate"),
                            }

                    offset += len(batch)
                    print(
                        f"  Scanned {offset:,} markets (closed={closed_flag}), "
                        f"found {len(markets_by_id):,} with volume >= ${min_volume:,.0f}"
                    )

                    # Stop if we got fewer markets than requested (end of results)
                    if len(batch) < batch_size:
                        print(f"  Reached end of results (got {len(batch)} < {batch_size} markets)")
                        break

                except requests.RequestException as e:
                    print(f"  Error fetching markets (closed={closed_flag}): {e}")
                    break
                except MemoryError as e:
                    print(f"  MemoryError fetching markets (closed={closed_flag}, offset={offset}): {e}")
                    print(f"  Reducing batch size and retrying...")
                    batch_size = max(10, batch_size // 2)
                    print(f"  New batch size: {batch_size}")
                    continue

        markets = list(markets_by_id.values())
        label = "active + closed" if include_closed else "active"
        print(f"  Found {len(markets)} {label} markets with volume >= ${min_volume:,.0f}")

        # Warn if we got exactly 1000 markets (suspiciously round number)
        if len(markets) == 1000 and max_markets is None:
            print(f"  ⚠️  WARNING: Found exactly 1000 markets. This may indicate a pagination limit.")
            print(f"     If you expected more markets, the API may have a limit or pagination issue.")

        if not markets:
            return 0

        # Sort by volume and limit (if max_markets specified)
        markets.sort(key=lambda x: x["volume24hr"], reverse=True)
        if max_markets is not None:
            markets = markets[:max_markets]
            print(f"  Limited to top {len(markets):,} markets by volume")

        # Concurrent fetch
        print(f"Fetching price history for {len(markets)} markets with {workers} workers...")

        progress = {"fetched": 0, "errors": 0, "total_points": 0, "total": len(markets)}
        lock = threading.Lock()

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    self._fetch_single_market_history,
                    market,
                    output_dir,
                    max_days,
                    progress,
                    lock,
                ): market
                for market in markets
            }

            for future in as_completed(futures):
                # Results are tracked via progress dict
                pass

        # Save market index
        index = [{
            "id": m["id"],
            "slug": m["slug"],
            "question": m["question"],
            "volume24hr": m["volume24hr"],
        } for m in markets]
        self._save_json(index, output_dir / "market_index.json")

        print(f"\n  Complete: {progress['fetched']} markets, {progress['total_points']:,} data points")
        if progress["errors"] > 0:
            print(f"  Errors: {progress['errors']}")

        return progress["fetched"]

    def fetch_polymarket_order_filled_events(
        self,
        output_dir: str = "data/parquet/order_filled_events",
        state_path: Optional[str] = None,
        endpoint: str = POLYMARKET_ORDERBOOK_API,
        entity: str = "orderFilledEvents",
        fields: Optional[List[str]] = None,
        batch_size: int = 1000,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
        sleep: float = 0.2,
        max_batches: int = 0,
        reset: bool = False,
    ) -> Dict[str, Any]:
        """
        Backfill Polymarket order-filled events from present to past (resume-safe).

        Writes parquet part files into output_dir and stores state for resume.
        """
        self._require_pyarrow()
        import pyarrow as pa
        import pyarrow.parquet as pq

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        state_file = Path(state_path) if state_path else out_dir / "_state.json"

        if fields is None:
            fields = list(DEFAULT_ORDER_FILLED_FIELDS)
        if "id" not in fields:
            fields = ["id"] + fields
        if "timestamp" not in fields:
            fields = ["timestamp"] + fields

        start_ts = start_ts or int(time.time())
        end_ts = end_ts or None

        existing_state = state_file.exists() and not reset
        state = self._load_state(
            state_file,
            defaults={
                "entity": entity,
                "fields": fields,
                "cursor_timestamp": start_ts,
                "sticky_timestamp": None,
                "last_id": None,
                "part_seq": 0,
            },
            reset=reset,
        )

        if existing_state:
            if state.get("entity") != entity or state.get("fields") != fields:
                raise RuntimeError(
                    "State entity/fields mismatch. Use reset=True or a new output dir."
                )

        cursor_ts = int(state.get("cursor_timestamp") or start_ts)
        sticky_ts = state.get("sticky_timestamp")
        last_id = state.get("last_id")
        part_seq = int(state.get("part_seq") or 0)

        session = self.session
        batches = 0
        total_rows = 0

        def _build_query(where_clause: str) -> str:
            fields_str = "\n            ".join(fields)
            return f"""
                query BackfillQuery {{
                    {entity}(
                        orderBy: timestamp,
                        orderDirection: desc,
                        first: {batch_size},
                        where: {{{where_clause}}}
                    ) {{
                        {fields_str}
                    }}
                }}
            """.strip()

        while True:
            if end_ts is not None and cursor_ts <= end_ts:
                break

            if sticky_ts is not None and last_id:
                where_clause = f'timestamp: "{sticky_ts}", id_lt: "{last_id}"'
            else:
                where_clause = f'timestamp_lt: "{cursor_ts}"'

            query = _build_query(where_clause)
            payload = {"query": query}

            rows: List[Dict[str, Any]] = []
            for _ in range(5):
                try:
                    resp = session.post(endpoint, json=payload, timeout=60)
                    if resp.status_code == 429:
                        time.sleep(max(sleep, 0.5))
                        continue
                    resp.raise_for_status()
                    data = resp.json()
                    if "errors" in data:
                        raise RuntimeError(f"GraphQL error: {data['errors']}")
                    rows = data.get("data", {}).get(entity, []) or []
                    break
                except requests.RequestException:
                    time.sleep(max(sleep, 0.5))
                    continue

            if not rows:
                if sticky_ts is not None:
                    cursor_ts = int(sticky_ts)
                    sticky_ts = None
                    last_id = None
                    continue
                break

            # Deduplicate by id within batch
            seen_ids: Set[str] = set()
            deduped = []
            for row in rows:
                row_id = str(row.get("id"))
                if not row_id or row_id in seen_ids:
                    continue
                seen_ids.add(row_id)
                deduped.append(row)
            rows = deduped

            # Sort descending by timestamp, id
            def _row_key(r: Dict[str, Any]) -> Tuple[int, str]:
                try:
                    ts_val = int(r.get("timestamp", 0))
                except (TypeError, ValueError):
                    ts_val = 0
                return (ts_val, str(r.get("id") or ""))

            rows.sort(key=_row_key, reverse=True)

            batch_first_ts = int(rows[0]["timestamp"])
            batch_last_ts = int(rows[-1]["timestamp"])
            batch_last_id = str(rows[-1].get("id"))

            # Bucket rows by month and write parquet
            buckets: Dict[str, Dict[str, List[Any]]] = {}
            for row in rows:
                try:
                    ts_val = int(row.get("timestamp"))
                except (TypeError, ValueError):
                    continue
                if end_ts is not None and ts_val < end_ts:
                    continue
                month = datetime.utcfromtimestamp(ts_val).strftime("%Y-%m")
                if month not in buckets:
                    buckets[month] = {f: [] for f in fields}
                for f in fields:
                    val = row.get(f)
                    if f == "timestamp":
                        val = ts_val
                    buckets[month][f].append(val)

            for month, cols in buckets.items():
                if not cols or not cols[fields[0]]:
                    continue
                table = pa.table(cols)
                filepath = out_dir / f"order_filled_{month}_part{part_seq:05d}.parquet"
                pq.write_table(
                    table,
                    filepath,
                    compression="zstd",
                    compression_level=3,
                    row_group_size=500_000,  # Optimize row groups for predicate pushdown
                )
                part_seq += 1
                total_rows += table.num_rows

            # Update cursor
            if len(rows) >= batch_size:
                sticky_ts = batch_last_ts
                last_id = batch_last_id
            else:
                cursor_ts = batch_last_ts
                sticky_ts = None
                last_id = None

            state.update(
                {
                    "cursor_timestamp": cursor_ts,
                    "sticky_timestamp": sticky_ts,
                    "last_id": last_id,
                    "part_seq": part_seq,
                }
            )
            self._save_state(state_file, state)

            batches += 1
            if max_batches and batches >= max_batches:
                break
            if sleep > 0:
                time.sleep(sleep)

        return {
            "rows_written": total_rows,
            "batches": batches,
            "output_dir": str(out_dir),
            "state_file": str(state_file),
        }

    def process_polymarket_trades_from_order_filled(
        self,
        order_filled_dir: str = "data/parquet/order_filled_events",
        output_dir: str = "data/parquet/trades",
        markets_dir: str = "data/parquet/markets",
        state_path: Optional[str] = None,
        reset: bool = False,
        max_files: int = 0,
        fetch_missing_tokens: bool = False,
    ) -> Dict[str, Any]:
        """
        Convert order-filled events parquet -> processed trades parquet (resume-safe).
        """
        self._require_pyarrow()
        import pyarrow as pa
        import pyarrow.parquet as pq
        import pandas as pd
        import numpy as np

        in_dir = Path(order_filled_dir)
        out_dir = Path(output_dir)
        markets_path = Path(markets_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        state_file = Path(state_path) if state_path else out_dir / "_order_filled_state.json"

        state = self._load_state(
            state_file,
            defaults={
                "last_input_file": None,
                "part_seq": 0,
            },
            reset=reset,
        )

        token_map = self._load_polymarket_token_map(markets_path)
        if not token_map:
            raise RuntimeError(
                f"No token map found. Ensure markets parquet exists in {markets_path}."
            )

        files = sorted(in_dir.glob("*.parquet"))
        if not files:
            return {"rows_written": 0, "files": 0}

        last_file = state.get("last_input_file")
        part_seq = int(state.get("part_seq") or 0)
        processed_files = 0
        total_rows = 0
        skipped_missing = 0

        def _apply_mapping(df: "pd.DataFrame") -> "pd.DataFrame":
            nonusdc = df["makerAssetId"].where(df["makerAssetId"] != "0", df["takerAssetId"])
            df["nonusdc_asset_id"] = nonusdc
            df["market_id"] = df["nonusdc_asset_id"].map(lambda x: token_map.get(x, (None, None))[0])
            df["nonusdc_side"] = df["nonusdc_asset_id"].map(lambda x: token_map.get(x, (None, None))[1])
            return df

        for f in files:
            if last_file and f.name <= last_file:
                continue
            df = pd.read_parquet(f)
            if df.empty:
                state["last_input_file"] = f.name
                self._save_state(state_file, state)
                continue

            required = [
                "timestamp",
                "maker",
                "makerAssetId",
                "makerAmountFilled",
                "taker",
                "takerAssetId",
                "takerAmountFilled",
                "transactionHash",
            ]
            missing_cols = [c for c in required if c not in df.columns]
            if missing_cols:
                raise RuntimeError(f"Missing columns in order-filled data: {missing_cols}")

            df["makerAssetId"] = df["makerAssetId"].astype(str)
            df["takerAssetId"] = df["takerAssetId"].astype(str)
            df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
            df = df.dropna(subset=["timestamp"])

            if "id" in df.columns:
                df = df.drop_duplicates(subset=["id"])

            df = _apply_mapping(df)
            missing_tokens = df[df["market_id"].isna()]["nonusdc_asset_id"].dropna().unique().tolist()
            if missing_tokens and fetch_missing_tokens:
                fetched = []
                for token_id in missing_tokens:
                    market = self._fetch_market_by_token(str(token_id))
                    if market:
                        fetched.append(market)
                        tokens = self._parse_token_list(market.get("clobTokenIds"))
                        if len(tokens) >= 2:
                            token_map[tokens[0]] = (str(market.get("id")), "token1")
                            token_map[tokens[1]] = (str(market.get("id")), "token2")
                if fetched:
                    self._append_markets_parquet(fetched, markets_path)
                    df = _apply_mapping(df)

            before = len(df)
            df = df.dropna(subset=["market_id", "nonusdc_side"])
            skipped_missing += (before - len(df))
            if df.empty:
                state["last_input_file"] = f.name
                self._save_state(state_file, state)
                continue

            df["makerAmountFilled"] = pd.to_numeric(df["makerAmountFilled"], errors="coerce") / 1e6
            df["takerAmountFilled"] = pd.to_numeric(df["takerAmountFilled"], errors="coerce") / 1e6

            df["makerAsset"] = np.where(df["makerAssetId"] == "0", "USDC", df["nonusdc_side"])
            df["takerAsset"] = np.where(df["takerAssetId"] == "0", "USDC", df["nonusdc_side"])

            df["taker_direction"] = np.where(df["takerAsset"] == "USDC", "BUY", "SELL")
            df["maker_direction"] = np.where(df["takerAsset"] == "USDC", "SELL", "BUY")

            df["usd_amount"] = np.where(
                df["takerAsset"] == "USDC",
                df["takerAmountFilled"],
                df["makerAmountFilled"],
            )
            df["token_amount"] = np.where(
                df["takerAsset"] != "USDC",
                df["takerAmountFilled"],
                df["makerAmountFilled"],
            )

            df = df.dropna(subset=["usd_amount", "token_amount"])
            df = df[df["token_amount"] > 0]
            df["price"] = df["usd_amount"] / df["token_amount"]

            dt = pd.to_datetime(df["timestamp"], unit="s", utc=True)
            df["timestamp"] = dt.dt.strftime("%Y-%m-%d %H:%M:%S")
            df["month"] = dt.dt.strftime("%Y-%m")

            out_cols = [
                "timestamp",
                "market_id",
                "maker",
                "taker",
                "nonusdc_side",
                "maker_direction",
                "taker_direction",
                "price",
                "usd_amount",
                "token_amount",
                "transactionHash",
            ]
            df = df[out_cols + ["month"]]

            for month, group in df.groupby("month"):
                out_df = group.drop(columns=["month"])
                table = pa.Table.from_pandas(out_df, preserve_index=False)
                filepath = out_dir / f"trades_{month}_part{part_seq:05d}.parquet"
                pq.write_table(
                    table,
                    filepath,
                    compression="zstd",
                    compression_level=3,
                    row_group_size=500_000,  # Optimize row groups for predicate pushdown
                )
                part_seq += 1
                total_rows += table.num_rows

            processed_files += 1
            state.update({"last_input_file": f.name, "part_seq": part_seq})
            self._save_state(state_file, state)

            if max_files and processed_files >= max_files:
                break

        return {
            "rows_written": total_rows,
            "files": processed_files,
            "skipped_missing": skipped_missing,
            "output_dir": str(out_dir),
            "state_file": str(state_file),
        }

    # =========================================================================
    # CONVENIENCE METHODS
    # =========================================================================

    def fetch_all(
        self,
        manifold_bets: int = 5000000,
        manifold_markets: int = 200000,
        polymarket_markets: int = 200000,
        polymarket_clob: bool = False,
        clob_min_volume: float = 10000,
        clob_max_markets: int = 500,
    ) -> dict:
        """
        Fetch all data from all sources.

        Args:
            manifold_bets: Max Manifold bets to fetch
            manifold_markets: Max Manifold markets to fetch
            polymarket_markets: Max Polymarket markets to fetch
            polymarket_clob: Whether to fetch CLOB price history
            clob_min_volume: Min 24hr volume for CLOB markets
            clob_max_markets: Max markets for CLOB history

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

        if polymarket_clob:
            results["polymarket_clob"] = self.fetch_polymarket_clob_data(
                min_volume=clob_min_volume,
                max_markets=clob_max_markets,
            )

        print("\n" + "=" * 60)
        print("FETCH COMPLETE")
        print("=" * 60)
        print(f"  Manifold bets: {results['manifold_bets']:,}")
        print(f"  Manifold markets: {results['manifold_markets']:,}")
        print(f"  Polymarket markets: {results['polymarket_markets']:,}")
        if polymarket_clob:
            print(f"  Polymarket CLOB: {results.get('polymarket_clob', 0):,} markets")

        return results

    def data_exists(self) -> dict:
        """Check what data already exists locally."""
        manifold_dir = self.data_dir / "manifold"
        polymarket_dir = self.data_dir / "polymarket"
        polymarket_clob_dir = self.data_dir / "polymarket_clob"
        order_filled_dir = Path("data/parquet/order_filled_events")
        trades_parquet_dir = Path("data/parquet/trades")

        return {
            "manifold_bets": list(manifold_dir.glob("bets_*.json")) if manifold_dir.exists() else [],
            "manifold_markets": list(manifold_dir.glob("markets_*.json")) if manifold_dir.exists() else [],
            "polymarket_markets": list(polymarket_dir.glob("markets_*.json")) if polymarket_dir.exists() else [],
            "polymarket_clob": list(polymarket_clob_dir.glob("market_*.json")) if polymarket_clob_dir.exists() else [],
            "polymarket_order_filled": list(order_filled_dir.glob("*.parquet")) if order_filled_dir.exists() else [],
            "polymarket_trades_parquet": list(trades_parquet_dir.glob("trades_*.parquet")) if trades_parquet_dir.exists() else [],
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
    import sys

    fetcher = DataFetcher()

    if len(sys.argv) > 1 and sys.argv[1] == "--clob":
        # Fetch CLOB price history
        print("Fetching Polymarket CLOB price history...")
        min_vol = float(sys.argv[2]) if len(sys.argv) > 2 else 10000
        max_mkts = int(sys.argv[3]) if len(sys.argv) > 3 else 100
        max_days = int(sys.argv[4]) if len(sys.argv) > 4 else 90
        include_closed = False
        if len(sys.argv) > 5:
            include_closed = str(sys.argv[5]).strip().lower() in {"1", "true", "yes", "y", "on"}
        fetcher.fetch_polymarket_clob_data(
            min_volume=min_vol,
            max_markets=max_mkts,
            max_days=max_days,
            include_closed=include_closed,
        )
    else:
        # Show existing data
        existing = fetcher.data_exists()
        print("Existing data:")
        for key, files in existing.items():
            print(f"  {key}: {len(files)} files")
        print(
            "\nTo fetch CLOB data: python -m src.trading.data_modules.fetcher "
            "--clob [min_volume] [max_markets] [max_days] [include_closed]"
        )

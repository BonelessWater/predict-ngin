#!/usr/bin/env python3
"""
Fetch Polymarket CLOB data in shards for parallel execution.

Examples:
  # 8 shards in 8 terminals
  python scripts/fetch_clob_shard.py 0 8 --market-slugs data/market_slugs.json --max-days 3650 --workers 80
  python scripts/fetch_clob_shard.py 1 8 --market-slugs data/market_slugs.json --max-days 3650 --workers 80
  ...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Iterable, List, Optional

import requests

try:
    import polars as pl
except ImportError:  # pragma: no cover - optional dependency
    pl = None

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - optional dependency
    pa = None
    pq = None

POLYMARKET_API = "https://gamma-api.polymarket.com"
POLYMARKET_CLOB_API = "https://clob.polymarket.com"
CLOB_MAX_RANGE_SECONDS = 7 * 24 * 60 * 60

# Reasonable defaults; override via CLI.
DEFAULT_WORKERS = 64
DEFAULT_REQUEST_DELAY = 0.01


def month_from_ts(ts: int) -> str:
    return time.strftime("%Y-%m", time.gmtime(ts))


def _thread_local_session(local: threading.local) -> requests.Session:
    if not hasattr(local, "session"):
        local.session = requests.Session()
        local.session.headers.update({"User-Agent": "PredictionMarketResearch/1.0"})
    return local.session


def _parse_csv_list(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [v.strip().lower() for v in value.split(",") if v.strip()]


def _matches_filter(value: Optional[str], allowed: Iterable[str]) -> bool:
    allowed_list = list(allowed)
    if not allowed_list:
        return True
    if value is None:
        return False
    return value.strip().lower() in allowed_list


def fetch_markets_from_gamma(
    min_volume: float,
    max_markets: int,
    include_closed: bool,
    categories: Iterable[str],
    market_types: Iterable[str],
) -> List[Dict[str, Any]]:
    markets: List[Dict[str, Any]] = []
    offset = 0
    batch_size = 100

    closed_flags = ["false", "true"] if include_closed else ["false"]

    for closed_flag in closed_flags:
        while len(markets) < max_markets:
            try:
                resp = requests.get(
                    f"{POLYMARKET_API}/markets",
                    params={"limit": batch_size, "offset": offset, "closed": closed_flag},
                    timeout=30,
                )
                resp.raise_for_status()
                batch = resp.json()
                if not batch:
                    break

                for m in batch:
                    vol = float(m.get("volume24hr", 0) or 0)
                    tokens = m.get("clobTokenIds", "[]")
                    category = m.get("category")
                    market_type = m.get("marketType")
                    if isinstance(tokens, str):
                        try:
                            tokens = json.loads(tokens)
                        except json.JSONDecodeError:
                            tokens = []
                    if (
                        vol >= min_volume
                        and tokens
                        and _matches_filter(category, categories)
                        and _matches_filter(market_type, market_types)
                    ):
                        markets.append(
                            {
                                "id": m.get("id"),
                                "slug": m.get("slug"),
                                "token_ids": tokens,
                                "volume24hr": vol,
                                "question": m.get("question", "")[:80],
                                "category": category,
                                "marketType": market_type,
                            }
                        )

                offset += len(batch)
                if len(batch) < batch_size:
                    break
                time.sleep(0.05)
            except Exception as exc:
                print(f"Error fetching markets (closed={closed_flag}): {exc}")
                break

    markets.sort(key=lambda x: x.get("volume24hr", 0), reverse=True)
    return markets[:max_markets]


def fetch_markets_from_slugs(market_slugs_path: Path) -> List[Dict[str, Any]]:
    with market_slugs_path.open(encoding="utf-8") as f:
        data = json.load(f)

    markets: List[Dict[str, Any]] = []
    for row in data:
        tokens = row.get("clob_token_ids")
        if isinstance(tokens, str):
            try:
                tokens = json.loads(tokens)
            except json.JSONDecodeError:
                tokens = []
        if not tokens or len(tokens) < 2:
            continue
        markets.append(
            {
                "id": row.get("gamma_id"),
                "slug": row.get("slug"),
                "token_ids": tokens,
                "volume24hr": 0.0,
                "question": row.get("question", "")[:80],
            }
        )
    return markets


def fetch_markets_from_parquet(
    parquet_dir: Path,
    min_clob_volume: float,
    max_markets: int,
    categories: Iterable[str],
    market_types: Iterable[str],
) -> List[Dict[str, Any]]:
    if not parquet_dir.exists():
        return []
    files = sorted(parquet_dir.glob("markets_*.parquet"))
    if not files:
        return []

    if pl is None:
        raise RuntimeError("polars is required to load parquet markets.")

    select_cols = [
        "id",
        "slug",
        "question",
        "clobTokenIds",
        "category",
        "marketType",
        "volume24hr",
        "volume1wkClob",
        "volume1moClob",
        "volume1yrClob",
    ]

    def _ensure_columns(df_part: "pl.DataFrame") -> "pl.DataFrame":
        missing = [c for c in select_cols if c not in df_part.columns]
        if missing:
            df_part = df_part.with_columns([pl.lit(None).alias(c) for c in missing])
        return df_part.select(select_cols)

    dfs = []
    for f in files:
        try:
            df_part = pl.read_parquet(str(f))
        except Exception:
            continue
        df_part = _ensure_columns(df_part)
        df_part = df_part.with_columns(
            [
                pl.col("id").cast(pl.Utf8, strict=False),
                pl.col("slug").cast(pl.Utf8, strict=False),
                pl.col("question").cast(pl.Utf8, strict=False),
                pl.col("volume24hr").cast(pl.Float64, strict=False),
                pl.col("volume1wkClob").cast(pl.Float64, strict=False),
                pl.col("volume1moClob").cast(pl.Float64, strict=False),
                pl.col("volume1yrClob").cast(pl.Float64, strict=False),
            ]
        )
        dfs.append(df_part)
    df = pl.concat(dfs, how="vertical") if dfs else pl.DataFrame()
    if df.is_empty():
        return []

    markets: List[Dict[str, Any]] = []
    for row in df.iter_rows(named=True):
        tokens = row.get("clobTokenIds")
        category = row.get("category")
        market_type = row.get("marketType")
        if isinstance(tokens, str):
            try:
                tokens = json.loads(tokens)
            except json.JSONDecodeError:
                tokens = []
        if not tokens or len(tokens) < 2:
            continue
        if not _matches_filter(category, categories) or not _matches_filter(market_type, market_types):
            continue

        def _to_float(val: Any) -> float:
            try:
                return float(val)
            except Exception:
                return 0.0

        clob_vol = max(
            _to_float(row.get("volume1yrClob")),
            _to_float(row.get("volume1moClob")),
            _to_float(row.get("volume1wkClob")),
            _to_float(row.get("volume24hr")),
        )
        if clob_vol < min_clob_volume:
            continue

        markets.append(
            {
                "id": row.get("id"),
                "slug": row.get("slug"),
                "token_ids": tokens,
                "volume24hr": _to_float(row.get("volume24hr")),
                "question": (row.get("question") or "")[:80],
                "_clob_volume": clob_vol,
                "category": category,
                "marketType": market_type,
            }
        )

    markets.sort(key=lambda x: x.get("_clob_volume", 0), reverse=True)
    return markets[:max_markets]


class ParquetPriceWriter:
    def __init__(
        self,
        output_dir: Path,
        chunk_rows: int = 500_000,
        single_file_per_month: bool = False,
    ) -> None:
        if pa is None or pq is None:
            raise RuntimeError("pyarrow is required for parquet output.")
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_rows = chunk_rows
        self.single_file_per_month = single_file_per_month
        self.buffers: Dict[str, Dict[str, List]] = {}
        self.part_counters: Dict[str, int] = {}
        self.writers: Dict[str, pq.ParquetWriter] = {}
        self.lock = threading.Lock()
        self.write_lock = threading.Lock()

    def _ensure_buffer(self, month: str) -> Dict[str, List]:
        if month not in self.buffers:
            self.buffers[month] = {
                "market_id": [],
                "outcome": [],
                "timestamp": [],
                "price": [],
            }
        return self.buffers[month]

    def _drain_month(self, month: str) -> Optional[Dict[str, List]]:
        buf = self.buffers.get(month)
        if not buf or not buf["market_id"]:
            return None
        self.buffers[month] = {
            "market_id": [],
            "outcome": [],
            "timestamp": [],
            "price": [],
        }
        return buf

    def _next_part_path(self, month: str) -> Path:
        part_idx = self.part_counters.get(month, 0)
        self.part_counters[month] = part_idx + 1
        return self.output_dir / f"prices_{month}_part{part_idx:05d}.parquet"

    def _write_table(self, month: str, table: "pa.Table") -> None:
        if not self.single_file_per_month:
            pq.write_table(table, self._next_part_path(month), compression="snappy")
            return
        with self.write_lock:
            writer = self.writers.get(month)
            if writer is None:
                path = self.output_dir / f"prices_{month}.parquet"
                writer = pq.ParquetWriter(path, table.schema, compression="snappy")
                self.writers[month] = writer
            writer.write_table(table)

    def add_history(self, market_id: str, outcome: str, history: List[Dict[str, Any]]) -> int:
        if not history:
            return 0
        local: Dict[str, Dict[str, List]] = {}
        rows = 0
        for point in history:
            ts = point.get("t")
            price = point.get("p")
            if ts is None or price is None:
                continue
            try:
                ts_int = int(ts)
                price_f = float(price)
            except Exception:
                continue
            month = month_from_ts(ts_int)
            buf = local.get(month)
            if buf is None:
                buf = {
                    "market_id": [],
                    "outcome": [],
                    "timestamp": [],
                    "price": [],
                }
                local[month] = buf
            buf["market_id"].append(market_id)
            buf["outcome"].append(outcome)
            buf["timestamp"].append(ts_int)
            buf["price"].append(price_f)
            rows += 1

        flush_targets: List[tuple[str, Dict[str, List]]] = []
        with self.lock:
            for month, buf in local.items():
                shared = self._ensure_buffer(month)
                shared["market_id"].extend(buf["market_id"])
                shared["outcome"].extend(buf["outcome"])
                shared["timestamp"].extend(buf["timestamp"])
                shared["price"].extend(buf["price"])
                if len(shared["market_id"]) >= self.chunk_rows:
                    drained = self._drain_month(month)
                    if drained:
                        flush_targets.append((month, drained))

        for month, drained in flush_targets:
            table = pa.table(drained)
            self._write_table(month, table)

        return rows

    def close(self) -> None:
        flush_targets: List[tuple[str, Dict[str, List]]] = []
        with self.lock:
            for month in list(self.buffers.keys()):
                drained = self._drain_month(month)
                if drained:
                    flush_targets.append((month, drained))

        for month, drained in flush_targets:
            table = pa.table(drained)
            self._write_table(month, table)

        if self.single_file_per_month:
            with self.write_lock:
                for writer in self.writers.values():
                    writer.close()
                self.writers = {}


def fetch_price_history(
    token_id: str,
    max_days: int,
    request_delay: float,
    local: threading.local,
) -> List[Dict[str, Any]]:
    session = _thread_local_session(local)
    now = int(time.time())
    start_ts = now - (max_days * 24 * 60 * 60)

    all_history: List[Dict[str, Any]] = []
    current_start = start_ts

    while current_start < now:
        current_end = min(current_start + CLOB_MAX_RANGE_SECONDS, now)
        if request_delay > 0:
            time.sleep(request_delay)

        try:
            response = session.get(
                f"{POLYMARKET_CLOB_API}/prices-history",
                params={"market": token_id, "startTs": current_start, "endTs": current_end},
                timeout=30,
            )
            if response.status_code == 200:
                data = response.json()
                all_history.extend(data.get("history", []))
                current_start = current_end
            elif response.status_code == 429:
                time.sleep(1)
                continue
            else:
                current_start = current_end
        except Exception:
            current_start = current_end

    return all_history


def fetch_market(
    market: Dict[str, Any],
    output_dir: Path,
    max_days: int,
    request_delay: float,
    skip_existing: bool,
    progress: Dict[str, int],
    lock: threading.Lock,
    local: threading.local,
    parquet_writer: Optional[ParquetPriceWriter],
    meta_rows: Optional[List[Dict[str, Any]]],
    meta_lock: Optional[threading.Lock],
    write_json: bool,
) -> bool:
    try:
        safe_slug = market.get("slug") or str(market.get("id"))
        safe_slug = safe_slug[:50]
        safe_slug = "".join(c if c.isalnum() or c == "-" else "_" for c in safe_slug)
        filepath = output_dir / f"market_{safe_slug}.json"

        if write_json and skip_existing and filepath.exists():
            with lock:
                progress["done"] += 1
            return True

        market_data = {"market_info": market, "price_history": {}}
        total_points = 0
        market_id_str = str(market.get("id", ""))

        for j, token_id in enumerate(market["token_ids"][:2]):
            history = fetch_price_history(
                token_id,
                max_days=max_days,
                request_delay=request_delay,
                local=local,
            )
            outcome = "YES" if j == 0 else "NO"
            market_data["price_history"][outcome] = {
                "token_id": token_id,
                "history": history,
                "data_points": len(history),
            }
            total_points += len(history)
            if parquet_writer is not None:
                parquet_writer.add_history(market_id_str, outcome, history)

        if write_json:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(market_data, f)

        if meta_rows is not None and meta_lock is not None:
            meta_row = {
                "id": market_id_str,
                "slug": market.get("slug"),
                "question": market.get("question"),
                "category": market.get("category"),
                "marketType": market.get("marketType"),
                "volume24hr": market.get("volume24hr"),
                "clobTokenIds": market.get("token_ids"),
            }
            if "_clob_volume" in market:
                meta_row["clob_volume"] = market.get("_clob_volume")
            with meta_lock:
                meta_rows.append(meta_row)

        with lock:
            progress["done"] += 1
            progress["points"] += total_points
            if total_points == 0:
                progress["empty"] += 1
            if progress["done"] % 10 == 0:
                print(
                    f"  Progress: {progress['done']}/{progress['total']} markets, "
                    f"{progress['points']:,} points, empty={progress['empty']:,}"
                )
        return True
    except Exception:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch Polymarket CLOB price history in shards.")
    parser.add_argument("shard_id", type=int, help="Shard index (0-based)")
    parser.add_argument("total_shards", type=int, help="Total number of shards")
    parser.add_argument("--min-volume", type=float, default=0.0, help="Min 24h volume filter (Gamma only)")
    parser.add_argument("--max-markets", type=int, default=1_000_000, help="Max markets to fetch")
    parser.add_argument("--max-days", type=int, default=3650, help="Max history days per market")
    parser.add_argument("--include-closed", action="store_true", help="Include closed markets (Gamma only)")
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Comma-separated categories to include (case-insensitive, Gamma/parquet only)",
    )
    parser.add_argument(
        "--market-type",
        type=str,
        default=None,
        help="Comma-separated market types to include (case-insensitive, Gamma/parquet only)",
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Thread worker count")
    parser.add_argument("--request-delay", type=float, default=DEFAULT_REQUEST_DELAY, help="Delay between requests")
    parser.add_argument(
        "--market-slugs",
        type=str,
        default=None,
        help="Path to market_slugs.json (uses all markets, skips Gamma scan)",
    )
    parser.add_argument(
        "--markets-parquet",
        type=str,
        default=None,
        help="Path to parquet markets dir (filters by CLOB volume if provided)",
    )
    parser.add_argument(
        "--min-clob-volume",
        type=float,
        default=0.0,
        help="Minimum CLOB volume (max of 1yr/1mo/1wk/24h) when using parquet markets",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip markets already fetched (resume-safe)",
    )
    parser.add_argument(
        "--parquet-output-dir",
        type=str,
        default=None,
        help="Write price history to parquet parts in this directory",
    )
    parser.add_argument(
        "--parquet-meta-path",
        type=str,
        default=None,
        help="Write market metadata to a parquet file (default: <parquet-output-dir>/markets_meta.parquet)",
    )
    parser.add_argument(
        "--parquet-chunk-rows",
        type=int,
        default=500_000,
        help="Flush parquet buffers after this many rows per month",
    )
    parser.add_argument(
        "--parquet-one-file-per-month",
        action="store_true",
        help="Write a single parquet file per month (prices_YYYY-MM.parquet)",
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Do not write per-market JSON files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: data/polymarket_clob_shard{shard_id})",
    )
    args = parser.parse_args()

    shard_id = args.shard_id
    total_shards = args.total_shards
    if shard_id < 0 or shard_id >= total_shards:
        print("Invalid shard_id/total_shards.")
        return 1

    print(f"=== SHARD {shard_id + 1}/{total_shards} ===")

    markets: List[Dict[str, Any]]
    market_slugs_path: Optional[Path] = Path(args.market_slugs) if args.market_slugs else None
    markets_parquet_path: Optional[Path] = Path(args.markets_parquet) if args.markets_parquet else None
    categories = _parse_csv_list(args.category)
    market_types = _parse_csv_list(args.market_type)
    if markets_parquet_path and markets_parquet_path.exists():
        print(f"Loading markets from parquet {markets_parquet_path}...")
        markets = fetch_markets_from_parquet(
            markets_parquet_path,
            min_clob_volume=args.min_clob_volume,
            max_markets=args.max_markets,
            categories=categories,
            market_types=market_types,
        )
    elif market_slugs_path and market_slugs_path.exists():
        print(f"Loading markets from {market_slugs_path}...")
        markets = fetch_markets_from_slugs(market_slugs_path)
    else:
        print("Fetching markets from Gamma API...")
        markets = fetch_markets_from_gamma(
            min_volume=args.min_volume,
            max_markets=args.max_markets,
            include_closed=args.include_closed,
            categories=categories,
            market_types=market_types,
        )

    if not markets:
        print("No markets found.")
        return 0

    # Shard markets
    shard_markets = [m for i, m in enumerate(markets) if i % total_shards == shard_id]
    print(f"Total markets: {len(markets)}")
    print(f"This shard: {len(shard_markets)}")

    if not shard_markets:
        print("No markets to fetch in this shard.")
        return 0

    output_dir = Path(args.output_dir) if args.output_dir else Path(f"data/polymarket_clob_shard{shard_id}")
    output_dir.mkdir(parents=True, exist_ok=True)

    progress = {"done": 0, "total": len(shard_markets), "points": 0, "empty": 0}
    lock = threading.Lock()
    local = threading.local()
    parquet_writer: Optional[ParquetPriceWriter] = None
    meta_rows: Optional[List[Dict[str, Any]]] = None
    meta_lock: Optional[threading.Lock] = None

    if args.parquet_output_dir:
        parquet_writer = ParquetPriceWriter(
            Path(args.parquet_output_dir),
            chunk_rows=args.parquet_chunk_rows,
            single_file_per_month=args.parquet_one_file_per_month,
        )
        meta_rows = []
        meta_lock = threading.Lock()
        if args.no_json and args.skip_existing:
            print("Warning: --no-json disables --skip-existing (no per-market files to check).")

    print(f"Fetching with {args.workers} workers...")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                fetch_market,
                m,
                output_dir,
                args.max_days,
                args.request_delay,
                args.skip_existing,
                progress,
                lock,
                local,
                parquet_writer,
                meta_rows,
                meta_lock,
                not args.no_json,
            ): m
            for m in shard_markets
        }
        for _ in as_completed(futures):
            pass

    if parquet_writer is not None:
        parquet_writer.close()
        meta_path = (
            Path(args.parquet_meta_path)
            if args.parquet_meta_path
            else Path(args.parquet_output_dir) / "markets_meta.parquet"
        )
        if meta_rows:
            table = pa.Table.from_pylist(meta_rows)
            pq.write_table(table, meta_path, compression="snappy")

    print(f"\nShard {shard_id} complete: {progress['done']} markets, {progress['points']:,} points")
    print(f"Output: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

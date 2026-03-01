#!/usr/bin/env python3
"""
Run whale-following backtest at category level using data/research.

Strategy: Follow high-conviction whale trades with category-level limits (30% max
per category), position sizing (Kelly), and "last signal wins" for conflicts.

Requires:
- data/research/{category}/trades.parquet, prices.parquet, markets_filtered.csv
- data/research/resolutions.csv (market_id, winner) OR prediction_markets.db for resolutions

Usage:
    python scripts/backtest/run_whale_category_backtest.py
    python scripts/backtest/run_whale_category_backtest.py --capital 1000000 --min-usd 1000
"""

import argparse
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "src"))

import numpy as np
import pandas as pd

from src.whale_strategy.research_data_loader import (
    load_research_trades,
    load_research_markets,
    load_resolution_winners,
    ResearchPriceStore,
    get_research_categories,
)
from src.whale_strategy.polymarket_whales import (
    identify_polymarket_whales,
    build_price_snapshot,
)
from src.whale_strategy.whale_following_strategy import (
    filter_and_score_signals,
    find_conflicting_position,
    handle_conflicting_signal,
    calculate_position_size,
    StrategyState,
    Position,
    WhaleSignal,
)
from src.whale_strategy.whale_config import load_whale_config, WhaleConfig
from src.whale_strategy.whale_scoring import WHALE_CRITERIA, MIN_WHALE_SCORE
from src.whale_strategy.whale_surprise import (
    build_surprise_positive_whale_set,
    build_volume_whale_set,
)
from trading.data_modules.costs import CostModel, DEFAULT_COST_MODEL
from trading.reporting import generate_quantstats_report


def _monthly_whale_worker(args):
    """Top-level worker: build surprise-positive whale set for one month.

    Reads trades from a temp parquet path so the large DataFrame is not
    serialized over pickle — the OS page cache makes re-reads fast.
    """
    (year, month, cutoff_str, trades_parquet_path, resolution_winners,
     min_surprise, min_trades_for_surprise, min_whale_wr,
     require_positive_surprise, volume_percentile) = args

    from src.whale_strategy.whale_surprise import build_surprise_positive_whale_set

    trades = pd.read_parquet(trades_parquet_path)
    cutoff = pd.Timestamp(cutoff_str)
    hist_df = trades[trades["datetime"] <= cutoff]
    if len(hist_df) < 1000:
        return (year, month), (set(), {}, {})

    w, s, wr = build_surprise_positive_whale_set(
        hist_df,
        resolution_winners,
        min_surprise=min_surprise,
        min_trades=min_trades_for_surprise,
        min_actual_win_rate=min_whale_wr,
        require_positive_surprise=require_positive_surprise,
        volume_percentile=volume_percentile,
    )
    return (year, month), (w, s, wr)


def _category_backtest_worker(args):
    """Top-level worker: run whale backtest for a single category."""
    (cat, research_dir_str, capital, min_usd, position_size, train_ratio,
     start_date, end_date, db_path,
     mode, volume_percentile, unfavored_only, unfavored_max_price,
     min_surprise, min_trades_for_surprise, min_whale_wr, require_positive_surprise,
     rebalance_freq) = args

    whale_config = WhaleConfig(
        mode=mode,
        volume_percentile=volume_percentile,
        unfavored_only=unfavored_only,
        unfavored_max_price=unfavored_max_price,
        min_surprise=min_surprise,
        min_trades_for_surprise=min_trades_for_surprise,
        min_whale_wr=min_whale_wr,
        require_positive_surprise=require_positive_surprise,
    )
    result = run_whale_category_backtest(
        research_dir=Path(research_dir_str),
        capital=capital,
        min_usd=min_usd,
        position_size=position_size,
        train_ratio=train_ratio,
        start_date=start_date,
        end_date=end_date,
        categories=[cat],
        db_path=db_path,
        whale_config=whale_config,
        surprise_only=whale_config.surprise_only,
        volume_only=whale_config.volume_only,
        unfavored_only=unfavored_only,
        rebalance_freq=rebalance_freq,
        n_workers=1,
    )
    return cat, result


def _to_datetime_safe(value) -> pd.Timestamp:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return pd.NaT
    if isinstance(value, (int, float)):
        ts = float(value)
        if ts > 10_000_000_000:
            ts /= 1000
        return pd.to_datetime(ts, unit="s", errors="coerce")
    return pd.to_datetime(value, errors="coerce")


def build_resolution_map_from_winners(
    resolution_winners: dict,
    markets_df: pd.DataFrame,
) -> dict:
    """Build resolution map {market_id: {resolution, resolution_date, ...}} from winners + markets."""
    resolutions = {}
    if markets_df.empty:
        return resolutions

    for _, row in markets_df.iterrows():
        mid = str(row.get("market_id", row.get("conditionId", ""))).strip().replace(".0", "")
        if not mid:
            continue

        winner = resolution_winners.get(mid)
        if winner not in ("YES", "NO"):
            resolutions[mid] = {
                "resolution": None,
                "is_resolved": False,
                "resolution_date": None,
                "market_close_date": None,
                "has_actual_resolution": False,
            }
            continue

        resolution = 1.0 if winner == "YES" else 0.0
        for k in ("closedTime", "endDateIso", "endDate"):
            if k in row and pd.notna(row.get(k)):
                try:
                    resolution_date = _to_datetime_safe(row[k])
                    if pd.notna(resolution_date):
                        break
                except Exception:
                    pass
        else:
            resolution_date = None

        resolutions[mid] = {
            "resolution": resolution,
            "is_resolved": True,
            "resolution_date": resolution_date,
            "market_close_date": resolution_date,
            "has_actual_resolution": True,
        }

    return resolutions


def run_whale_category_backtest(
    research_dir: Path,
    capital: float = 1_000_000,
    min_usd: float = 100,
    position_size: float = 25_000,
    train_ratio: float = 0.3,
    start_date: str = None,
    end_date: str = None,
    categories: list = None,
    db_path: str = None,
    whale_config: WhaleConfig = None,
    surprise_only: bool = None,
    volume_only: bool = None,
    unfavored_only: bool = None,
    unfavored_max_price: float = None,
    rebalance_freq: str = "1M",
    n_workers: int = 1,
) -> dict:
    """
    Run whale-following backtest with category limits.
    Uses whale_config for whale definition; CLI args override config when provided.
    """
    research_dir = Path(research_dir)
    cfg = whale_config or load_whale_config()
    surprise_only = surprise_only if surprise_only is not None else cfg.surprise_only
    volume_only = volume_only if volume_only is not None else cfg.volume_only
    unfavored_only = unfavored_only if unfavored_only is not None else cfg.unfavored_only
    unfavored_max_price = unfavored_max_price if unfavored_max_price is not None else cfg.unfavored_max_price
    min_usd = max(min_usd, cfg.min_usd)

    categories = categories or get_research_categories(research_dir)
    if not categories:
        return {"error": "No categories with trades.parquet found"}

    # Load data
    trades_df = load_research_trades(
        research_dir,
        categories=categories,
        min_usd=min_usd,
        start_date=start_date,
        end_date=end_date,
    )
    if trades_df.empty:
        return {"error": "No trades loaded"}

    markets_df = load_research_markets(research_dir, categories=categories)
    resolution_winners = load_resolution_winners(research_dir, db_path=db_path)

    if not resolution_winners:
        if surprise_only:
            return {"error": "Surprise-only mode requires resolutions. Run --extract-resolutions first."}
        if cfg.min_whale_wr > 0 or cfg.require_positive_surprise:
            return {"error": "Performance filter (WR>=50%%, positive surprise) requires resolutions. Run --extract-resolutions first."}
        print("Warning: No resolution data. Using simplified whale identification (no scoring).")

    # Train/test split
    split_date = trades_df["datetime"].quantile(train_ratio)
    train_df = trades_df[trades_df["datetime"] <= split_date]
    test_df = trades_df[trades_df["datetime"] > split_date]

    # Performance filter: WR >= min_whale_wr and positive surprise (default when resolutions exist)
    use_performance_filter = resolution_winners and (cfg.min_whale_wr > 0 or cfg.require_positive_surprise)

    # Build monthly whale sets for rolling rebalancing (when resolutions + rebalance_freq)
    monthly_whale_sets: dict = {}  # (year, month) -> (whales, scores, winrates)
    whale_scores_override = None
    whale_winrates_override = None
    whales = set()

    if use_performance_filter and rebalance_freq:
        # Rolling monthly: re-identify whales at start of each month from data up to end of previous month
        test_months = test_df["datetime"].dt.to_period("M").unique()
        if n_workers > 1 and len(test_months) > 1:
            # Write trades to temp parquet so workers load from disk (no pickle of large DataFrame)
            _tmp_fd, _tmp_path = tempfile.mkstemp(suffix=".parquet")
            os.close(_tmp_fd)
            try:
                trades_df.to_parquet(_tmp_path)
                month_args = [
                    (p.year, p.month,
                     str(p.to_timestamp(how="start") - pd.Timedelta(days=1)),
                     _tmp_path, resolution_winners,
                     cfg.min_surprise, cfg.min_trades_for_surprise, cfg.min_whale_wr,
                     cfg.require_positive_surprise, cfg.volume_percentile)
                    for p in sorted(test_months)
                ]
                n_month_workers = min(n_workers, len(month_args))
                print(f"  Parallel monthly whale sets: {len(month_args)} months, {n_month_workers} workers")
                with ProcessPoolExecutor(max_workers=n_month_workers) as pool:
                    for ym, whale_data in pool.map(_monthly_whale_worker, month_args):
                        monthly_whale_sets[ym] = whale_data
                        whales |= whale_data[0]
            finally:
                Path(_tmp_path).unlink(missing_ok=True)
        else:
            for period in sorted(test_months):
                ym = (period.year, period.month)
                cutoff = period.to_timestamp(how="start") - pd.Timedelta(days=1)  # End of previous month
                hist_df = trades_df[trades_df["datetime"] <= cutoff]
                if len(hist_df) < 1000:
                    monthly_whale_sets[ym] = (set(), {}, {})
                    continue
                w, s, wr = build_surprise_positive_whale_set(
                    hist_df,
                    resolution_winners,
                    min_surprise=cfg.min_surprise,
                    min_trades=cfg.min_trades_for_surprise,
                    min_actual_win_rate=cfg.min_whale_wr,
                    require_positive_surprise=cfg.require_positive_surprise,
                    volume_percentile=cfg.volume_percentile,
                )
                monthly_whale_sets[ym] = (w, s, wr)
                whales |= w
        if whales:
            # Merge scores/winrates from latest month (or first non-empty)
            for ym in reversed(sorted(monthly_whale_sets.keys())):
                w, s, wr = monthly_whale_sets[ym]
                if s:
                    whale_scores_override = s
                    whale_winrates_override = wr
                    break
        print(f"  Rolling monthly: {len(whales)} qualified whales (WR>={cfg.min_whale_wr*100:.0f}%, surprise>0) across {len(monthly_whale_sets)} months")
    elif use_performance_filter and resolution_winners:
        # Single train-period whale set with performance filter
        whales, whale_scores_override, whale_winrates_override = build_surprise_positive_whale_set(
            train_df,
            resolution_winners,
            min_surprise=cfg.min_surprise,
            min_trades=cfg.min_trades_for_surprise,
            min_actual_win_rate=cfg.min_whale_wr,
            require_positive_surprise=cfg.require_positive_surprise,
            volume_percentile=cfg.volume_percentile,
        )
        print(f"  Qualified whales: {len(whales)} (WR>={cfg.min_whale_wr*100:.0f}%, surprise>0)")
    elif volume_only and not resolution_winners:
        whales, whale_scores_override, whale_winrates_override = build_volume_whale_set(
            train_df, volume_percentile=cfg.volume_percentile
        )
        print(f"  Volume-only (no resolutions): {len(whales)} whales")
    elif volume_only:
        whales, whale_scores_override, whale_winrates_override = build_volume_whale_set(
            train_df, volume_percentile=cfg.volume_percentile
        )
        print(f"  Volume-only (no perf filter): {len(whales)} whales")
    else:
        price_snapshot = build_price_snapshot(train_df)
        try:
            whales = identify_polymarket_whales(
                train_df,
                method="mid_price_accuracy" if resolution_winners else "top_returns",
                role="maker",
                min_trades=10,
                min_volume=1000,
                resolution_winners=resolution_winners if resolution_winners else None,
                price_snapshot=price_snapshot,
            )
        except ValueError:
            # Fallback when resolution_winners empty and top_returns fails
            whales = identify_polymarket_whales(
                train_df,
                method="volume_top10",
                role="maker",
                min_trades=10,
                min_volume=1000,
            )

    if not whales:
        # Fallback: top by volume
        stats = train_df.groupby("maker").agg(
            total_volume=("usd_amount", "sum"),
            trade_count=("usd_amount", "count"),
        ).reset_index()
        stats = stats[stats["trade_count"] >= 10]
        if stats.empty:
            return {"error": "No qualifying whales"}
        whales = set(stats.nlargest(50, "total_volume")["maker"])

    # Build resolution map
    resolutions = build_resolution_map_from_winners(resolution_winners, markets_df)

    # Market liquidity
    market_liquidity = {}
    for _, row in markets_df.iterrows():
        mid = str(row.get("market_id", "")).strip().replace(".0", "")
        liq = row.get("liquidityNum") or row.get("liquidity") or row.get("volumeNum") or row.get("volume")
        market_liquidity[mid] = float(liq) if liq is not None else 100_000

    # Filter to unfavored trades only (underdog: BUY <=40c, SELL >=60c)
    signals_df = test_df
    if unfavored_only:
        direction = test_df["maker_direction"].str.upper()
        price = test_df["price"].astype(float)
        mask = (
            ((direction == "BUY") & (price <= unfavored_max_price)) |
            ((direction == "SELL") & (price >= (1 - unfavored_max_price)))
        )
        signals_df = test_df[mask].copy()
        print(f"  Unfavored-only: {len(signals_df):,} / {len(test_df):,} test trades")

    # Get signals (use union of all monthly whale sets when rolling, so we capture all potential signals)
    signal_min_usd = min(min_usd, 1000)  # Lower for research data
    min_pos_override = 1000 if (volume_only or unfavored_only or use_performance_filter) else None
    min_liq_override = 10000 if (volume_only or unfavored_only or use_performance_filter) else None
    signals = filter_and_score_signals(
        signals_df,
        resolution_winners or {},
        markets_df,
        whale_set=whales,
        min_usd=signal_min_usd,
        role="maker",
        whale_scores_override=whale_scores_override,
        whale_winrates_override=whale_winrates_override,
        min_position_size_override=min_pos_override,
        min_market_liquidity_override=min_liq_override,
    )

    if not signals:
        # Fallback: use whale trades directly as signals
        whale_trades = signals_df[
            (signals_df["maker"].isin(whales)) &
            (signals_df["usd_amount"] >= min_usd)
        ]
        signals = []
        for _, row in whale_trades.iterrows():
            mid = str(row["market_id"]).strip().replace(".0", "")
            signals.append(WhaleSignal(
                market_id=mid,
                category=row.get("category", "Unknown"),
                whale_address=row["maker"],
                side=row.get("maker_direction", "BUY").upper(),
                price=float(row["price"]),
                size_usd=float(row["usd_amount"]),
                score=7.0,  # Default
                datetime=row["datetime"],
                historical_winrate=0.55,
            ))

    # Price store
    price_store = ResearchPriceStore(research_dir, categories=categories)

    # Backtest loop
    state = StrategyState(total_capital=capital)
    open_positions = {}
    closed_trades = []
    cost_model = DEFAULT_COST_MODEL

    # Sort signals by datetime
    signals_sorted = sorted(signals, key=lambda s: s.datetime)
    test_dates = sorted(set(s.datetime.normalize() for s in signals_sorted))

    for current_date in test_dates:
        # 1. Close positions that resolve
        to_close = []
        for mid, pos in open_positions.items():
            res = resolutions.get(mid, {})
            res_date = res.get("resolution_date")
            if res_date and pd.to_datetime(res_date).date() <= current_date.date():
                to_close.append((mid, "resolution", res.get("resolution")))
                continue
            close_date = res.get("market_close_date")
            if close_date and pd.to_datetime(close_date).date() <= current_date.date():
                to_close.append((mid, "market_close", None))

        for mid, reason, resolution in to_close:
            pos = open_positions.pop(mid)
            if reason == "resolution" and resolution is not None:
                # resolution = YES token payout (1 if YES wins, 0 if NO wins)
                if pos.side.upper() == "BUY":
                    exit_price = resolution  # YES token
                else:
                    exit_price = 1.0 - resolution  # SELL = short NO; NO token = 1 when NO wins
            else:
                clob_price = price_store.price_at_or_before(mid, current_date)
                if clob_price is None:
                    clob_price = pos.entry_price  # Fallback
                if pos.side.upper() == "BUY":
                    exit_price = clob_price  # YES price from CLOB
                else:
                    exit_price = 1.0 - clob_price  # NO price = 1 - YES price
            direction = pos.side.upper()
            if direction == "BUY":
                # BUY YES: shares = size_usd/entry_price, PnL = (exit - entry) * shares
                gross_pnl = (exit_price - pos.entry_price) * (pos.size_usd / pos.entry_price)
            else:
                # SELL NO: shares = size_usd/entry_price (entry = NO price), PnL = (entry - exit) * shares
                gross_pnl = (pos.entry_price - exit_price) * (pos.size_usd / pos.entry_price)

            net_pnl = gross_pnl * 0.97  # Cost estimate
            closed_trades.append({
                "market_id": mid,
                "entry_date": pos.entry_date,
                "exit_date": current_date,
                "direction": direction,
                "entry_price": pos.entry_price,
                "exit_price": exit_price,
                "gross_pnl": gross_pnl,
                "net_pnl": net_pnl,
                "position_size": pos.size_usd,
                "whale_address": pos.whale_address,
                "category": pos.category,
            })

            # Update state
            state.positions = [p for p in state.positions if p.market_id != mid]
            state.category_exposure[pos.category] = state.category_exposure.get(pos.category, 0) - pos.size_usd
            state.whale_exposure[pos.whale_address] = state.whale_exposure.get(pos.whale_address, 0) - pos.size_usd
            state.market_exposure.pop(mid, None)

        # 2. Process signals for today
        today_signals = [s for s in signals_sorted if s.datetime.normalize() == current_date]
        for sig in today_signals:
            # Rolling: only follow if whale is in active set for this month
            if monthly_whale_sets:
                ym = (sig.datetime.year, sig.datetime.month)
                active_whales, _, _ = monthly_whale_sets.get(ym, (set(), {}, {}))
                if sig.whale_address not in active_whales:
                    continue
            mid = sig.market_id
            if mid in open_positions:
                pos = open_positions[mid]
                if pos.side != sig.side:
                    # Conflicting: close and flip
                    clob_price = price_store.price_at_or_before(mid, current_date)
                    if clob_price is None:
                        continue
                    if pos.side == "BUY":
                        exit_price = clob_price
                        gross = (exit_price - pos.entry_price) * (pos.size_usd / pos.entry_price)
                    else:
                        exit_price = 1.0 - clob_price  # NO price
                        gross = (pos.entry_price - exit_price) * (pos.size_usd / pos.entry_price)
                    closed_trades.append({
                        "market_id": mid, "entry_date": pos.entry_date, "exit_date": current_date,
                        "direction": pos.side, "entry_price": pos.entry_price, "exit_price": exit_price,
                        "gross_pnl": gross, "net_pnl": gross * 0.97, "position_size": pos.size_usd,
                        "whale_address": pos.whale_address, "category": pos.category,
                        "reason": "CONFLICTING_SIGNAL",
                    })
                    open_positions.pop(mid)
                    state.positions = [p for p in state.positions if p.market_id != mid]
                    state.category_exposure[pos.category] = state.category_exposure.get(pos.category, 0) - pos.size_usd
                    state.whale_exposure[pos.whale_address] = state.whale_exposure.get(pos.whale_address, 0) - pos.size_usd
                    state.market_exposure.pop(mid, None)
                else:
                    continue  # Same side, skip

            if mid in open_positions:
                continue

            res = resolutions.get(mid, {})
            if res.get("resolution_date") and pd.to_datetime(res["resolution_date"]).date() <= current_date.date():
                continue

            size = calculate_position_size(sig, state, market_liquidity.get(mid, 100_000))
            if size is None or size < 1000:
                continue

            if state.available() < size:
                continue

            direction = "buy" if sig.side == "BUY" else "sell"
            entry_price = sig.price

            pos = Position(
                market_id=mid,
                category=sig.category,
                side=sig.side,
                entry_price=entry_price,
                size_usd=size,
                whale_address=sig.whale_address,
                whale_score=sig.score,
                entry_date=pd.to_datetime(current_date),
            )
            open_positions[mid] = pos
            state.positions.append(pos)
            state.category_exposure[sig.category] = state.category_exposure.get(sig.category, 0) + size
            state.whale_exposure[sig.whale_address] = state.whale_exposure.get(sig.whale_address, 0) + size
            state.market_exposure[mid] = size

    # Close remaining at last date
    if test_dates:
        last_date = test_dates[-1]
        for mid, pos in list(open_positions.items()):
            clob_price = price_store.price_at_or_before(mid, last_date) or pos.entry_price
            if pos.side == "BUY":
                exit_price = clob_price
                gross = (exit_price - pos.entry_price) * (pos.size_usd / pos.entry_price)
            else:
                exit_price = 1.0 - clob_price  # NO price
                gross = (pos.entry_price - exit_price) * (pos.size_usd / pos.entry_price)
            closed_trades.append({
                "market_id": mid, "entry_date": pos.entry_date, "exit_date": last_date,
                "direction": pos.side, "entry_price": pos.entry_price, "exit_price": exit_price,
                "gross_pnl": gross, "net_pnl": gross * 0.97, "position_size": pos.size_usd,
                "whale_address": pos.whale_address, "category": pos.category,
                "reason": "OPEN_AT_END",
            })

    price_store.close()

    # Summary
    if not closed_trades:
        return {"error": "No closed trades", "whales": len(whales), "signals": len(signals)}

    df = pd.DataFrame(closed_trades)
    total_pnl = df["net_pnl"].sum()
    wins = (df["net_pnl"] > 0).sum()
    total = len(df)

    # Build daily equity curve for QuantStats (PnL realized on exit_date)
    daily_pnl = df.groupby(pd.to_datetime(df["exit_date"]).dt.normalize())["net_pnl"].sum()
    date_range = pd.date_range(start=daily_pnl.index.min(), end=daily_pnl.index.max(), freq="D")
    daily_pnl = daily_pnl.reindex(date_range, fill_value=0).sort_index()
    cumulative_pnl = daily_pnl.cumsum()
    equity = capital + cumulative_pnl
    daily_returns = equity.pct_change().dropna()

    return {
        "total_trades": total,
        "win_rate": wins / total if total else 0,
        "total_net_pnl": total_pnl,
        "roi_pct": (total_pnl / capital) * 100,
        "whales_followed": len(whales),
        "signals_processed": len(signals),
        "categories": categories,
        "trades_df": df,
        "daily_returns": daily_returns,
        "daily_equity": equity,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Whale-following backtest at category level")
    parser.add_argument("--research-dir", type=Path, default=_project_root / "data" / "research")
    parser.add_argument("--capital", type=float, default=1_000_000)
    parser.add_argument("--min-usd", type=float, default=100)
    parser.add_argument("--position-size", type=float, default=25_000)
    parser.add_argument("--train-ratio", type=float, default=0.3)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument(
        "--category",
        default=None,
        help="Single category to run (e.g. Tech). Overrides --categories.",
    )
    parser.add_argument(
        "--categories",
        default=None,
        help="Comma-separated categories (default: all). Ignored if --category set.",
    )
    parser.add_argument(
        "--mode",
        choices=["combined", "per-category"],
        default="combined",
        help="combined: all categories together; per-category: run each category separately",
    )
    parser.add_argument("--db-path", default=None, help="DB for resolutions if no CSV")
    parser.add_argument(
        "--extract-resolutions",
        action="store_true",
        help="Extract resolutions from markets_filtered.csv before backtest (run if no resolutions.csv)",
    )
    parser.add_argument(
        "--surprise-only",
        action="store_true",
        help="Only follow whales with positive surprise. Requires resolutions. Overrides config whale_mode.",
    )
    parser.add_argument(
        "--volume-only",
        action="store_true",
        help="Whales = Nth percentile volume in market only. Overrides config whale_mode.",
    )
    parser.add_argument(
        "--unfavored-only",
        action="store_true",
        help="Only follow unfavored (underdog) trades: BUY <=40c, SELL >=60c. Overrides config.",
    )
    parser.add_argument("--output", default=None, help="Output CSV for trades")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_project_root / "data" / "output" / "whale_following",
        help="Output directory for QuantStats tearsheet and trades (default: data/output/whale_following)",
    )
    parser.add_argument("--no-quantstats", action="store_true", help="Skip QuantStats tearsheet")
    parser.add_argument("--no-tracker", action="store_true", help="Skip experiment tracker (experiments.db)")
    parser.add_argument("--backtests-dir", type=Path, default=_project_root / "backtests", help="Directory for backtest storage and experiments.db")
    parser.add_argument("--no-rebalance", action="store_true", help="Disable monthly whale rebalancing (use single train-period whale set)")
    parser.add_argument(
        "--workers", type=int, default=os.cpu_count() or 1,
        help="Parallel workers: per-category mode distributes categories; combined mode parallelises monthly whale building (default: all CPUs)",
    )
    args = parser.parse_args()

    research_dir = args.research_dir

    # Extract resolutions first if requested or missing
    resolutions_path = research_dir / "resolutions.csv"
    if args.extract_resolutions or (not resolutions_path.exists() and args.db_path is None):
        import subprocess
        extract_script = _project_root / "scripts" / "data" / "extract_resolutions_from_markets.py"
        if extract_script.exists():
            cmd = [sys.executable, str(extract_script), "--research-dir", str(research_dir)]
            if args.category:
                cmd += ["--categories", args.category]
            elif args.categories:
                cmd += ["--categories", args.categories]
            try:
                subprocess.run(cmd, cwd=str(_project_root), check=True)
            except subprocess.CalledProcessError as e:
                print(f"Warning: Resolution extraction failed: {e}")

    # Resolve categories
    if args.category:
        categories = [args.category]
    elif args.categories:
        categories = [c.strip() for c in args.categories.split(",")]
    else:
        categories = None

    # Resolve whale mode from CLI or config
    whale_config = load_whale_config()
    if args.volume_only:
        whale_config.mode = "volume_only"
    if args.surprise_only:
        whale_config.mode = "surprise_only"
    surprise_only = whale_config.surprise_only
    volume_only = whale_config.volume_only
    unfavored_only = whale_config.unfavored_only

    if surprise_only and not (research_dir / "resolutions.csv").exists() and not args.db_path:
        print("Error: --surprise-only requires resolutions. Run with --extract-resolutions first.")
        return 1

    print("Whale Category Backtest")
    print("  research_dir ", research_dir)
    print("  capital      ", args.capital)
    print("  min_usd      ", args.min_usd)
    print("  mode         ", args.mode)
    print("  whale_config ", whale_config)
    print("  surprise_only", surprise_only)
    print("  volume_only  ", volume_only)
    print("  unfavored_only", unfavored_only)
    print("  categories   ", categories or "all")
    print("  workers      ", args.workers)

    if args.mode == "per-category":
        # Run each category separately
        cats_to_run = categories or get_research_categories(research_dir)
        if not cats_to_run:
            print("No categories found.")
            return 1

        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        all_results = []

        def _save_cat_result(cat, result):
            if "error" in result:
                print(f"  {cat}: Error: {result['error']}")
                return None
            cat_out = output_dir / cat.replace(" ", "_")
            cat_out.mkdir(parents=True, exist_ok=True)
            print(f"  {cat}: Trades={result['total_trades']:,}  Win%={result['win_rate']*100:.1f}  "
                  f"P&L=${result['total_net_pnl']:,.0f}  ROI={result['roi_pct']:.1f}%")
            if "trades_df" in result:
                result["trades_df"].to_csv(cat_out / "whale_backtest_trades.csv", index=False)
            if not args.no_quantstats and "daily_returns" in result and len(result["daily_returns"]) >= 5:
                ok = generate_quantstats_report(
                    result["daily_returns"],
                    str(cat_out / "quantstats_whale_following.html"),
                    title=f"Whale Following - {cat}",
                )
                if ok:
                    print(f"  QuantStats: {cat_out / 'quantstats_whale_following.html'}")
            return (cat, result)

        n_cat_workers = min(args.workers, len(cats_to_run))
        if n_cat_workers > 1:
            print(f"Running {len(cats_to_run)} categories in parallel ({n_cat_workers} workers)...")
            worker_args_list = [
                (cat, str(research_dir), args.capital, args.min_usd, args.position_size,
                 args.train_ratio, args.start_date, args.end_date, args.db_path,
                 whale_config.mode, whale_config.volume_percentile,
                 whale_config.unfavored_only, whale_config.unfavored_max_price,
                 whale_config.min_surprise, whale_config.min_trades_for_surprise,
                 whale_config.min_whale_wr, whale_config.require_positive_surprise,
                 "1M" if not args.no_rebalance else None)
                for cat in cats_to_run
            ]
            with ProcessPoolExecutor(max_workers=n_cat_workers) as pool:
                futures = {pool.submit(_category_backtest_worker, wa): wa[0] for wa in worker_args_list}
                for future in as_completed(futures):
                    cat = futures[future]
                    try:
                        cat, result = future.result()
                        r = _save_cat_result(cat, result)
                        if r:
                            all_results.append(r)
                    except Exception as exc:
                        print(f"  {cat}: Exception: {exc}")
        else:
            for cat in cats_to_run:
                print(f"\n--- Category: {cat} ---")
                result = run_whale_category_backtest(
                    research_dir=research_dir,
                    capital=args.capital,
                    min_usd=args.min_usd,
                    position_size=args.position_size,
                    train_ratio=args.train_ratio,
                    start_date=args.start_date,
                    end_date=args.end_date,
                    categories=[cat],
                    db_path=args.db_path,
                    whale_config=whale_config,
                    surprise_only=surprise_only,
                    volume_only=volume_only,
                    unfavored_only=unfavored_only,
                    rebalance_freq="1M" if not args.no_rebalance else None,
                    n_workers=1,
                )
                r = _save_cat_result(cat, result)
                if r:
                    all_results.append(r)

        if not all_results:
            return 1

        print("\n" + "=" * 60)
        print("SUMMARY (per-category)")
        print("=" * 60)
        for cat, r in all_results:
            print(f"  {cat}: {r['total_trades']} trades, {r['win_rate']*100:.1f}% win, "
                  f"${r['total_net_pnl']:,.0f} P&L, {r['roi_pct']:.1f}% ROI")
        return 0

    # combined mode (default)
    result = run_whale_category_backtest(
        research_dir=research_dir,
        capital=args.capital,
        min_usd=args.min_usd,
        position_size=args.position_size,
        train_ratio=args.train_ratio,
        start_date=args.start_date,
        end_date=args.end_date,
        categories=categories,
        db_path=args.db_path,
        whale_config=whale_config,
        surprise_only=surprise_only,
        volume_only=volume_only,
        unfavored_only=unfavored_only,
        rebalance_freq="1M" if not args.no_rebalance else None,
        n_workers=args.workers,
    )

    if "error" in result:
        print(f"\nError: {result['error']}")
        if "whales" in result:
            print(f"  Whales: {result['whales']}, Signals: {result.get('signals', 0)}")
        return 1

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Total Trades:     {result['total_trades']:,}")
    print(f"Win Rate:         {result['win_rate']*100:.1f}%")
    print(f"Net P&L:         ${result['total_net_pnl']:,.2f}")
    print(f"ROI:              {result['roi_pct']:.2f}%")
    print(f"Whales Followed:  {result['whales_followed']:,}")
    print(f"Signals:          {result['signals_processed']:,}")
    print(f"Categories:       {', '.join(result['categories'])}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.output and "trades_df" in result:
        result["trades_df"].to_csv(args.output, index=False)
        print(f"\nTrades saved to {args.output}")
    elif "trades_df" in result:
        trades_path = output_dir / "whale_backtest_trades.csv"
        result["trades_df"].to_csv(trades_path, index=False)
        print(f"\nTrades saved to {trades_path}")

    if not args.no_quantstats and "daily_returns" in result and len(result["daily_returns"]) >= 5:
        quantstats_path = output_dir / "quantstats_whale_following.html"
        if generate_quantstats_report(
            result["daily_returns"],
            str(quantstats_path),
            title="Whale Following Strategy (Category-Level)",
        ):
            print(f"QuantStats tearsheet: {quantstats_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

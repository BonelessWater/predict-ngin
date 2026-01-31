"""
Whale Analysis - Analyze trader patterns and whale behavior.
Works with available trade data to identify whales and their patterns.

Usage:
    python src/whales/analyze.py              # All markets
    python src/whales/analyze.py --xrp        # XRP only
"""
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Set
import json
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.config import get_config, Config


@dataclass
class WhaleStats:
    """Statistics for a detected whale."""
    address: str
    total_volume: float
    trade_count: int
    avg_trade_size: float
    buy_volume: float
    sell_volume: float
    buy_ratio: float
    markets_traded: int
    win_estimate: float  # Based on price movement after trades


@dataclass
class MarketStats:
    """Statistics for a market."""
    market: str
    total_volume: float
    trade_count: int
    unique_traders: int
    whale_volume: float
    whale_volume_pct: float
    avg_price: float
    price_range: float


def load_trade_data(xrp_only: bool = False) -> pd.DataFrame:
    """Load trade data with trader information."""
    config = get_config()
    bulk_dir = config.data.parquet_dir / 'bulk'

    files = list(bulk_dir.glob('all_trades_*.parquet'))
    if not files:
        return pd.DataFrame()

    df = pd.read_parquet(sorted(files)[-1])

    # Deduplicate
    if 'transactionHash' in df.columns:
        df = df.drop_duplicates(subset=['transactionHash'])

    # Parse timestamp
    if 'timestamp' in df.columns:
        df['timestamp_dt'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')

    # Filter XRP
    if xrp_only and 'title' in df.columns:
        mask = df['title'].str.lower().str.contains('xrp', na=False)
        df = df[mask]

    return df


def detect_whales_all_methods(trades: pd.DataFrame, config: Config) -> Dict[str, Set[str]]:
    """Detect whales using all three methods."""
    if trades.empty:
        return {}

    trader_col = 'proxyWallet'
    size_col = 'size'

    if trader_col not in trades.columns:
        return {}

    # Aggregate by trader
    stats = trades.groupby(trader_col).agg({
        size_col: ['sum', 'mean', 'count']
    })
    stats.columns = ['total_volume', 'avg_trade', 'trade_count']
    stats = stats.reset_index()

    whales = {}

    # Method 1: Top N by volume
    n = config.whale.top_n
    whales['top_n'] = set(stats.nlargest(n, 'total_volume')[trader_col])

    # Method 2: Above percentile
    pct = config.whale.percentile
    threshold = stats['total_volume'].quantile(pct / 100)
    whales['percentile'] = set(stats[stats['total_volume'] >= threshold][trader_col])

    # Method 3: Large average trade size
    min_size = config.whale.min_trade_size
    # Adjust for our data (may have smaller trades)
    actual_threshold = max(min_size, stats['avg_trade'].quantile(0.95))
    whales['trade_size'] = set(stats[stats['avg_trade'] >= actual_threshold][trader_col])

    return whales


def analyze_whale_behavior(trades: pd.DataFrame, whales: Set[str]) -> List[WhaleStats]:
    """Analyze behavior patterns for detected whales."""
    if trades.empty or not whales:
        return []

    trader_col = 'proxyWallet'
    size_col = 'size'

    whale_stats = []

    for whale in whales:
        whale_trades = trades[trades[trader_col] == whale]

        if whale_trades.empty:
            continue

        total_vol = whale_trades[size_col].sum()
        trade_count = len(whale_trades)
        avg_size = whale_trades[size_col].mean()

        # Buy/sell breakdown
        if 'side' in whale_trades.columns:
            buys = whale_trades[whale_trades['side'].str.upper() == 'BUY'][size_col].sum()
            sells = whale_trades[whale_trades['side'].str.upper() == 'SELL'][size_col].sum()
        else:
            buys = total_vol / 2
            sells = total_vol / 2

        buy_ratio = buys / (buys + sells) if (buys + sells) > 0 else 0.5

        # Markets traded
        if 'title' in whale_trades.columns:
            markets = whale_trades['title'].nunique()
        else:
            markets = 1

        # Estimate win rate (crude: based on price movement after trade)
        # This would need more sophisticated analysis with price data
        win_estimate = 0.5  # Placeholder

        whale_stats.append(WhaleStats(
            address=whale,
            total_volume=total_vol,
            trade_count=trade_count,
            avg_trade_size=avg_size,
            buy_volume=buys,
            sell_volume=sells,
            buy_ratio=buy_ratio,
            markets_traded=markets,
            win_estimate=win_estimate
        ))

    return sorted(whale_stats, key=lambda x: -x.total_volume)


def analyze_markets(trades: pd.DataFrame, whales: Set[str]) -> List[MarketStats]:
    """Analyze whale activity per market."""
    if trades.empty or 'title' not in trades.columns:
        return []

    market_stats = []
    trader_col = 'proxyWallet'
    size_col = 'size'

    for market in trades['title'].unique():
        market_trades = trades[trades['title'] == market]

        total_vol = market_trades[size_col].sum()
        trade_count = len(market_trades)
        unique_traders = market_trades[trader_col].nunique()

        # Whale volume in this market
        whale_trades = market_trades[market_trades[trader_col].isin(whales)]
        whale_vol = whale_trades[size_col].sum() if not whale_trades.empty else 0
        whale_pct = whale_vol / total_vol * 100 if total_vol > 0 else 0

        # Price stats
        if 'price' in market_trades.columns:
            avg_price = market_trades['price'].mean()
            price_range = market_trades['price'].max() - market_trades['price'].min()
        else:
            avg_price = 0
            price_range = 0

        market_stats.append(MarketStats(
            market=market,
            total_volume=total_vol,
            trade_count=trade_count,
            unique_traders=unique_traders,
            whale_volume=whale_vol,
            whale_volume_pct=whale_pct,
            avg_price=avg_price,
            price_range=price_range
        ))

    return sorted(market_stats, key=lambda x: -x.total_volume)


def calculate_consensus(trades: pd.DataFrame, whales: Set[str]) -> Dict[str, dict]:
    """Calculate whale consensus per market."""
    if trades.empty or not whales:
        return {}

    trader_col = 'proxyWallet'
    size_col = 'size'

    consensus = {}

    for market in trades['title'].unique() if 'title' in trades.columns else ['all']:
        if 'title' in trades.columns:
            market_trades = trades[trades['title'] == market]
        else:
            market_trades = trades

        whale_trades = market_trades[market_trades[trader_col].isin(whales)]

        if whale_trades.empty:
            continue

        if 'side' in whale_trades.columns:
            buy_vol = whale_trades[whale_trades['side'].str.upper() == 'BUY'][size_col].sum()
            sell_vol = whale_trades[whale_trades['side'].str.upper() == 'SELL'][size_col].sum()
        else:
            buy_vol = whale_trades[size_col].sum() / 2
            sell_vol = whale_trades[size_col].sum() / 2

        total = buy_vol + sell_vol
        if total > 0:
            buy_pct = buy_vol / total
            direction = 'BUY' if buy_pct > 0.5 else 'SELL'
            strength = max(buy_pct, 1 - buy_pct)

            consensus[market] = {
                'direction': direction,
                'strength': strength,
                'buy_volume': buy_vol,
                'sell_volume': sell_vol,
                'whale_count': whale_trades[trader_col].nunique()
            }

    return consensus


def print_analysis(
    trades: pd.DataFrame,
    whales_by_method: Dict[str, Set[str]],
    config: Config,
    xrp_only: bool = False
):
    """Print comprehensive whale analysis."""

    print("="*70)
    print(f"WHALE ANALYSIS {'(XRP ONLY)' if xrp_only else '(ALL MARKETS)'}")
    print("="*70)

    print(f"\nData Summary:")
    print(f"  Total trades: {len(trades):,}")
    print(f"  Unique traders: {trades['proxyWallet'].nunique():,}")
    print(f"  Total volume: ${trades['size'].sum():,.2f}")
    if 'title' in trades.columns:
        print(f"  Unique markets: {trades['title'].nunique()}")

    if 'timestamp_dt' in trades.columns:
        time_span = trades['timestamp_dt'].max() - trades['timestamp_dt'].min()
        print(f"  Time span: {time_span}")

    # Whale detection results
    print(f"\n{'='*70}")
    print("WHALE DETECTION METHODS")
    print("="*70)

    for method, whale_set in whales_by_method.items():
        whale_trades = trades[trades['proxyWallet'].isin(whale_set)]
        whale_vol = whale_trades['size'].sum() if not whale_trades.empty else 0
        vol_pct = whale_vol / trades['size'].sum() * 100 if trades['size'].sum() > 0 else 0

        print(f"\n{method.upper()}:")
        print(f"  Whales detected: {len(whale_set)}")
        print(f"  Whale volume: ${whale_vol:,.2f} ({vol_pct:.1f}% of total)")

    # Use top_n method for detailed analysis
    whales = whales_by_method.get('top_n', set())

    if whales:
        # Top whales
        print(f"\n{'='*70}")
        print(f"TOP {len(whales)} WHALES (by volume)")
        print("="*70)
        print(f"{'Address':<15} {'Volume':>12} {'Trades':>7} {'Avg Size':>10} {'Buy%':>6}")
        print("-"*70)

        whale_stats = analyze_whale_behavior(trades, whales)
        for ws in whale_stats[:15]:
            addr_short = ws.address[:12] + '...'
            print(f"{addr_short:<15} ${ws.total_volume:>10,.0f} {ws.trade_count:>7} ${ws.avg_trade_size:>9,.0f} {ws.buy_ratio*100:>5.1f}%")

        # Market breakdown
        print(f"\n{'='*70}")
        print("WHALE ACTIVITY BY MARKET")
        print("="*70)
        print(f"{'Market':<45} {'Volume':>10} {'Whale%':>7}")
        print("-"*70)

        market_stats = analyze_markets(trades, whales)
        for ms in market_stats[:15]:
            market_short = ms.market[:43] + '..' if len(ms.market) > 45 else ms.market
            print(f"{market_short:<45} ${ms.total_volume:>8,.0f} {ms.whale_volume_pct:>6.1f}%")

        # Whale consensus
        print(f"\n{'='*70}")
        print("WHALE CONSENSUS BY MARKET")
        print("="*70)
        print(f"{'Market':<40} {'Direction':>10} {'Strength':>10} {'Whales':>7}")
        print("-"*70)

        consensus = calculate_consensus(trades, whales)
        for market, data in sorted(consensus.items(), key=lambda x: -x[1]['strength'])[:15]:
            market_short = market[:38] + '..' if len(market) > 40 else market
            print(f"{market_short:<40} {data['direction']:>10} {data['strength']*100:>9.1f}% {data['whale_count']:>7}")

    # Method comparison
    print(f"\n{'='*70}")
    print("METHOD COMPARISON")
    print("="*70)

    # Find overlaps
    all_methods = list(whales_by_method.keys())
    if len(all_methods) >= 2:
        for i, m1 in enumerate(all_methods):
            for m2 in all_methods[i+1:]:
                overlap = whales_by_method[m1] & whales_by_method[m2]
                print(f"  {m1} & {m2}: {len(overlap)} whales in common")

        # All three methods
        if len(all_methods) == 3:
            all_overlap = whales_by_method['top_n'] & whales_by_method['percentile'] & whales_by_method['trade_size']
            print(f"  All methods agree: {len(all_overlap)} whales")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Whale Analysis')
    parser.add_argument('--xrp', action='store_true', help='XRP markets only')

    args = parser.parse_args()
    config = get_config()

    # Load data
    trades = load_trade_data(xrp_only=args.xrp)

    if trades.empty:
        print("No trade data found!")
        print("Run: python src/data/historical_collector.py")
        return

    # Detect whales
    whales_by_method = detect_whales_all_methods(trades, config)

    # Print analysis
    print_analysis(trades, whales_by_method, config, xrp_only=args.xrp)

    # Save results
    output_dir = config.data.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = '_xrp' if args.xrp else ''

    # Save whale list
    whale_data = {method: list(whales) for method, whales in whales_by_method.items()}
    with open(output_dir / f'whales{suffix}.json', 'w') as f:
        json.dump(whale_data, f, indent=2)

    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()

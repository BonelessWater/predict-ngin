#!/usr/bin/env python3
"""
Detailed analysis of market categories: returns, liquidity, users, trading patterns.

Generates plots and statistics for each category and saves results to docs/results/preliminary_research.
"""

import sys
from pathlib import Path
from collections import defaultdict
import json
from datetime import datetime

# Add project root to path
_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "src"))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from trading.data_modules.database import PredictionMarketDB
    from trading.data_modules.categories import categorize_market, CATEGORIES
    from trading.data_modules.parquet_store import PriceStore, TradeStore
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)


def load_markets(parquet_dir: str = "data/polymarket") -> pd.DataFrame:
    """Load markets from parquet files.
    Supports: markets.parquet (single) or markets_*.parquet (sharded).
    """
    markets_dir = Path(parquet_dir)
    if not markets_dir.exists():
        markets_dir = Path("data/parquet/markets")
        if not markets_dir.exists():
            return pd.DataFrame()

    single = markets_dir / "markets.parquet"
    files = [single] if single.exists() else sorted(markets_dir.glob("markets_*.parquet"))
    if not files:
        return pd.DataFrame()
    
    dfs = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            dfs.append(df)
        except Exception:
            continue
    
    if not dfs:
        return pd.DataFrame()
    
    markets_df = pd.concat(dfs, ignore_index=True)
    
    # Extract question
    question_col = None
    for col in ["question", "Question", "text", "name"]:
        if col in markets_df.columns:
            question_col = col
            break
    
    if question_col:
        markets_df["question"] = markets_df[question_col]
    else:
        markets_df["question"] = ""
    
    # Ensure id column
    if "id" not in markets_df.columns:
        for col in ["market_id", "marketId", "slug"]:
            if col in markets_df.columns:
                markets_df["id"] = markets_df[col]
                break
    
    # Categorize
    markets_df["category"] = markets_df["question"].apply(
        lambda q: categorize_market(str(q) if pd.notna(q) else "")
    )
    
    return markets_df


def analyze_returns_by_category(
    markets_df: pd.DataFrame,
    price_store: PriceStore,
    categories: list,
    max_markets_per_category: int = 100,
) -> dict:
    """Calculate return distributions per category."""
    results = {}
    
    for category in categories:
        cat_markets = markets_df[markets_df["category"] == category]["id"].astype(str).head(max_markets_per_category)
        returns = []
        
        print(f"  Analyzing returns for {category} ({len(cat_markets)} markets)...")
        for market_id in cat_markets:
            try:
                prices = price_store.get_price_history(market_id, "YES")
                if len(prices) < 2:
                    continue
                
                prices = prices.sort_values("timestamp")
                prices["return"] = prices["price"].pct_change()
                returns.extend(prices["return"].dropna().tolist())
            except Exception:
                continue
        
        if returns:
            returns = np.array(returns)
            results[category] = {
                "mean": float(np.mean(returns)),
                "std": float(np.std(returns)),
                "median": float(np.median(returns)),
                "p25": float(np.percentile(returns, 25)),
                "p75": float(np.percentile(returns, 75)),
                "min": float(np.min(returns)),
                "max": float(np.max(returns)),
                "count": len(returns),
            }
            print(f"    Found {len(returns):,} returns")
        else:
            results[category] = None
            print(f"    No returns found")
    
    return results


def analyze_liquidity_over_time(
    markets_df: pd.DataFrame,
    price_store: PriceStore,
    categories: list,
    max_markets_per_category: int = 50,
) -> dict:
    """Analyze liquidity trends over time per category."""
    results = {}
    
    for category in categories:
        cat_markets = markets_df[markets_df["category"] == category]["id"].astype(str).head(max_markets_per_category)
        liquidity_by_date = defaultdict(list)
        
        print(f"  Analyzing liquidity over time for {category}...")
        for market_id in cat_markets:
            try:
                prices = price_store.get_price_history(market_id, "YES")
                if prices.empty:
                    continue
                
                prices["date"] = pd.to_datetime(prices["timestamp"], unit="s").dt.date
                # Use price volatility as proxy for liquidity (or actual liquidity if available)
                daily = prices.groupby("date").agg({
                    "price": ["mean", "std", "count"]
                })
                for date, row in daily.iterrows():
                    liquidity_by_date[date].append(row[("price", "std")] or 0)
            except Exception:
                continue
        
        if liquidity_by_date:
            dates = sorted(liquidity_by_date.keys())
            avg_liquidity = [np.mean(liquidity_by_date[d]) for d in dates]
            results[category] = {
                "dates": [str(d) for d in dates],
                "avg_liquidity": avg_liquidity,
            }
        else:
            results[category] = None
    
    return results


def analyze_trading_patterns(
    markets_df: pd.DataFrame,
    trade_store: TradeStore,
    categories: list,
    max_trades: int = 100000,
) -> dict:
    """Analyze trading patterns: users, size, frequency."""
    results = {}
    
    # Load trades
    print("Loading trades...")
    try:
        trades_df = trade_store.load_trades(limit=max_trades)
        if trades_df.empty:
            return results
    except Exception as e:
        print(f"Error loading trades: {e}")
        return results
    
    # Map market_id to category
    market_to_category = dict(zip(markets_df["id"].astype(str), markets_df["category"]))
    
    if "market_id" not in trades_df.columns:
        return results
    
    trades_df["category"] = trades_df["market_id"].astype(str).map(market_to_category)
    trades_df = trades_df[trades_df["category"].notna()]
    
    for category in categories:
        cat_trades = trades_df[trades_df["category"] == category]
        if cat_trades.empty:
            results[category] = None
            continue
        
        # Users
        user_cols = ["maker", "taker", "user_id"]
        users = set()
        for col in user_cols:
            if col in cat_trades.columns:
                users.update(cat_trades[col].dropna().astype(str))
        
        # Trading size
        size_col = None
        for col in ["usd_amount", "amount", "size"]:
            if col in cat_trades.columns:
                size_col = col
                break
        
        avg_size = None
        if size_col:
            sizes = pd.to_numeric(cat_trades[size_col], errors="coerce").dropna()
            if not sizes.empty:
                avg_size = float(sizes.mean())
        
        # Trading frequency (trades per day)
        if "timestamp" in cat_trades.columns:
            cat_trades["date"] = pd.to_datetime(cat_trades["timestamp"], errors="coerce").dt.date
            trades_per_day = cat_trades.groupby("date").size()
            avg_frequency = float(trades_per_day.mean()) if not trades_per_day.empty else None
        else:
            avg_frequency = None
        
        results[category] = {
            "unique_users": len(users),
            "total_trades": len(cat_trades),
            "avg_trade_size": avg_size,
            "avg_trades_per_day": avg_frequency,
            "markets_traded": cat_trades["market_id"].nunique(),
        }
    
    return results


def plot_returns_distribution(returns_data: dict, output_dir: Path):
    """Plot return distributions per category."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for idx, (category, data) in enumerate(sorted(returns_data.items())):
        if idx >= len(axes) or data is None:
            continue
        
        ax = axes[idx]
        # Simulate distribution from stats
        mean = data["mean"]
        std = data["std"]
        samples = np.random.normal(mean, std, 10000)
        
        ax.hist(samples, bins=50, alpha=0.7, edgecolor="black")
        ax.axvline(mean, color="red", linestyle="--", label=f"Mean: {mean:.4f}")
        ax.set_title(f"{category.upper()}\n(n={data['count']:,})")
        ax.set_xlabel("Return")
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(returns_data), len(axes)):
        axes[idx].axis("off")
    
    plt.tight_layout()
    plt.savefig(output_dir / "returns_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_liquidity_over_time(liquidity_data: dict, output_dir: Path):
    """Plot liquidity trends over time."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for category, data in sorted(liquidity_data.items()):
        if data is None:
            continue
        
        dates = pd.to_datetime(data["dates"])
        ax.plot(dates, data["avg_liquidity"], label=category, marker="o", markersize=3, alpha=0.7)
    
    ax.set_xlabel("Date")
    ax.set_ylabel("Avg Liquidity Proxy (Price Std Dev)")
    ax.set_title("Liquidity Over Time by Category")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "liquidity_over_time.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_trading_metrics(trading_data: dict, output_dir: Path):
    """Plot trading metrics comparison."""
    categories = [c for c, d in trading_data.items() if d is not None]
    
    if not categories:
        print("  No trading data available for plotting")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Users
    users = [trading_data[c]["unique_users"] for c in categories]
    axes[0, 0].bar(categories, users, color="steelblue")
    axes[0, 0].set_title("Unique Users per Category")
    axes[0, 0].set_ylabel("Users")
    axes[0, 0].tick_params(axis="x", rotation=45)
    
    # Avg trade size
    sizes = [trading_data[c].get("avg_trade_size", 0) or 0 for c in categories]
    axes[0, 1].bar(categories, sizes, color="coral")
    axes[0, 1].set_title("Average Trade Size (USD)")
    axes[0, 1].set_ylabel("USD")
    axes[0, 1].tick_params(axis="x", rotation=45)
    
    # Total trades
    trades = [trading_data[c]["total_trades"] for c in categories]
    axes[1, 0].bar(categories, trades, color="green")
    axes[1, 0].set_title("Total Trades")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].tick_params(axis="x", rotation=45)
    
    # Avg frequency
    freqs = [trading_data[c].get("avg_trades_per_day", 0) or 0 for c in categories]
    axes[1, 1].bar(categories, freqs, color="purple")
    axes[1, 1].set_title("Average Trades per Day")
    axes[1, 1].set_ylabel("Trades/Day")
    axes[1, 1].tick_params(axis="x", rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / "trading_metrics.png", dpi=150, bbox_inches="tight")
    plt.close()


def generate_report(
    markets_df: pd.DataFrame,
    returns_data: dict,
    liquidity_data: dict,
    trading_data: dict,
    output_dir: Path,
):
    """Generate markdown report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = output_dir / "category_analysis_report.md"
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Market Category Analysis Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("**Source Script**: `scripts/analyze_category_details.py`\n\n")
        f.write("**Initial Category Analysis Script**: `scripts/analyze_categories.py`\n\n")
        f.write("---\n\n")
        
        # Summary
        f.write("## Summary\n\n")
        market_counts = markets_df["category"].value_counts().sort_values(ascending=False)
        f.write("| Category | Markets | % of Total | Avg Volume | Avg Liquidity |\n")
        f.write("|----------|---------|------------|------------|---------------|\n")
        total = len(markets_df)
        for cat, count in market_counts.items():
            pct = count / total * 100
            cat_df = markets_df[markets_df["category"] == cat]
            if "volume" in cat_df.columns:
                avg_vol = pd.to_numeric(cat_df["volume"], errors="coerce").mean()
            else:
                avg_vol = 0
            if "liquidity" in cat_df.columns:
                avg_liq = pd.to_numeric(cat_df["liquidity"], errors="coerce").mean()
            else:
                avg_liq = 0
            f.write(f"| {cat} | {count:,} | {pct:.1f}% | ${avg_vol:,.0f} | ${avg_liq:,.0f} |\n")
        
        # Returns
        f.write("\n## Return Distributions\n\n")
        has_returns = any(d for d in returns_data.values() if d is not None)
        if has_returns:
            f.write("| Category | Mean | Std Dev | Median | P25 | P75 | Min | Max | Sample Size |\n")
            f.write("|----------|------|---------|--------|-----|-----|-----|-----|-------------|\n")
            for category in sorted(returns_data.keys()):
                data = returns_data[category]
                if data:
                    f.write(f"| {category} | {data['mean']:.6f} | {data['std']:.6f} | "
                           f"{data['median']:.6f} | {data['p25']:.6f} | {data['p75']:.6f} | "
                           f"{data['min']:.6f} | {data['max']:.6f} | {data['count']:,} |\n")
            f.write("\n![Returns Distribution](returns_distribution.png)\n\n")
        else:
            f.write("*No return data available.*\n\n")
        
        # Liquidity
        f.write("## Liquidity Over Time\n\n")
        has_liquidity = any(d for d in liquidity_data.values() if d is not None)
        if has_liquidity:
            f.write("![Liquidity Over Time](liquidity_over_time.png)\n\n")
        else:
            f.write("*No liquidity time series data available.*\n\n")
        
        # Trading patterns
        f.write("## Trading Patterns\n\n")
        has_trading = any(d for d in trading_data.values() if d is not None)
        if has_trading:
            f.write("| Category | Unique Users | Total Trades | Avg Trade Size (USD) | Avg Trades/Day | Markets Traded |\n")
            f.write("|----------|--------------|--------------|----------------------|----------------|----------------|\n")
            for category in sorted(trading_data.keys()):
                data = trading_data[category]
                if data:
                    f.write(f"| {category} | {data['unique_users']:,} | {data['total_trades']:,} | "
                           f"${data.get('avg_trade_size', 0) or 0:,.2f} | "
                           f"{data.get('avg_trades_per_day', 0) or 0:.1f} | "
                           f"{data['markets_traded']:,} |\n")
            
            trading_metrics_path = output_dir / "trading_metrics.png"
            if trading_metrics_path.exists():
                f.write("\n![Trading Metrics](trading_metrics.png)\n\n")
        else:
            f.write("*No trading data available.*\n\n")
        
        # Category definitions
        f.write("## Category Definitions\n\n")
        for category, keywords in sorted(CATEGORIES.items()):
            f.write(f"### {category.upper()}\n\n")
            f.write(f"Keywords: {', '.join(keywords)}\n\n")
        
        # Methodology
        f.write("## Methodology\n\n")
        f.write("This analysis uses keyword-based categorization to classify markets into categories. ")
        f.write("Returns are calculated as percentage changes in price. ")
        f.write("Liquidity is proxied by price volatility (standard deviation) over time. ")
        f.write("Trading patterns are derived from trade data including user counts, trade sizes, and frequency.\n\n")
    
    print(f"\nReport saved to: {report_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Detailed category analysis")
    parser.add_argument("--output-dir", default="docs/results/preliminary_research", help="Output directory")
    parser.add_argument("--max-markets-returns", type=int, default=100, help="Max markets per category for returns")
    parser.add_argument("--max-markets-liquidity", type=int, default=50, help="Max markets per category for liquidity")
    parser.add_argument("--max-trades", type=int, default=100000, help="Max trades to load")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading markets...")
    markets_df = load_markets()
    if markets_df.empty:
        print("No markets found!")
        return 1
    
    print(f"Loaded {len(markets_df):,} markets")
    
    categories = sorted(markets_df["category"].unique())
    print(f"Categories: {', '.join(categories)}")
    
    # Initialize stores
    price_store = PriceStore("data/polymarket/prices")
    trade_store = TradeStore("data/polymarket/trades")
    
    # Analyze returns
    print("\nAnalyzing returns...")
    returns_data = analyze_returns_by_category(
        markets_df, price_store, categories, args.max_markets_returns
    )
    
    # Analyze liquidity
    print("\nAnalyzing liquidity over time...")
    liquidity_data = analyze_liquidity_over_time(
        markets_df, price_store, categories, args.max_markets_liquidity
    )
    
    # Analyze trading patterns
    print("\nAnalyzing trading patterns...")
    trading_data = analyze_trading_patterns(
        markets_df, trade_store, categories, args.max_trades
    )
    
    # Generate plots
    print("\nGenerating plots...")
    plot_returns_distribution(returns_data, output_dir)
    plot_liquidity_over_time(liquidity_data, output_dir)
    plot_trading_metrics(trading_data, output_dir)
    
    # Generate report
    print("\nGenerating report...")
    generate_report(markets_df, returns_data, liquidity_data, trading_data, output_dir)
    
    print("\nAnalysis complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
Comprehensive holistic research analysis of Polymarket markets.

Analyzes:
- Volume distribution by category
- Liquidity patterns
- Market lifecycle and age
- Price dynamics
- Trade patterns
- Market health indicators
- Correlations between metrics
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timezone
import json
import time
from typing import Dict, List, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.trading.data_modules.database import PredictionMarketDB
from src.trading.data_modules.parquet_store import TradeStore, PriceStore
from src.taxonomy.markets import MarketTaxonomy

# Set style
plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
sns.set_palette("husl")


def load_market_data(
    db_path: str = "data/prediction_markets.db",
    use_trades: bool = True,
    parquet_trades_dir: str = "data/polymarket/trades",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load market metadata and calculate volume from trades."""
    
    print("Loading market data...")
    db = PredictionMarketDB(db_path)
    
    # Load market metadata
    markets_query = """
        SELECT 
            id,
            slug,
            question,
            volume,
            volume_24hr,
            liquidity,
            end_date,
            outcomes,
            outcome_prices,
            created_at
        FROM polymarket_markets
    """
    
    markets_df = db.query(markets_query)
    db.close()
    
    print(f"  Loaded {len(markets_df):,} markets from database")
    
    # Try to load market index JSON if database is empty
    if len(markets_df) == 0:
        print("  Database empty, checking for market_index.json...")
        market_index_path = Path("data/polymarket/market_index.json")
        if market_index_path.exists():
            print(f"  Found market index at {market_index_path}")
            try:
                import json
                with open(market_index_path, 'r') as f:
                    market_index = json.load(f)
                if market_index:
                    markets_df = pd.DataFrame(market_index)
                    markets_df["id"] = markets_df["id"].astype(str)
                    # Rename volume24hr to volume_24hr for consistency
                    if "volume24hr" in markets_df.columns:
                        markets_df["volume_24hr"] = markets_df["volume24hr"]
                    print(f"  Loaded {len(markets_df):,} markets from index file")
            except Exception as e:
                print(f"  Error loading market index: {e}")
    
    # Calculate volume from trades if requested
    if use_trades:
        print("Calculating volume from trades...")
        try:
            trade_store = TradeStore(parquet_trades_dir)
            if trade_store.available():
                print("  Loading trades from parquet...")
                trades_df = trade_store.load_trades()
                
                if not trades_df.empty and "usd_amount" in trades_df.columns:
                    print(f"  Loaded {len(trades_df):,} trades")
                    volume_df = trades_df.groupby("market_id").agg({
                        "usd_amount": ["sum", "count", "mean"],
                    })
                    volume_df.columns = ["volume_from_trades", "trade_count", "avg_trade_size"]
                    volume_df = volume_df.reset_index()
                    
                    # Merge with markets
                    markets_df = markets_df.merge(
                        volume_df,
                        left_on="id",
                        right_on="market_id",
                        how="left"
                    )
                    markets_df["volume_from_trades"] = markets_df["volume_from_trades"].fillna(0)
                    markets_df["trade_count"] = markets_df["trade_count"].fillna(0)
                    markets_df["avg_trade_size"] = markets_df["avg_trade_size"].fillna(0)
                    
                    # Use trade volume if available, otherwise use metadata volume
                    markets_df["final_volume"] = markets_df["volume_from_trades"].fillna(
                        markets_df["volume_24hr"].fillna(markets_df["volume"])
                    )
                    
                    print(f"  Calculated volume for {len(volume_df):,} markets from trades")
                    
                    # If markets_df is empty, try to fetch metadata from API
                    if len(markets_df) == 0:
                        print("  No markets in database, creating from trades...")
                        print("  NOTE: Market metadata (questions/slugs) not available in trades.")
                        print("  To get proper categorization, either:")
                        print("    1. Fetch price data (creates market_index.json with questions)")
                        print("    2. Populate database with market metadata")
                        print("    3. Markets will be marked as 'uncategorized'")
                        
                        # Create markets from volume_df (no metadata available)
                        markets_df = volume_df.copy()
                        markets_df["id"] = markets_df["market_id"]
                        markets_df["question"] = "Market " + markets_df["market_id"].astype(str)
                        markets_df["slug"] = None
                        markets_df["volume"] = 0
                        markets_df["volume_24hr"] = 0
                        markets_df["liquidity"] = 0
                        markets_df["end_date"] = None
                        markets_df["outcomes"] = None
                        markets_df["outcome_prices"] = None
                        markets_df["created_at"] = None
                        
                        # Set final_volume from trades
                        if "final_volume" not in markets_df.columns:
                            markets_df["final_volume"] = markets_df["volume_from_trades"]
                    else:
                        # Merge with existing markets
                        markets_df = markets_df.merge(
                            volume_df,
                            left_on="id",
                            right_on="market_id",
                            how="left"
                        )
                        markets_df["volume_from_trades"] = markets_df["volume_from_trades"].fillna(0)
                        markets_df["trade_count"] = markets_df["trade_count"].fillna(0)
                        markets_df["avg_trade_size"] = markets_df["avg_trade_size"].fillna(0)
                        markets_df["final_volume"] = markets_df["volume_from_trades"].fillna(
                            markets_df["volume_24hr"].fillna(markets_df["volume"])
                        )
                else:
                    # No usd_amount column or empty trades
                    if len(markets_df) > 0:
                        markets_df["final_volume"] = markets_df["volume_24hr"].fillna(markets_df["volume"])
                    markets_df["volume_from_trades"] = 0
                    markets_df["trade_count"] = 0
                    markets_df["avg_trade_size"] = 0
            else:
                # Trade store not available
                if len(markets_df) > 0:
                    markets_df["final_volume"] = markets_df["volume_24hr"].fillna(markets_df["volume"])
                markets_df["volume_from_trades"] = 0
                markets_df["trade_count"] = 0
                markets_df["avg_trade_size"] = 0
        except Exception as e:
            print(f"  Error loading trades: {e}")
            import traceback
            traceback.print_exc()
            if len(markets_df) > 0:
                markets_df["final_volume"] = markets_df["volume_24hr"].fillna(markets_df["volume"])
            markets_df["volume_from_trades"] = 0
            markets_df["trade_count"] = 0
            markets_df["avg_trade_size"] = 0
    else:
        if len(markets_df) > 0:
            markets_df["final_volume"] = markets_df["volume_24hr"].fillna(markets_df["volume"])
        markets_df["volume_from_trades"] = 0
        markets_df["trade_count"] = 0
        markets_df["avg_trade_size"] = 0
    
    # Parse JSON columns
    if "outcomes" in markets_df.columns:
        markets_df["outcomes"] = markets_df["outcomes"].apply(
            lambda x: json.loads(x) if isinstance(x, str) else (x if isinstance(x, list) else [])
        )
    if "outcome_prices" in markets_df.columns:
        markets_df["outcome_prices"] = markets_df["outcome_prices"].apply(
            lambda x: json.loads(x) if isinstance(x, str) else (x if isinstance(x, list) else [])
        )
    
    # Calculate market age
    now = pd.Timestamp.now()
    if "created_at" in markets_df.columns:
        markets_df["created_at"] = pd.to_datetime(markets_df["created_at"], errors='coerce')
        # Make both timezone-naive for comparison
        markets_df["created_at"] = markets_df["created_at"].dt.tz_localize(None) if markets_df["created_at"].dt.tz else markets_df["created_at"]
        markets_df["market_age_days"] = (now - markets_df["created_at"]).dt.total_seconds() / 86400
    else:
        markets_df["market_age_days"] = np.nan
    
    # Parse end dates
    if "end_date" in markets_df.columns:
        markets_df["end_date_parsed"] = pd.to_datetime(markets_df["end_date"], errors='coerce')
        # Make timezone-naive
        markets_df["end_date_parsed"] = markets_df["end_date_parsed"].dt.tz_localize(None) if markets_df["end_date_parsed"].dt.tz else markets_df["end_date_parsed"]
        markets_df["days_until_end"] = (markets_df["end_date_parsed"] - now).dt.total_seconds() / 86400
        markets_df["is_closed"] = markets_df["days_until_end"] < 0
    else:
        markets_df["end_date_parsed"] = pd.NaT
        markets_df["days_until_end"] = np.nan
        markets_df["is_closed"] = False
    
    # Calculate volume velocity (volume per day)
    markets_df["volume_velocity"] = markets_df["final_volume"] / markets_df["market_age_days"].clip(lower=1)
    
    # Calculate volume-to-liquidity ratio
    markets_df["volume_liquidity_ratio"] = markets_df["final_volume"] / markets_df["liquidity"].replace(0, np.nan)
    
    # Ensure final_volume exists
    if "final_volume" not in markets_df.columns:
        markets_df["final_volume"] = markets_df.get("volume_24hr", markets_df.get("volume", 0))
    
    # Filter out markets with no volume
    markets_df = markets_df[markets_df["final_volume"] > 0].copy()
    
    print(f"  Final dataset: {len(markets_df):,} markets with volume > 0")
    
    if len(markets_df) == 0:
        print("  WARNING: No markets with volume found!")
        return pd.DataFrame(), pd.DataFrame()
    
    return markets_df, trades_df if use_trades and 'trades_df' in locals() else pd.DataFrame()


def categorize_markets(markets_df: pd.DataFrame) -> pd.DataFrame:
    """Categorize markets using taxonomy."""
    
    print("\nCategorizing markets...")
    
    taxonomy = MarketTaxonomy()
    
    categories = []
    skipped = 0
    for idx, row in markets_df.iterrows():
        question = str(row.get("question", ""))
        market_id = str(row.get("id", ""))
        
        # Skip markets with generic/question-less questions
        if not question or question.startswith("Market ") or len(question) < 10:
            categories.append("uncategorized")
            skipped += 1
            continue
        
        result = taxonomy.classify(market_id=market_id, question=question)
        categories.append(result.primary_category)
    
    markets_df["category"] = categories
    
    category_counts = markets_df["category"].value_counts()
    print(f"  Categories found: {len(category_counts)}")
    print(f"  Markets without questions (uncategorized): {skipped:,}")
    for cat, count in category_counts.head(10).items():
        print(f"    {cat}: {count:,} markets")
    
    return markets_df


def analyze_by_category(markets_df: pd.DataFrame) -> pd.DataFrame:
    """Analyze metrics by category."""
    
    print("\nAnalyzing by category...")
    
    category_stats = markets_df.groupby("category").agg({
        "id": "count",
        "final_volume": ["sum", "mean", "median"],
        "liquidity": ["mean", "median"],
        "volume_velocity": ["mean", "median"],
        "volume_liquidity_ratio": ["mean", "median"],
        "trade_count": ["sum", "mean"],
        "avg_trade_size": ["mean", "median"],
        "market_age_days": ["mean", "median"],
        "is_closed": "sum",
    })
    
    category_stats.columns = [
        "market_count", "total_volume", "avg_volume", "median_volume",
        "avg_liquidity", "median_liquidity",
        "avg_velocity", "median_velocity",
        "avg_vl_ratio", "median_vl_ratio",
        "total_trades", "avg_trades_per_market",
        "avg_trade_size", "median_trade_size",
        "avg_age_days", "median_age_days",
        "closed_count"
    ]
    
    category_stats = category_stats.sort_values("total_volume", ascending=False)
    category_stats["pct_of_total_volume"] = (category_stats["total_volume"] / category_stats["total_volume"].sum()) * 100
    category_stats["pct_of_markets"] = (category_stats["market_count"] / category_stats["market_count"].sum()) * 100
    
    print(f"  Top categories by volume:")
    for idx, row in category_stats.head(10).iterrows():
        print(f"    {idx}: {row['total_volume']:,.0f} ({row['pct_of_total_volume']:.1f}%)")
    
    return category_stats


def analyze_liquidity_patterns(markets_df: pd.DataFrame) -> Dict:
    """Analyze liquidity patterns."""
    
    print("\nAnalyzing liquidity patterns...")
    
    # Filter markets with liquidity data
    liq_df = markets_df[markets_df["liquidity"] > 0].copy()
    
    if len(liq_df) == 0:
        return {}
    
    stats = {
        "total_markets_with_liquidity": len(liq_df),
        "pct_with_liquidity": (len(liq_df) / len(markets_df)) * 100,
        "mean_liquidity": liq_df["liquidity"].mean(),
        "median_liquidity": liq_df["liquidity"].median(),
        "liquidity_percentiles": {
            p: np.percentile(liq_df["liquidity"], p)
            for p in [10, 25, 50, 75, 90, 95, 99]
        },
        "volume_liquidity_correlation": liq_df["final_volume"].corr(liq_df["liquidity"]),
        "mean_vl_ratio": liq_df["volume_liquidity_ratio"].mean(),
        "median_vl_ratio": liq_df["volume_liquidity_ratio"].median(),
    }
    
    print(f"  Markets with liquidity: {stats['total_markets_with_liquidity']:,} ({stats['pct_with_liquidity']:.1f}%)")
    print(f"  Mean liquidity: ${stats['mean_liquidity']:,.2f}")
    print(f"  Volume-Liquidity correlation: {stats['volume_liquidity_correlation']:.3f}")
    
    return stats


def analyze_market_lifecycle(markets_df: pd.DataFrame) -> Dict:
    """Analyze market lifecycle patterns."""
    
    print("\nAnalyzing market lifecycle...")
    
    age_df = markets_df[markets_df["market_age_days"].notna()].copy()
    
    if len(age_df) == 0:
        return {}
    
    # Age buckets
    age_buckets = [
        (0, 7, "0-7 days"),
        (7, 30, "7-30 days"),
        (30, 90, "30-90 days"),
        (90, 180, "90-180 days"),
        (180, 365, "180-365 days"),
        (365, float('inf'), "1+ years"),
    ]
    
    bucket_stats = []
    for min_age, max_age, label in age_buckets:
        mask = (age_df["market_age_days"] >= min_age) & (age_df["market_age_days"] < max_age)
        bucket_markets = age_df[mask]
        if len(bucket_markets) > 0:
            bucket_stats.append({
                "age_range": label,
                "market_count": len(bucket_markets),
                "avg_volume": bucket_markets["final_volume"].mean(),
                "median_volume": bucket_markets["final_volume"].median(),
                "avg_velocity": bucket_markets["volume_velocity"].mean(),
            })
    
    stats = {
        "mean_age_days": age_df["market_age_days"].mean(),
        "median_age_days": age_df["market_age_days"].median(),
        "age_buckets": bucket_stats,
        "closed_markets_pct": (markets_df["is_closed"].sum() / len(markets_df)) * 100 if "is_closed" in markets_df.columns else 0,
        "age_volume_correlation": age_df["market_age_days"].corr(age_df["final_volume"]),
    }
    
    print(f"  Mean market age: {stats['mean_age_days']:.1f} days")
    print(f"  Closed markets: {stats['closed_markets_pct']:.1f}%")
    print(f"  Age-Volume correlation: {stats['age_volume_correlation']:.3f}")
    
    return stats


def create_comprehensive_visualizations(
    markets_df: pd.DataFrame,
    category_stats: pd.DataFrame,
    liquidity_stats: Dict,
    lifecycle_stats: Dict,
    output_dir: Path,
):
    """Create comprehensive visualization suite."""
    
    print("\nGenerating visualizations...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Volume Distribution by Category
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Market Analysis: Volume by Category', fontsize=16, fontweight='bold')
    
    # Top left: Total volume by category
    ax1 = axes[0, 0]
    top_cats = category_stats.head(10)
    colors = plt.cm.viridis(np.linspace(0, 1, len(top_cats)))
    bars = ax1.barh(range(len(top_cats)), top_cats["total_volume"] / 1e9, color=colors)
    ax1.set_yticks(range(len(top_cats)))
    ax1.set_yticklabels(top_cats.index, fontsize=10)
    ax1.set_xlabel('Total Volume (Billions USD)', fontsize=11, fontweight='bold')
    ax1.set_title('Total Volume by Category', fontsize=12, fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3, axis='x')
    for i, (idx, row) in enumerate(top_cats.iterrows()):
        ax1.text(row["total_volume"] / 1e9, i, f'${row["total_volume"]/1e9:.2f}B',
                va='center', ha='left', fontsize=9)
    
    # Top right: Market count by category
    ax2 = axes[0, 1]
    bars = ax2.barh(range(len(top_cats)), top_cats["market_count"], color=colors)
    ax2.set_yticks(range(len(top_cats)))
    ax2.set_yticklabels(top_cats.index, fontsize=10)
    ax2.set_xlabel('Number of Markets', fontsize=11, fontweight='bold')
    ax2.set_title('Market Count by Category', fontsize=12, fontweight='bold')
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.3, axis='x')
    for i, (idx, row) in enumerate(top_cats.iterrows()):
        ax2.text(row["market_count"], i, f'{int(row["market_count"]):,}',
                va='center', ha='left', fontsize=9)
    
    # Bottom left: Average volume per market
    ax3 = axes[1, 0]
    bars = ax3.barh(range(len(top_cats)), top_cats["avg_volume"] / 1e6, color=colors)
    ax3.set_yticks(range(len(top_cats)))
    ax3.set_yticklabels(top_cats.index, fontsize=10)
    ax3.set_xlabel('Average Volume per Market (Millions USD)', fontsize=11, fontweight='bold')
    ax3.set_title('Average Volume by Category', fontsize=12, fontweight='bold')
    ax3.invert_yaxis()
    ax3.grid(True, alpha=0.3, axis='x')
    
    # Bottom right: Volume share pie chart
    ax4 = axes[1, 1]
    top_5_cats = category_stats.head(5)
    other_volume = category_stats.iloc[5:]["total_volume"].sum()
    pie_data = list(top_5_cats["total_volume"]) + [other_volume]
    pie_labels = list(top_5_cats.index) + ["Other"]
    ax4.pie(pie_data, labels=pie_labels, autopct='%1.1f%%', startangle=90)
    ax4.set_title('Volume Share by Category', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'volume_by_category.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: volume_by_category.png")
    
    # 2. Liquidity Analysis
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Liquidity Analysis', fontsize=16, fontweight='bold')
    
    liq_df = markets_df[markets_df["liquidity"] > 0].copy()
    
    if len(liq_df) > 0:
        # Liquidity distribution
        ax1 = axes[0, 0]
        liq_log = np.log10(liq_df["liquidity"][liq_df["liquidity"] > 0])
        ax1.hist(liq_log, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        ax1.set_xlabel('Liquidity (log10 USD)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Number of Markets', fontsize=11, fontweight='bold')
        ax1.set_title('Liquidity Distribution (Log Scale)', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Volume vs Liquidity scatter
        ax2 = axes[0, 1]
        sample = liq_df.sample(min(5000, len(liq_df)))
        ax2.scatter(np.log10(sample["liquidity"]), np.log10(sample["final_volume"]),
                   alpha=0.3, s=10)
        ax2.set_xlabel('Liquidity (log10 USD)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Volume (log10 USD)', fontsize=11, fontweight='bold')
        ax2.set_title('Volume vs Liquidity', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Volume-to-Liquidity ratio
        ax3 = axes[1, 0]
        vl_ratio = liq_df["volume_liquidity_ratio"].dropna()
        if len(vl_ratio) > 0:
            vl_ratio_log = np.log10(vl_ratio[vl_ratio > 0])
            ax3.hist(vl_ratio_log, bins=50, edgecolor='black', alpha=0.7, color='green')
            ax3.set_xlabel('Volume/Liquidity Ratio (log10)', fontsize=11, fontweight='bold')
            ax3.set_ylabel('Number of Markets', fontsize=11, fontweight='bold')
            ax3.set_title('Volume-to-Liquidity Ratio Distribution', fontsize=12, fontweight='bold')
            ax3.grid(True, alpha=0.3)
        
        # Liquidity by category
        ax4 = axes[1, 1]
        liq_by_cat = markets_df.groupby("category")["liquidity"].mean().sort_values(ascending=False).head(10)
        bars = ax4.barh(range(len(liq_by_cat)), liq_by_cat / 1e6, color=plt.cm.viridis(np.linspace(0, 1, len(liq_by_cat))))
        ax4.set_yticks(range(len(liq_by_cat)))
        ax4.set_yticklabels(liq_by_cat.index, fontsize=10)
        ax4.set_xlabel('Average Liquidity (Millions USD)', fontsize=11, fontweight='bold')
        ax4.set_title('Average Liquidity by Category', fontsize=12, fontweight='bold')
        ax4.invert_yaxis()
        ax4.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'liquidity_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: liquidity_analysis.png")
    
    # 3. Market Lifecycle Analysis
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Market Lifecycle Analysis', fontsize=16, fontweight='bold')
    
    age_df = markets_df[markets_df["market_age_days"].notna()].copy()
    
    if len(age_df) > 0:
        # Age distribution
        ax1 = axes[0, 0]
        age_log = np.log10(age_df["market_age_days"].clip(lower=1))
        ax1.hist(age_log, bins=50, edgecolor='black', alpha=0.7, color='orange')
        ax1.set_xlabel('Market Age (log10 days)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Number of Markets', fontsize=11, fontweight='bold')
        ax1.set_title('Market Age Distribution', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Volume velocity by age
        ax2 = axes[0, 1]
        sample = age_df.sample(min(5000, len(age_df)))
        ax2.scatter(np.log10(sample["market_age_days"].clip(lower=1)),
                   np.log10(sample["volume_velocity"].clip(lower=1)),
                   alpha=0.3, s=10)
        ax2.set_xlabel('Market Age (log10 days)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Volume Velocity (log10 USD/day)', fontsize=11, fontweight='bold')
        ax2.set_title('Volume Velocity vs Market Age', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Volume by age bucket
        if lifecycle_stats.get("age_buckets"):
            ax3 = axes[1, 0]
            buckets_df = pd.DataFrame(lifecycle_stats["age_buckets"])
            bars = ax3.bar(range(len(buckets_df)), buckets_df["avg_volume"] / 1e6,
                          color=plt.cm.plasma(np.linspace(0, 0.9, len(buckets_df))))
            ax3.set_xticks(range(len(buckets_df)))
            ax3.set_xticklabels(buckets_df["age_range"], rotation=45, ha='right', fontsize=9)
            ax3.set_ylabel('Average Volume (Millions USD)', fontsize=11, fontweight='bold')
            ax3.set_title('Average Volume by Market Age', fontsize=12, fontweight='bold')
            ax3.grid(True, alpha=0.3, axis='y')
        
        # Market count by age bucket
        if lifecycle_stats.get("age_buckets"):
            ax4 = axes[1, 1]
            bars = ax4.bar(range(len(buckets_df)), buckets_df["market_count"],
                          color=plt.cm.plasma(np.linspace(0, 0.9, len(buckets_df))))
            ax4.set_xticks(range(len(buckets_df)))
            ax4.set_xticklabels(buckets_df["age_range"], rotation=45, ha='right', fontsize=9)
            ax4.set_ylabel('Number of Markets', fontsize=11, fontweight='bold')
            ax4.set_title('Market Count by Age', fontsize=12, fontweight='bold')
            ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'market_lifecycle.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: market_lifecycle.png")
    
    # 4. Correlation Matrix
    fig, ax = plt.subplots(figsize=(12, 10))
    
    numeric_cols = ["final_volume", "liquidity", "volume_velocity", "volume_liquidity_ratio",
                   "trade_count", "avg_trade_size", "market_age_days"]
    corr_df = markets_df[numeric_cols].corr()
    
    mask = np.triu(np.ones_like(corr_df, dtype=bool))
    sns.heatmap(corr_df, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title('Correlation Matrix: Market Metrics', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: correlation_matrix.png")
    
    # 5. Comprehensive Overview Dashboard
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    fig.suptitle('Polymarket Holistic Research Dashboard', fontsize=18, fontweight='bold', y=0.98)
    
    # Volume distribution
    ax1 = fig.add_subplot(gs[0, 0])
    volumes_log = np.log10(markets_df["final_volume"][markets_df["final_volume"] > 0])
    ax1.hist(volumes_log, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.set_xlabel('Volume (log10 USD)', fontsize=9)
    ax1.set_ylabel('Markets', fontsize=9)
    ax1.set_title('Volume Distribution', fontsize=10, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Category volume share
    ax2 = fig.add_subplot(gs[0, 1])
    top_5 = category_stats.head(5)
    ax2.pie(top_5["total_volume"], labels=top_5.index, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Top 5 Categories by Volume', fontsize=10, fontweight='bold')
    
    # Liquidity distribution
    ax3 = fig.add_subplot(gs[0, 2])
    if len(liq_df) > 0:
        liq_log = np.log10(liq_df["liquidity"][liq_df["liquidity"] > 0])
        ax3.hist(liq_log, bins=50, edgecolor='black', alpha=0.7, color='green')
        ax3.set_xlabel('Liquidity (log10 USD)', fontsize=9)
        ax3.set_ylabel('Markets', fontsize=9)
        ax3.set_title('Liquidity Distribution', fontsize=10, fontweight='bold')
        ax3.grid(True, alpha=0.3)
    
    # Age distribution
    ax4 = fig.add_subplot(gs[0, 3])
    if len(age_df) > 0:
        age_log = np.log10(age_df["market_age_days"].clip(lower=1))
        ax4.hist(age_log, bins=50, edgecolor='black', alpha=0.7, color='orange')
        ax4.set_xlabel('Age (log10 days)', fontsize=9)
        ax4.set_ylabel('Markets', fontsize=9)
        ax4.set_title('Market Age Distribution', fontsize=10, fontweight='bold')
        ax4.grid(True, alpha=0.3)
    
    # Volume vs Liquidity
    ax5 = fig.add_subplot(gs[1, 0])
    if len(liq_df) > 0:
        sample = liq_df.sample(min(2000, len(liq_df)))
        ax5.scatter(np.log10(sample["liquidity"]), np.log10(sample["final_volume"]),
                   alpha=0.2, s=5)
        ax5.set_xlabel('Liquidity (log10)', fontsize=9)
        ax5.set_ylabel('Volume (log10)', fontsize=9)
        ax5.set_title('Volume vs Liquidity', fontsize=10, fontweight='bold')
        ax5.grid(True, alpha=0.3)
    
    # Volume velocity vs age
    ax6 = fig.add_subplot(gs[1, 1])
    if len(age_df) > 0:
        sample = age_df.sample(min(2000, len(age_df)))
        ax6.scatter(np.log10(sample["market_age_days"].clip(lower=1)),
                   np.log10(sample["volume_velocity"].clip(lower=1)),
                   alpha=0.2, s=5)
        ax6.set_xlabel('Age (log10 days)', fontsize=9)
        ax6.set_ylabel('Velocity (log10 USD/day)', fontsize=9)
        ax6.set_title('Volume Velocity vs Age', fontsize=10, fontweight='bold')
        ax6.grid(True, alpha=0.3)
    
    # Top categories by market count
    ax7 = fig.add_subplot(gs[1, 2])
    top_10_cats = category_stats.head(10)
    bars = ax7.barh(range(len(top_10_cats)), top_10_cats["market_count"],
                   color=plt.cm.viridis(np.linspace(0, 1, len(top_10_cats))))
    ax7.set_yticks(range(len(top_10_cats)))
    ax7.set_yticklabels(top_10_cats.index, fontsize=8)
    ax7.set_xlabel('Market Count', fontsize=9)
    ax7.set_title('Top Categories by Count', fontsize=10, fontweight='bold')
    ax7.invert_yaxis()
    ax7.grid(True, alpha=0.3, axis='x')
    
    # Volume by category (avg)
    ax8 = fig.add_subplot(gs[1, 3])
    bars = ax8.barh(range(len(top_10_cats)), top_10_cats["avg_volume"] / 1e6,
                   color=plt.cm.viridis(np.linspace(0, 1, len(top_10_cats))))
    ax8.set_yticks(range(len(top_10_cats)))
    ax8.set_yticklabels(top_10_cats.index, fontsize=8)
    ax8.set_xlabel('Avg Volume (M USD)', fontsize=9)
    ax8.set_title('Avg Volume by Category', fontsize=10, fontweight='bold')
    ax8.invert_yaxis()
    ax8.grid(True, alpha=0.3, axis='x')
    
    # Key statistics text
    ax9 = fig.add_subplot(gs[2, :2])
    ax9.axis('off')
    stats_text = f"""
    KEY STATISTICS
    {'='*60}
    Total Markets: {len(markets_df):,}
    Total Volume: ${markets_df['final_volume'].sum()/1e9:.2f}B
    Mean Volume: ${markets_df['final_volume'].mean():,.0f}
    Median Volume: ${markets_df['final_volume'].median():,.0f}
    
    Markets with Liquidity: {liquidity_stats.get('total_markets_with_liquidity', 0):,}
    Mean Liquidity: ${liquidity_stats.get('mean_liquidity', 0):,.0f}
    Volume-Liquidity Correlation: {liquidity_stats.get('volume_liquidity_correlation', 0):.3f}
    
    Mean Market Age: {lifecycle_stats.get('mean_age_days', 0):.1f} days
    Closed Markets: {lifecycle_stats.get('closed_markets_pct', 0):.1f}%
    Age-Volume Correlation: {lifecycle_stats.get('age_volume_correlation', 0):.3f}
    
    Top Category: {category_stats.index[0] if len(category_stats) > 0 else 'N/A'}
    Top Category Volume Share: {category_stats.iloc[0]['pct_of_total_volume']:.1f}%
    """
    ax9.text(0.05, 0.95, stats_text, transform=ax9.transAxes,
            fontsize=11, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Top markets table
    ax10 = fig.add_subplot(gs[2, 2:])
    ax10.axis('off')
    top_markets = markets_df.nlargest(10, "final_volume")[["question", "final_volume", "category"]].copy()
    top_markets["volume_b"] = top_markets["final_volume"] / 1e9
    top_markets["question_short"] = top_markets["question"].str[:50] + "..."
    table_data = []
    for idx, row in top_markets.iterrows():
        table_data.append([row["question_short"], f"${row['volume_b']:.2f}B", row["category"]])
    
    table = ax10.table(cellText=table_data,
                       colLabels=["Market", "Volume", "Category"],
                       cellLoc='left',
                       loc='center',
                       colWidths=[0.6, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)
    ax10.set_title('Top 10 Markets by Volume', fontsize=10, fontweight='bold', pad=20)
    
    plt.savefig(output_dir / 'comprehensive_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: comprehensive_dashboard.png")


def generate_report(
    markets_df: pd.DataFrame,
    category_stats: pd.DataFrame,
    liquidity_stats: Dict,
    lifecycle_stats: Dict,
    output_dir: Path,
):
    """Generate comprehensive research report."""
    
    print("\nGenerating research report...")
    
    report_path = output_dir / "holistic_research_report.md"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Polymarket Holistic Research Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        
        # Executive Summary
        f.write("## Executive Summary\n\n")
        f.write(f"This report provides a comprehensive analysis of {len(markets_df):,} Polymarket markets, ")
        f.write(f"covering volume distribution, category analysis, liquidity patterns, and market lifecycle dynamics.\n\n")
        
        f.write(f"- **Total Markets Analyzed**: {len(markets_df):,}\n")
        f.write(f"- **Total Volume**: ${markets_df['final_volume'].sum()/1e9:.2f}B\n")
        f.write(f"- **Mean Volume**: ${markets_df['final_volume'].mean():,.0f}\n")
        f.write(f"- **Median Volume**: ${markets_df['final_volume'].median():,.0f}\n")
        f.write(f"- **Categories Identified**: {len(category_stats)}\n\n")
        
        # Volume Analysis
        f.write("## Volume Analysis\n\n")
        f.write("### Overall Distribution\n\n")
        f.write(f"- **Total Volume**: ${markets_df['final_volume'].sum()/1e9:.2f}B\n")
        f.write(f"- **Mean Volume**: ${markets_df['final_volume'].mean():,.0f}\n")
        f.write(f"- **Median Volume**: ${markets_df['final_volume'].median():,.0f}\n")
        f.write(f"- **Std Deviation**: ${markets_df['final_volume'].std():,.0f}\n")
        f.write(f"- **Min Volume**: ${markets_df['final_volume'].min():,.2f}\n")
        f.write(f"- **Max Volume**: ${markets_df['final_volume'].max():,.2f}\n\n")
        
        f.write("### Volume Percentiles\n\n")
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        for p in percentiles:
            val = np.percentile(markets_df["final_volume"], p)
            f.write(f"- **{p}th percentile**: ${val:,.2f}\n")
        f.write("\n")
        
        # Category Analysis
        f.write("## Category Analysis\n\n")
        f.write("### Top Categories by Volume\n\n")
        f.write("| Category | Markets | Total Volume | Avg Volume | % of Total |\n")
        f.write("|----------|---------|--------------|------------|------------|\n")
        for idx, row in category_stats.head(15).iterrows():
            f.write(f"| {idx} | {int(row['market_count']):,} | ${row['total_volume']/1e6:.1f}M | ")
            f.write(f"${row['avg_volume']:,.0f} | {row['pct_of_total_volume']:.1f}% |\n")
        f.write("\n")
        
        # Liquidity Analysis
        f.write("## Liquidity Analysis\n\n")
        if liquidity_stats:
            f.write(f"- **Markets with Liquidity Data**: {liquidity_stats.get('total_markets_with_liquidity', 0):,} ")
            f.write(f"({liquidity_stats.get('pct_with_liquidity', 0):.1f}%)\n")
            f.write(f"- **Mean Liquidity**: ${liquidity_stats.get('mean_liquidity', 0):,.2f}\n")
            f.write(f"- **Median Liquidity**: ${liquidity_stats.get('median_liquidity', 0):,.2f}\n")
            f.write(f"- **Volume-Liquidity Correlation**: {liquidity_stats.get('volume_liquidity_correlation', 0):.3f}\n")
            f.write(f"- **Mean Volume/Liquidity Ratio**: {liquidity_stats.get('mean_vl_ratio', 0):.2f}\n\n")
        
        # Lifecycle Analysis
        f.write("## Market Lifecycle Analysis\n\n")
        if lifecycle_stats:
            f.write(f"- **Mean Market Age**: {lifecycle_stats.get('mean_age_days', 0):.1f} days\n")
            f.write(f"- **Median Market Age**: {lifecycle_stats.get('median_age_days', 0):.1f} days\n")
            f.write(f"- **Closed Markets**: {lifecycle_stats.get('closed_markets_pct', 0):.1f}%\n")
            f.write(f"- **Age-Volume Correlation**: {lifecycle_stats.get('age_volume_correlation', 0):.3f}\n\n")
            
            if lifecycle_stats.get("age_buckets"):
                f.write("### Volume by Market Age\n\n")
                f.write("| Age Range | Markets | Avg Volume | Median Volume | Avg Velocity |\n")
                f.write("|-----------|---------|------------|---------------|--------------|\n")
                for bucket in lifecycle_stats["age_buckets"]:
                    f.write(f"| {bucket['age_range']} | {bucket['market_count']:,} | ")
                    f.write(f"${bucket['avg_volume']:,.0f} | ${bucket['median_volume']:,.0f} | ")
                    f.write(f"${bucket['avg_velocity']:,.0f}/day |\n")
                f.write("\n")
        
        # Key Findings
        f.write("## Key Findings\n\n")
        f.write("1. **Volume Concentration**: The market shows high concentration with a small number of markets ")
        f.write("accounting for the majority of volume.\n\n")
        
        f.write("2. **Category Dominance**: ")
        if len(category_stats) > 0:
            top_cat = category_stats.index[0]
            top_pct = category_stats.iloc[0]['pct_of_total_volume']
            f.write(f"{top_cat} dominates with {top_pct:.1f}% of total volume.\n\n")
        
        f.write("3. **Liquidity Patterns**: ")
        if liquidity_stats.get('volume_liquidity_correlation', 0) > 0.5:
            f.write("Strong positive correlation between volume and liquidity indicates active markets ")
            f.write("tend to have better liquidity.\n\n")
        else:
            f.write("Volume and liquidity show moderate correlation, suggesting other factors influence liquidity.\n\n")
        
        f.write("4. **Market Lifecycle**: ")
        if lifecycle_stats.get('age_volume_correlation', 0) > 0:
            f.write("Older markets tend to have higher volume, suggesting accumulation over time.\n\n")
        else:
            f.write("Market age shows limited correlation with volume, suggesting volume is driven by other factors.\n\n")
        
        # Recommendations
        f.write("## Recommendations\n\n")
        f.write("1. **Focus on High-Volume Categories**: Concentrate research and trading efforts on categories ")
        f.write("with consistently high volume.\n\n")
        
        f.write("2. **Liquidity Considerations**: When trading, prioritize markets with adequate liquidity ")
        f.write("to minimize execution costs.\n\n")
        
        f.write("3. **Market Selection**: Consider market age and volume velocity when selecting markets ")
        f.write("for trading strategies.\n\n")
        
        f.write("---\n\n")
        f.write("*Report generated by holistic_market_research.py*\n")
    
    print(f"  Saved: holistic_research_report.md")


def main():
    """Main analysis function."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive holistic market research")
    parser.add_argument("--db-path", default="data/prediction_markets.db", help="Path to database")
    parser.add_argument("--use-trades", action="store_true", help="Calculate volume from trades")
    parser.add_argument("--parquet-trades-dir", default="data/polymarket/trades", help="Parquet trades directory")
    parser.add_argument("--output-dir", default="docs/results/holistic_research", help="Output directory")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    markets_df, trades_df = load_market_data(
        db_path=args.db_path,
        use_trades=args.use_trades,
        parquet_trades_dir=args.parquet_trades_dir,
    )
    
    if len(markets_df) == 0:
        print("\nERROR: No markets found. Cannot proceed with analysis.")
        return
    
    # Categorize markets
    markets_df = categorize_markets(markets_df)
    
    # Analyze by category
    category_stats = analyze_by_category(markets_df)
    
    # Analyze liquidity
    liquidity_stats = analyze_liquidity_patterns(markets_df)
    
    # Analyze lifecycle
    lifecycle_stats = analyze_market_lifecycle(markets_df)
    
    # Create visualizations
    try:
        create_comprehensive_visualizations(
            markets_df, category_stats, liquidity_stats, lifecycle_stats, output_dir
        )
    except Exception as e:
        print(f"  Error creating visualizations: {e}")
        import traceback
        traceback.print_exc()
    
    # Generate report
    try:
        generate_report(
            markets_df, category_stats, liquidity_stats, lifecycle_stats, output_dir
        )
    except Exception as e:
        print(f"  Error generating report: {e}")
        import traceback
        traceback.print_exc()
    
    # Save data
    try:
        markets_df.to_csv(output_dir / "markets_data.csv", index=False)
        category_stats.to_csv(output_dir / "category_stats.csv")
    except Exception as e:
        print(f"  Error saving data: {e}")
    
    print(f"\n{'='*80}")
    print(f"Analysis complete! Results saved to: {output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

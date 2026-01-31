"""
Whale Following Strategy Backtest for Prediction Markets
Simulates following whale traders on Polymarket with realistic trading costs
"""

import json
import glob
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("WHALE FOLLOWING STRATEGY BACKTEST")
print("=" * 60)

# =============================================================================
# 1. LOAD DATA
# =============================================================================
print("\n[1] Loading Manifold bets data...")

bets = []
bet_files = sorted(glob.glob('data/manifold/bets_*.json'))
print(f"    Found {len(bet_files)} bet files")

for i, f in enumerate(bet_files):
    try:
        with open(f, encoding='utf-8') as file:
            batch = json.load(file)
            bets.extend(batch)
    except Exception as e:
        pass
    if (i + 1) % 500 == 0:
        print(f"    Loaded {i+1}/{len(bet_files)} files ({len(bets):,} bets)")

print(f"    Total bets loaded: {len(bets):,}")

# Convert to DataFrame
df = pd.DataFrame(bets)
df['datetime'] = pd.to_datetime(df['createdTime'], unit='ms')
df['date'] = df['datetime'].dt.date
df['amount_abs'] = df['amount'].abs()

print(f"    Date range: {df['datetime'].min()} to {df['datetime'].max()}")
print(f"    Unique users: {df['userId'].nunique():,}")
print(f"    Unique markets: {df['contractId'].nunique():,}")

# =============================================================================
# 2. LOAD MARKETS FOR RESOLUTION DATA
# =============================================================================
print("\n[2] Loading markets data...")

markets = []
market_files = sorted(glob.glob('data/manifold/markets_*.json'))
for f in market_files:
    try:
        with open(f, encoding='utf-8') as file:
            markets.extend(json.load(file))
    except:
        pass

markets_df = pd.DataFrame(markets)
print(f"    Total markets: {len(markets_df):,}")

# Get resolved markets
resolved = markets_df[markets_df['isResolved'] == True].copy()
print(f"    Resolved markets: {len(resolved):,}")

# Create resolution lookup
resolution_map = {}
for _, m in resolved.iterrows():
    mid = m['id']
    resolution = m.get('resolution')
    if resolution in ['YES', 'NO']:
        resolution_map[mid] = 1.0 if resolution == 'YES' else 0.0

print(f"    Markets with YES/NO resolution: {len(resolution_map):,}")

# =============================================================================
# 3. POLYMARKET TRADING COSTS SIMULATION
# =============================================================================
print("\n[3] Setting up Polymarket cost model...")

# Polymarket costs:
# - No trading fees (0% maker/taker)
# - Spread cost: typically 1-3% depending on liquidity
# - Slippage: ~0.5-2% for larger orders
SPREAD_COST = 0.02  # 2% round-trip spread
SLIPPAGE = 0.005    # 0.5% slippage per trade

def apply_trading_costs(entry_price, exit_price, position_size):
    """Apply realistic Polymarket trading costs"""
    # Entry cost
    entry_cost = entry_price * (1 + SPREAD_COST/2 + SLIPPAGE)
    # Exit proceeds
    exit_proceeds = exit_price * (1 - SPREAD_COST/2 - SLIPPAGE)

    gross_pnl = (exit_price - entry_price) * position_size
    net_pnl = (exit_proceeds - entry_cost) * position_size

    return net_pnl, gross_pnl

print(f"    Spread cost: {SPREAD_COST*100}%")
print(f"    Slippage: {SLIPPAGE*100}%")
print(f"    Total round-trip cost: ~{(SPREAD_COST + 2*SLIPPAGE)*100}%")

# =============================================================================
# 4. IDENTIFY WHALES - THREE METHODS
# =============================================================================
print("\n[4] Identifying whales using three methods...")

# Calculate user-level stats
user_stats = df.groupby('userId').agg({
    'amount_abs': ['sum', 'mean', 'count'],
    'datetime': ['min', 'max']
}).reset_index()
user_stats.columns = ['userId', 'total_volume', 'avg_trade_size', 'trade_count', 'first_trade', 'last_trade']

# METHOD 1: Top 10 by total trading volume
print("\n    METHOD 1: Top 10 by total trading volume")
whales_top10 = set(user_stats.nlargest(10, 'total_volume')['userId'].tolist())
top10_stats = user_stats[user_stats['userId'].isin(whales_top10)]
print(f"    Whale count: {len(whales_top10)}")
print(f"    Min volume: {top10_stats['total_volume'].min():,.0f}")
print(f"    Max volume: {top10_stats['total_volume'].max():,.0f}")

# METHOD 2: 95th percentile trading volume
print("\n    METHOD 2: 95th percentile trading volume")
threshold_95 = user_stats['total_volume'].quantile(0.95)
whales_95pct = set(user_stats[user_stats['total_volume'] >= threshold_95]['userId'].tolist())
print(f"    95th percentile threshold: {threshold_95:,.0f}")
print(f"    Whale count: {len(whales_95pct)}")

# METHOD 3: Large per-trade size (>10K equivalent, scaled for Manifold's play money)
# Manifold uses mana, so we'll use >1000 mana as "large" (equivalent concept)
print("\n    METHOD 3: Per-trade size based (>1000 mana avg in last week)")
LARGE_TRADE_THRESHOLD = 1000  # Manifold mana (adjust as needed)

# Get last week of data
last_date = df['datetime'].max()
week_ago = last_date - timedelta(days=7)
recent_df = df[df['datetime'] >= week_ago]

recent_user_stats = recent_df.groupby('userId').agg({
    'amount_abs': 'mean'
}).reset_index()
recent_user_stats.columns = ['userId', 'avg_recent_trade']

whales_large_trades = set(
    recent_user_stats[recent_user_stats['avg_recent_trade'] >= LARGE_TRADE_THRESHOLD]['userId'].tolist()
)
print(f"    Large trade threshold: {LARGE_TRADE_THRESHOLD} mana")
print(f"    Whale count: {len(whales_large_trades)}")

# =============================================================================
# 5. BACKTEST FUNCTION
# =============================================================================
print("\n[5] Running backtests...")

def backtest_whale_strategy(df, whale_set, resolution_map, strategy_name):
    """
    Backtest strategy: Follow whale trades into resolved markets

    Strategy:
    - When a whale buys YES, we buy YES at current probability
    - When a whale buys NO, we buy NO at current probability
    - Hold until market resolution
    - Calculate P&L based on resolution
    """

    # Filter to whale trades only
    whale_trades = df[df['userId'].isin(whale_set)].copy()

    # Filter to markets that resolved
    whale_trades = whale_trades[whale_trades['contractId'].isin(resolution_map.keys())]

    if len(whale_trades) == 0:
        return None

    trades = []

    for _, trade in whale_trades.iterrows():
        contract_id = trade['contractId']
        outcome = trade['outcome']
        prob_after = trade.get('probAfter', 0.5)
        amount = abs(trade['amount'])
        resolution = resolution_map.get(contract_id)

        if resolution is None or pd.isna(prob_after):
            continue

        # Determine our position
        if outcome == 'YES':
            # Whale bought YES, we follow
            entry_price = prob_after
            exit_price = resolution  # 1.0 if YES resolves, 0.0 if NO
            position = 'YES'
        elif outcome == 'NO':
            # Whale bought NO, we follow (buy NO = sell YES)
            entry_price = 1 - prob_after
            exit_price = 1 - resolution  # 1.0 if NO resolves, 0.0 if YES
            position = 'NO'
        else:
            continue

        # Normalize position size (use fixed $100 per trade for comparison)
        position_size = 100

        # Calculate P&L with costs
        net_pnl, gross_pnl = apply_trading_costs(entry_price, exit_price, position_size)

        trades.append({
            'datetime': trade['datetime'],
            'contract_id': contract_id,
            'whale_id': trade['userId'],
            'position': position,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'gross_pnl': gross_pnl,
            'net_pnl': net_pnl,
            'position_size': position_size
        })

    if not trades:
        return None

    trades_df = pd.DataFrame(trades)
    trades_df = trades_df.sort_values('datetime')
    trades_df['cumulative_pnl'] = trades_df['net_pnl'].cumsum()
    trades_df['cumulative_gross_pnl'] = trades_df['gross_pnl'].cumsum()

    # Calculate daily returns for quantstats
    trades_df['date'] = trades_df['datetime'].dt.date
    daily_pnl = trades_df.groupby('date')['net_pnl'].sum()

    # Convert to returns (assume $10,000 starting capital)
    starting_capital = 10000
    daily_returns = daily_pnl / starting_capital

    # Stats
    total_trades = len(trades_df)
    winning_trades = (trades_df['net_pnl'] > 0).sum()
    losing_trades = (trades_df['net_pnl'] < 0).sum()
    win_rate = winning_trades / total_trades if total_trades > 0 else 0

    total_gross_pnl = trades_df['gross_pnl'].sum()
    total_net_pnl = trades_df['net_pnl'].sum()
    total_costs = total_gross_pnl - total_net_pnl

    avg_win = trades_df[trades_df['net_pnl'] > 0]['net_pnl'].mean() if winning_trades > 0 else 0
    avg_loss = trades_df[trades_df['net_pnl'] < 0]['net_pnl'].mean() if losing_trades > 0 else 0

    results = {
        'strategy_name': strategy_name,
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': win_rate,
        'total_gross_pnl': total_gross_pnl,
        'total_net_pnl': total_net_pnl,
        'total_costs': total_costs,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': abs(avg_win * winning_trades / (avg_loss * losing_trades)) if losing_trades > 0 and avg_loss != 0 else np.inf,
        'sharpe': daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0,
        'max_drawdown': (trades_df['cumulative_pnl'].cummax() - trades_df['cumulative_pnl']).max(),
        'trades_df': trades_df,
        'daily_returns': daily_returns
    }

    return results

# Run backtests for all three methods
results = {}

print("\n    Backtesting Method 1: Top 10 Whales...")
results['top10'] = backtest_whale_strategy(df, whales_top10, resolution_map, "Top 10 Whales")

print("    Backtesting Method 2: 95th Percentile Whales...")
results['pct95'] = backtest_whale_strategy(df, whales_95pct, resolution_map, "95th Percentile Whales")

print("    Backtesting Method 3: Large Trade Whales...")
results['large_trades'] = backtest_whale_strategy(df, whales_large_trades, resolution_map, "Large Trade Whales")

# =============================================================================
# 6. RESULTS SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print("BACKTEST RESULTS")
print("=" * 60)

for key, res in results.items():
    if res is None:
        print(f"\n{key.upper()}: No trades found")
        continue

    print(f"\n{'='*60}")
    print(f"STRATEGY: {res['strategy_name']}")
    print(f"{'='*60}")
    print(f"Total Trades:      {res['total_trades']:,}")
    print(f"Winning Trades:    {res['winning_trades']:,} ({res['win_rate']*100:.1f}%)")
    print(f"Losing Trades:     {res['losing_trades']:,}")
    print(f"")
    print(f"Gross P&L:         ${res['total_gross_pnl']:,.2f}")
    print(f"Trading Costs:     ${res['total_costs']:,.2f}")
    print(f"Net P&L:           ${res['total_net_pnl']:,.2f}")
    print(f"")
    print(f"Avg Win:           ${res['avg_win']:,.2f}")
    print(f"Avg Loss:          ${res['avg_loss']:,.2f}")
    print(f"Profit Factor:     {res['profit_factor']:.2f}")
    print(f"")
    print(f"Sharpe Ratio:      {res['sharpe']:.2f}")
    print(f"Max Drawdown:      ${res['max_drawdown']:,.2f}")

# =============================================================================
# 7. QUANTSTATS REPORTS
# =============================================================================
print("\n" + "=" * 60)
print("GENERATING QUANTSTATS REPORTS")
print("=" * 60)

import quantstats as qs

for key, res in results.items():
    if res is None:
        continue

    daily_returns = res['daily_returns']
    if len(daily_returns) < 2:
        print(f"\n{key}: Not enough data for quantstats")
        continue

    # Convert to proper datetime index
    returns_series = pd.Series(daily_returns.values, index=pd.to_datetime(daily_returns.index))
    returns_series = returns_series.sort_index()

    # Generate HTML report
    report_path = f"data/output/whale_strategy_{key}_report.html"
    try:
        qs.reports.html(returns_series, output=report_path, title=f"Whale Strategy: {res['strategy_name']}")
        print(f"\n{key}: Report saved to {report_path}")
    except Exception as e:
        print(f"\n{key}: Could not generate HTML report: {e}")

    # Print basic quantstats metrics
    print(f"\n--- {res['strategy_name']} - QuantStats Metrics ---")
    try:
        print(f"CAGR:              {qs.stats.cagr(returns_series)*100:.2f}%")
        print(f"Sharpe:            {qs.stats.sharpe(returns_series):.2f}")
        print(f"Sortino:           {qs.stats.sortino(returns_series):.2f}")
        print(f"Max Drawdown:      {qs.stats.max_drawdown(returns_series)*100:.2f}%")
        print(f"Volatility (ann):  {qs.stats.volatility(returns_series)*100:.2f}%")
        print(f"Calmar Ratio:      {qs.stats.calmar(returns_series):.2f}")
        print(f"Win Rate:          {qs.stats.win_rate(returns_series)*100:.1f}%")
    except Exception as e:
        print(f"Error calculating metrics: {e}")

# =============================================================================
# 8. MOST SUCCESSFUL MARKETS
# =============================================================================
print("\n" + "=" * 60)
print("MOST SUCCESSFUL MARKETS (by whale P&L)")
print("=" * 60)

for key, res in results.items():
    if res is None:
        continue

    trades_df = res['trades_df']

    # Aggregate by market
    market_pnl = trades_df.groupby('contract_id').agg({
        'net_pnl': 'sum',
        'gross_pnl': 'sum',
        'datetime': 'count'
    }).reset_index()
    market_pnl.columns = ['contract_id', 'total_pnl', 'gross_pnl', 'trade_count']
    market_pnl = market_pnl.sort_values('total_pnl', ascending=False)

    # Get market names
    market_names = {m['id']: m.get('question', 'Unknown')[:60] for m in markets}
    market_pnl['question'] = market_pnl['contract_id'].map(market_names)

    print(f"\n--- {res['strategy_name']} ---")
    print("\nTop 10 Most Profitable Markets:")
    print("-" * 80)
    for i, row in market_pnl.head(10).iterrows():
        print(f"  ${row['total_pnl']:>8,.0f} | {row['trade_count']:>3} trades | {row['question']}")

    print("\nTop 10 Least Profitable Markets:")
    print("-" * 80)
    for i, row in market_pnl.tail(10).iterrows():
        print(f"  ${row['total_pnl']:>8,.0f} | {row['trade_count']:>3} trades | {row['question']}")

# =============================================================================
# 9. SAVE DETAILED RESULTS
# =============================================================================
print("\n" + "=" * 60)
print("SAVING DETAILED RESULTS")
print("=" * 60)

import os
os.makedirs('data/output', exist_ok=True)

for key, res in results.items():
    if res is None:
        continue

    # Save trades
    trades_path = f"data/output/whale_trades_{key}.csv"
    res['trades_df'].to_csv(trades_path, index=False)
    print(f"Saved: {trades_path}")

# Summary comparison
summary_data = []
for key, res in results.items():
    if res is None:
        continue
    summary_data.append({
        'Strategy': res['strategy_name'],
        'Total Trades': res['total_trades'],
        'Win Rate': f"{res['win_rate']*100:.1f}%",
        'Gross P&L': f"${res['total_gross_pnl']:,.0f}",
        'Net P&L': f"${res['total_net_pnl']:,.0f}",
        'Trading Costs': f"${res['total_costs']:,.0f}",
        'Sharpe': f"{res['sharpe']:.2f}",
        'Profit Factor': f"{res['profit_factor']:.2f}"
    })

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv('data/output/whale_strategy_summary.csv', index=False)
print("\nSaved: data/output/whale_strategy_summary.csv")

print("\n" + "=" * 60)
print("BACKTEST COMPLETE")
print("=" * 60)

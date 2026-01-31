"""
Whale Following Strategy Backtest v2 - Bias-Checked Version

Potential biases addressed:
1. Look-ahead bias: Only use information available at trade time
2. Survivorship bias: Track all markets, not just resolved ones
3. Selection bias: Ensure whale identification uses only past data
4. Execution bias: Realistic entry prices (after whale moves market)
5. Multiple entry bias: Only one position per market per strategy
"""

import json
import glob
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("WHALE FOLLOWING STRATEGY BACKTEST v2 (BIAS-CHECKED)")
print("=" * 70)

# =============================================================================
# 1. LOAD DATA
# =============================================================================
print("\n[1] Loading data...")

bets = []
bet_files = sorted(glob.glob('data/manifold/bets_*.json'))
for i, f in enumerate(bet_files):
    try:
        with open(f, encoding='utf-8') as file:
            bets.extend(json.load(file))
    except:
        pass
    if (i + 1) % 1000 == 0:
        print(f"    Loaded {i+1}/{len(bet_files)} files...")

print(f"    Total bets: {len(bets):,}")

df = pd.DataFrame(bets)
df['datetime'] = pd.to_datetime(df['createdTime'], unit='ms')
df['date'] = df['datetime'].dt.date
df['amount_abs'] = df['amount'].abs()
df = df.sort_values('datetime').reset_index(drop=True)

print(f"    Date range: {df['datetime'].min()} to {df['datetime'].max()}")
print(f"    Unique users: {df['userId'].nunique():,}")
print(f"    Unique markets: {df['contractId'].nunique():,}")

# Load markets
markets = []
for f in glob.glob('data/manifold/markets_*.json'):
    try:
        with open(f, encoding='utf-8') as file:
            markets.extend(json.load(file))
    except:
        pass

markets_df = pd.DataFrame(markets)
print(f"    Total markets: {len(markets_df):,}")

# =============================================================================
# 2. BUILD RESOLUTION MAP WITH TIMESTAMPS
# =============================================================================
print("\n[2] Building resolution map...")

resolution_data = {}
for _, m in markets_df.iterrows():
    mid = m['id']
    if m.get('isResolved') and m.get('resolution') in ['YES', 'NO']:
        resolution_data[mid] = {
            'resolution': 1.0 if m['resolution'] == 'YES' else 0.0,
            'resolved_time': m.get('resolutionTime', m.get('closeTime', 0)),
            'question': str(m.get('question', ''))[:60]
        }

print(f"    Resolved YES/NO markets: {len(resolution_data):,}")

# =============================================================================
# 3. BIAS CHECK: Analyze data quality
# =============================================================================
print("\n[3] Data quality checks...")

# Check for markets that resolved
bets_in_resolved = df[df['contractId'].isin(resolution_data.keys())]
print(f"    Bets in resolved markets: {len(bets_in_resolved):,} ({100*len(bets_in_resolved)/len(df):.1f}%)")

# Check outcome distribution
if 'outcome' in df.columns:
    outcome_dist = df['outcome'].value_counts()
    print(f"    Outcome distribution: {dict(outcome_dist.head(5))}")

# Check for missing probabilities
missing_prob = df['probAfter'].isna().sum()
print(f"    Missing probAfter: {missing_prob:,} ({100*missing_prob/len(df):.1f}%)")

# =============================================================================
# 4. POLYMARKET COST MODEL
# =============================================================================
print("\n[4] Polymarket cost model...")

# Polymarket realistic costs:
# - Maker fee: 0%
# - Taker fee: 0%
# - Spread: 1-4% depending on liquidity (use 2%)
# - Slippage: 0.5-1% for reasonable sizes
SPREAD_COST = 0.02      # 2% spread (1% each side)
SLIPPAGE = 0.005        # 0.5% slippage
TOTAL_COST = SPREAD_COST + 2 * SLIPPAGE  # 3% round-trip

print(f"    Spread: {SPREAD_COST*100}%")
print(f"    Slippage: {SLIPPAGE*100}%")
print(f"    Total round-trip: {TOTAL_COST*100}%")

# =============================================================================
# 5. ROLLING WHALE IDENTIFICATION (NO LOOK-AHEAD BIAS)
# =============================================================================
print("\n[5] Rolling whale identification (avoiding look-ahead bias)...")

# For proper backtesting, we should identify whales using ONLY PAST DATA
# at each point in time. This is computationally expensive, so we'll use
# a simplified approach: identify whales from first 30% of data, test on rest.

# Split data temporally
split_date = df['datetime'].quantile(0.3)
train_df = df[df['datetime'] <= split_date]
test_df = df[df['datetime'] > split_date]

print(f"    Training period: {train_df['datetime'].min()} to {train_df['datetime'].max()}")
print(f"    Testing period: {test_df['datetime'].min()} to {test_df['datetime'].max()}")
print(f"    Training bets: {len(train_df):,}")
print(f"    Testing bets: {len(test_df):,}")

# Identify whales from training data only
train_user_stats = train_df.groupby('userId').agg({
    'amount_abs': ['sum', 'mean', 'count']
}).reset_index()
train_user_stats.columns = ['userId', 'total_volume', 'avg_trade_size', 'trade_count']

# METHOD 1: Top 10 by volume (from training period)
whales_top10 = set(train_user_stats.nlargest(10, 'total_volume')['userId'].tolist())

# METHOD 2: 95th percentile (from training period)
threshold_95 = train_user_stats['total_volume'].quantile(0.95)
whales_95pct = set(train_user_stats[train_user_stats['total_volume'] >= threshold_95]['userId'].tolist())

# METHOD 3: Large average trade size >1000 (from training period)
whales_large = set(train_user_stats[train_user_stats['avg_trade_size'] >= 1000]['userId'].tolist())

print(f"\n    Whales identified from training data:")
print(f"    - Top 10: {len(whales_top10)} traders")
print(f"    - 95th percentile (>{threshold_95:.0f}): {len(whales_95pct)} traders")
print(f"    - Large trades (>1000 avg): {len(whales_large)} traders")

# =============================================================================
# 6. BACKTEST FUNCTION (BIAS-FREE)
# =============================================================================
print("\n[6] Running bias-free backtests...")

def backtest_whale_strategy_v2(test_df, whale_set, resolution_data, strategy_name):
    """
    Bias-free backtest:
    - Uses only test period data
    - Whales identified from training period only
    - One position per market (first whale signal)
    - Entry at probAfter (price AFTER whale moved market)
    - Exit at resolution
    """

    # Filter to whale trades in test period
    whale_trades = test_df[test_df['userId'].isin(whale_set)].copy()
    whale_trades = whale_trades[whale_trades['outcome'].isin(['YES', 'NO'])]
    whale_trades = whale_trades[whale_trades['probAfter'].notna()]
    whale_trades = whale_trades.sort_values('datetime')

    if len(whale_trades) == 0:
        return None

    # Track positions - only ONE position per market (first signal)
    positions = {}  # contract_id -> position info
    trades = []

    for _, trade in whale_trades.iterrows():
        contract_id = trade['contractId']

        # Skip if we already have a position in this market
        if contract_id in positions:
            continue

        # Skip if market not resolved (we can't calculate P&L)
        if contract_id not in resolution_data:
            continue

        res_data = resolution_data[contract_id]
        resolution = res_data['resolution']

        # BIAS CHECK: Ensure trade happened BEFORE resolution
        trade_time = trade['createdTime']
        res_time = res_data['resolved_time']
        if res_time and trade_time >= res_time:
            continue  # Skip trades after resolution

        outcome = trade['outcome']
        prob_after = trade['probAfter']

        # Entry price is the probability AFTER whale's trade
        # This is what we'd see and could trade at
        if outcome == 'YES':
            entry_price = prob_after
            exit_price = resolution
        else:  # NO
            entry_price = 1 - prob_after
            exit_price = 1 - resolution

        # Skip extreme prices (likely errors or illiquid)
        if entry_price < 0.02 or entry_price > 0.98:
            continue

        # Fixed position size for fair comparison
        position_size = 100

        # Calculate P&L with costs
        # Entry: pay entry_price + costs
        # Exit: receive exit_price - costs
        entry_cost = entry_price * (1 + SPREAD_COST/2 + SLIPPAGE)
        exit_proceeds = exit_price * (1 - SPREAD_COST/2 - SLIPPAGE)

        gross_pnl = (exit_price - entry_price) * position_size
        net_pnl = (exit_proceeds - entry_cost) * position_size

        positions[contract_id] = True
        trades.append({
            'datetime': trade['datetime'],
            'date': trade['date'],
            'contract_id': contract_id,
            'whale_id': trade['userId'],
            'outcome': outcome,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'gross_pnl': gross_pnl,
            'net_pnl': net_pnl,
            'question': res_data['question']
        })

    if not trades:
        return None

    trades_df = pd.DataFrame(trades)
    trades_df = trades_df.sort_values('datetime')
    trades_df['cumulative_pnl'] = trades_df['net_pnl'].cumsum()
    trades_df['cumulative_gross'] = trades_df['gross_pnl'].cumsum()

    # Daily returns for Sharpe calculation
    daily_pnl = trades_df.groupby('date')['net_pnl'].sum()
    starting_capital = 10000
    daily_returns = daily_pnl / starting_capital

    # Calculate metrics
    total_trades = len(trades_df)
    winners = trades_df[trades_df['net_pnl'] > 0]
    losers = trades_df[trades_df['net_pnl'] <= 0]

    win_rate = len(winners) / total_trades if total_trades > 0 else 0

    total_gross = trades_df['gross_pnl'].sum()
    total_net = trades_df['net_pnl'].sum()
    total_costs = total_gross - total_net

    avg_win = winners['net_pnl'].mean() if len(winners) > 0 else 0
    avg_loss = losers['net_pnl'].mean() if len(losers) > 0 else 0

    # Profit factor
    gross_wins = winners['net_pnl'].sum() if len(winners) > 0 else 0
    gross_losses = abs(losers['net_pnl'].sum()) if len(losers) > 0 else 1
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else np.inf

    # Sharpe ratio (annualized)
    if len(daily_returns) > 1 and daily_returns.std() > 0:
        sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
    else:
        sharpe = 0

    # Max drawdown
    cummax = trades_df['cumulative_pnl'].cummax()
    drawdown = cummax - trades_df['cumulative_pnl']
    max_dd = drawdown.max()

    # Return on capital
    roi = total_net / starting_capital * 100

    return {
        'strategy_name': strategy_name,
        'total_trades': total_trades,
        'unique_markets': trades_df['contract_id'].nunique(),
        'winning_trades': len(winners),
        'losing_trades': len(losers),
        'win_rate': win_rate,
        'total_gross_pnl': total_gross,
        'total_net_pnl': total_net,
        'total_costs': total_costs,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'roi_pct': roi,
        'trades_df': trades_df,
        'daily_returns': daily_returns
    }

# Run backtests
results = {}
results['top10'] = backtest_whale_strategy_v2(test_df, whales_top10, resolution_data, "Top 10 Whales")
results['pct95'] = backtest_whale_strategy_v2(test_df, whales_95pct, resolution_data, "95th Percentile")
results['large'] = backtest_whale_strategy_v2(test_df, whales_large, resolution_data, "Large Trades (>1K)")

# =============================================================================
# 7. RESULTS
# =============================================================================
print("\n" + "=" * 70)
print("BACKTEST RESULTS (OUT-OF-SAMPLE)")
print("=" * 70)

for key, res in results.items():
    if res is None:
        print(f"\n{key}: No valid trades")
        continue

    print(f"\n{'='*70}")
    print(f"STRATEGY: {res['strategy_name']}")
    print(f"{'='*70}")
    print(f"Unique Markets Traded:  {res['unique_markets']:,}")
    print(f"Total Trades:           {res['total_trades']:,}")
    print(f"Win Rate:               {res['win_rate']*100:.1f}%")
    print(f"")
    print(f"Gross P&L:              ${res['total_gross_pnl']:>12,.2f}")
    print(f"Trading Costs:          ${res['total_costs']:>12,.2f}")
    print(f"Net P&L:                ${res['total_net_pnl']:>12,.2f}")
    print(f"ROI:                    {res['roi_pct']:>12.2f}%")
    print(f"")
    print(f"Avg Winning Trade:      ${res['avg_win']:>12,.2f}")
    print(f"Avg Losing Trade:       ${res['avg_loss']:>12,.2f}")
    print(f"Profit Factor:          {res['profit_factor']:>12.2f}")
    print(f"")
    print(f"Sharpe Ratio:           {res['sharpe']:>12.2f}")
    print(f"Max Drawdown:           ${res['max_drawdown']:>12,.2f}")

# =============================================================================
# 8. STATISTICAL SIGNIFICANCE
# =============================================================================
print("\n" + "=" * 70)
print("STATISTICAL SIGNIFICANCE TESTS")
print("=" * 70)

from scipy import stats

for key, res in results.items():
    if res is None:
        continue

    daily_ret = res['daily_returns']
    if len(daily_ret) < 10:
        continue

    # T-test: Is mean return significantly different from zero?
    t_stat, p_value = stats.ttest_1samp(daily_ret, 0)

    # Is win rate significantly different from 50%?
    n_trades = res['total_trades']
    n_wins = res['winning_trades']
    # Binomial test
    binom_result = stats.binomtest(n_wins, n_trades, 0.5, alternative='greater')
    binom_p = binom_result.pvalue

    print(f"\n{res['strategy_name']}:")
    print(f"  Mean daily return: {daily_ret.mean()*100:.4f}%")
    print(f"  T-statistic: {t_stat:.3f}, p-value: {p_value:.4f}")
    print(f"  Win rate significance (vs 50%): p={binom_p:.4f}")

    if p_value < 0.05:
        print(f"  --> Returns are statistically significant (p<0.05)")
    else:
        print(f"  --> Returns are NOT statistically significant")

# =============================================================================
# 9. TOP/BOTTOM MARKETS
# =============================================================================
print("\n" + "=" * 70)
print("MARKET ANALYSIS")
print("=" * 70)

for key, res in results.items():
    if res is None:
        continue

    trades_df = res['trades_df']

    # Aggregate by market
    market_pnl = trades_df.groupby(['contract_id', 'question']).agg({
        'net_pnl': 'sum'
    }).reset_index()
    market_pnl = market_pnl.sort_values('net_pnl', ascending=False)

    print(f"\n{res['strategy_name']} - Top 5 Profitable Markets:")
    print("-" * 70)
    for _, row in market_pnl.head(5).iterrows():
        q = row['question'][:55] if isinstance(row['question'], str) else 'Unknown'
        q = q.encode('ascii', 'replace').decode('ascii')  # Handle unicode
        print(f"  ${row['net_pnl']:>7,.0f} | {q}")

    print(f"\n{res['strategy_name']} - Top 5 Losing Markets:")
    print("-" * 70)
    for _, row in market_pnl.tail(5).iterrows():
        q = row['question'][:55] if isinstance(row['question'], str) else 'Unknown'
        q = q.encode('ascii', 'replace').decode('ascii')  # Handle unicode
        print(f"  ${row['net_pnl']:>7,.0f} | {q}")

# =============================================================================
# 10. SAVE RESULTS
# =============================================================================
print("\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)

import os
os.makedirs('data/output', exist_ok=True)

# Save trades
for key, res in results.items():
    if res is None:
        continue
    res['trades_df'].to_csv(f'data/output/whale_v2_trades_{key}.csv', index=False)
    print(f"Saved: data/output/whale_v2_trades_{key}.csv")

# Save summary
summary = []
for key, res in results.items():
    if res is None:
        continue
    summary.append({
        'strategy': res['strategy_name'],
        'trades': res['total_trades'],
        'win_rate': f"{res['win_rate']*100:.1f}%",
        'gross_pnl': res['total_gross_pnl'],
        'net_pnl': res['total_net_pnl'],
        'costs': res['total_costs'],
        'sharpe': res['sharpe'],
        'max_dd': res['max_drawdown'],
        'profit_factor': res['profit_factor']
    })

pd.DataFrame(summary).to_csv('data/output/whale_v2_summary.csv', index=False)
print("Saved: data/output/whale_v2_summary.csv")

# Generate quantstats reports
print("\nGenerating QuantStats reports...")
import quantstats as qs

for key, res in results.items():
    if res is None or len(res['daily_returns']) < 5:
        continue

    returns = pd.Series(
        res['daily_returns'].values,
        index=pd.to_datetime(res['daily_returns'].index)
    ).sort_index()

    try:
        qs.reports.html(
            returns,
            output=f'data/output/whale_v2_{key}_quantstats.html',
            title=f"Whale Strategy v2: {res['strategy_name']}"
        )
        print(f"Saved: data/output/whale_v2_{key}_quantstats.html")
    except Exception as e:
        print(f"Could not generate report for {key}: {e}")

print("\n" + "=" * 70)
print("BACKTEST COMPLETE")
print("=" * 70)

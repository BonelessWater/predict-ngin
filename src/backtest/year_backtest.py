"""
Year-long backtest runner with detailed analysis.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import INITIAL_CAPITAL, CONSENSUS_THRESHOLD, PARQUET_DIR
from src.whales.detector import WhaleDetector, WhaleMethod
from src.strategy.copycat import CopycatStrategy, StrategyState
from src.backtest.simulator import generate_year_data, analyze_regime_performance
from src.backtest.engine import BacktestEngine, BacktestMetrics

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def calculate_extended_metrics(state: StrategyState, prices_df: pd.DataFrame) -> dict:
    """Calculate additional metrics for detailed analysis."""
    trades = [t for t in state.trades if t.action != 'hold']

    if not trades:
        return {}

    # Trade-level analysis
    trade_returns = []
    trade_durations = []
    entry_time = None

    for t in trades:
        if t.action in ['buy', 'sell']:
            entry_time = t.timestamp
        elif t.action in ['close_long', 'close_short'] and entry_time:
            duration = (t.timestamp - entry_time).total_seconds() / 3600  # hours
            trade_durations.append(duration)
            entry_time = None

        if t.capital_before > 0:
            ret = (t.capital_after - t.capital_before) / t.capital_before
            trade_returns.append(ret)

    # Consecutive wins/losses
    max_consecutive_wins = 0
    max_consecutive_losses = 0
    current_wins = 0
    current_losses = 0

    for ret in trade_returns:
        if ret > 0:
            current_wins += 1
            current_losses = 0
            max_consecutive_wins = max(max_consecutive_wins, current_wins)
        else:
            current_losses += 1
            current_wins = 0
            max_consecutive_losses = max(max_consecutive_losses, current_losses)

    # Monthly returns
    equity_by_month = {}
    current_capital = INITIAL_CAPITAL

    for t in state.trades:
        if t.action in ['close_long', 'close_short']:
            month_key = t.timestamp.strftime('%Y-%m')
            current_capital = t.capital_after
            equity_by_month[month_key] = current_capital

    monthly_returns = []
    prev_equity = INITIAL_CAPITAL
    for month, equity in sorted(equity_by_month.items()):
        monthly_ret = (equity - prev_equity) / prev_equity
        monthly_returns.append({'month': month, 'return': monthly_ret, 'equity': equity})
        prev_equity = equity

    return {
        'avg_trade_duration_hours': np.mean(trade_durations) if trade_durations else 0,
        'max_trade_duration_hours': max(trade_durations) if trade_durations else 0,
        'min_trade_duration_hours': min(trade_durations) if trade_durations else 0,
        'max_consecutive_wins': max_consecutive_wins,
        'max_consecutive_losses': max_consecutive_losses,
        'monthly_returns': monthly_returns,
        'best_month': max(monthly_returns, key=lambda x: x['return']) if monthly_returns else None,
        'worst_month': min(monthly_returns, key=lambda x: x['return']) if monthly_returns else None,
        'positive_months': sum(1 for m in monthly_returns if m['return'] > 0),
        'total_months': len(monthly_returns)
    }


def run_year_backtest():
    """Run comprehensive year-long backtest."""
    print("\n" + "=" * 70)
    print("YEAR-LONG WHALE COPYCAT STRATEGY BACKTEST")
    print("=" * 70)

    # Generate data
    logger.info("Generating 1 year of synthetic market data...")
    trades_df, prices_df = generate_year_data(seed=42)

    # Save raw data
    trades_df.to_parquet(PARQUET_DIR / "synthetic_trades_1y.parquet")
    prices_df.to_parquet(PARQUET_DIR / "synthetic_prices_1y.parquet")

    detector = WhaleDetector()
    results = {}
    all_metrics = {}

    for method in WhaleMethod:
        print(f"\n{'='*70}")
        print(f"BACKTESTING: {method.value.upper()}")
        print(f"{'='*70}")

        strategy = CopycatStrategy(
            whale_detector=detector,
            whale_method=method,
            consensus_threshold=CONSENSUS_THRESHOLD,
            initial_capital=INITIAL_CAPITAL
        )

        state = strategy.run_backtest(
            trades_df=trades_df,
            prices_df=prices_df,
            signal_interval_hours=4
        )

        results[method] = state

        # Calculate metrics
        actual_trades = [t for t in state.trades if t.action != 'hold']
        trade_returns = []
        wins = []
        losses = []

        for t in actual_trades:
            if t.capital_before > 0:
                ret = (t.capital_after - t.capital_before) / t.capital_before
                trade_returns.append(ret)
                if ret > 0:
                    wins.append(ret)
                elif ret < 0:
                    losses.append(ret)

        extended = calculate_extended_metrics(state, prices_df)

        metrics = {
            'method': method.value,
            'initial_capital': INITIAL_CAPITAL,
            'final_capital': state.capital,
            'total_return_pct': (state.capital / INITIAL_CAPITAL - 1) * 100,
            'num_trades': len(actual_trades),
            'num_wins': len(wins),
            'num_losses': len(losses),
            'win_rate': len(wins) / len(trade_returns) * 100 if trade_returns else 0,
            'avg_win_pct': np.mean(wins) * 100 if wins else 0,
            'avg_loss_pct': np.mean(losses) * 100 if losses else 0,
            'largest_win_pct': max(wins) * 100 if wins else 0,
            'largest_loss_pct': min(losses) * 100 if losses else 0,
            'profit_factor': abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float('inf'),
            'expectancy': np.mean(trade_returns) * 100 if trade_returns else 0,
            **extended
        }

        all_metrics[method.value] = metrics

        # Print summary
        print(f"\n📊 RESULTS: {method.value.upper()}")
        print("-" * 50)
        print(f"Initial Capital:     ${INITIAL_CAPITAL:>12,.2f}")
        print(f"Final Capital:       ${state.capital:>12,.2f}")
        print(f"Total Return:        {metrics['total_return_pct']:>12.2f}%")
        print(f"")
        print(f"Total Trades:        {metrics['num_trades']:>12}")
        print(f"Winning Trades:      {metrics['num_wins']:>12}")
        print(f"Losing Trades:       {metrics['num_losses']:>12}")
        print(f"Win Rate:            {metrics['win_rate']:>12.1f}%")
        print(f"")
        print(f"Average Win:         {metrics['avg_win_pct']:>12.2f}%")
        print(f"Average Loss:        {metrics['avg_loss_pct']:>12.2f}%")
        print(f"Largest Win:         {metrics['largest_win_pct']:>12.2f}%")
        print(f"Largest Loss:        {metrics['largest_loss_pct']:>12.2f}%")
        print(f"")
        print(f"Profit Factor:       {metrics['profit_factor']:>12.2f}")
        print(f"Expectancy/Trade:    {metrics['expectancy']:>12.2f}%")
        print(f"")
        print(f"Avg Trade Duration:  {metrics['avg_trade_duration_hours']:>12.1f} hours")
        print(f"Max Consec. Wins:    {metrics['max_consecutive_wins']:>12}")
        print(f"Max Consec. Losses:  {metrics['max_consecutive_losses']:>12}")
        print(f"")
        print(f"Positive Months:     {metrics['positive_months']:>12} / {metrics['total_months']}")

        if metrics.get('best_month'):
            print(f"Best Month:          {metrics['best_month']['month']} ({metrics['best_month']['return']*100:+.1f}%)")
        if metrics.get('worst_month'):
            print(f"Worst Month:         {metrics['worst_month']['month']} ({metrics['worst_month']['return']*100:+.1f}%)")

        # Regime analysis
        print(f"\n📈 Performance by Market Regime:")
        regime_df = analyze_regime_performance(trades_df, state.trades, prices_df)
        if not regime_df.empty:
            regime_summary = regime_df.groupby('regime').agg({
                'pnl': ['sum', 'mean', 'count'],
                'return_pct': 'mean'
            }).round(2)
            print(regime_summary.to_string())

    # Comparison table
    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY")
    print(f"{'='*70}")

    comparison_df = pd.DataFrame([
        {
            'Method': m['method'],
            'Return %': f"{m['total_return_pct']:.1f}%",
            'Trades': m['num_trades'],
            'Win Rate': f"{m['win_rate']:.1f}%",
            'Avg Win': f"{m['avg_win_pct']:.1f}%",
            'Avg Loss': f"{m['avg_loss_pct']:.1f}%",
            'Profit Factor': f"{m['profit_factor']:.2f}",
            'Expectancy': f"{m['expectancy']:.2f}%"
        }
        for m in all_metrics.values()
    ])
    print(comparison_df.to_string(index=False))

    # Win rate explanation
    print(f"\n{'='*70}")
    print("WHY WIN RATE < 50% IS OK")
    print(f"{'='*70}")

    for method, m in all_metrics.items():
        if m['avg_loss_pct'] != 0:
            reward_risk = abs(m['avg_win_pct'] / m['avg_loss_pct'])
            breakeven_wr = 1 / (1 + reward_risk) * 100
            print(f"\n{method}:")
            print(f"  Reward:Risk Ratio = {reward_risk:.2f}:1")
            print(f"  Breakeven Win Rate = {breakeven_wr:.1f}%")
            print(f"  Actual Win Rate = {m['win_rate']:.1f}%")
            print(f"  Edge = {m['win_rate'] - breakeven_wr:+.1f}% above breakeven")

    # Save results
    with open(PARQUET_DIR / "year_backtest_metrics.json", "w") as f:
        # Convert non-serializable items
        save_metrics = {}
        for k, v in all_metrics.items():
            save_metrics[k] = {
                key: val if not isinstance(val, (np.floating, np.integer)) else float(val)
                for key, val in v.items()
                if key != 'monthly_returns'
            }
        json.dump(save_metrics, f, indent=2, default=str)

    # Save trade logs
    for method, state in results.items():
        trades_data = [{
            'timestamp': str(t.timestamp),
            'action': t.action,
            'price': t.price,
            'consensus': t.whale_consensus,
            'capital_after': t.capital_after,
            'reason': t.reason
        } for t in state.trades if t.action != 'hold']

        df = pd.DataFrame(trades_data)
        df.to_parquet(PARQUET_DIR / f"year_trades_{method.value}.parquet")

    print(f"\n✅ Results saved to {PARQUET_DIR}")

    return all_metrics


if __name__ == "__main__":
    run_year_backtest()

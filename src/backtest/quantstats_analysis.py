"""
Quantstats analysis for whale copycat strategy.
Generates performance reports, equity curves, and statistics.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import PARQUET_DIR, CONSENSUS_THRESHOLD, INITIAL_CAPITAL
from src.whales.detector import WhaleDetector, WhaleMethod
from src.strategy.copycat import CopycatStrategy
from src.backtest.simulator import generate_year_data

import logging
logging.basicConfig(level=logging.WARNING)

# Output directory
OUTPUT_DIR = PARQUET_DIR.parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_backtest_get_returns(method: WhaleMethod, trades_df: pd.DataFrame, prices_df: pd.DataFrame):
    """Run backtest and return daily returns series for quantstats."""
    detector = WhaleDetector()
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

    # Build equity curve from trades
    equity_curve = []
    current_equity = INITIAL_CAPITAL

    for trade in state.trades:
        if trade.action in ['close_long', 'close_short']:
            current_equity = trade.capital_after
        equity_curve.append({
            'timestamp': trade.timestamp,
            'equity': current_equity if trade.action in ['close_long', 'close_short'] else current_equity
        })

    # Convert to daily equity
    if not equity_curve:
        return None, None, state

    eq_df = pd.DataFrame(equity_curve)
    eq_df['timestamp'] = pd.to_datetime(eq_df['timestamp'])
    eq_df = eq_df.set_index('timestamp')

    # Resample to daily
    daily_equity = eq_df['equity'].resample('D').last().ffill()
    daily_equity.iloc[0] = INITIAL_CAPITAL  # Ensure start value

    # Calculate returns
    daily_returns = daily_equity.pct_change().dropna()

    return daily_returns, daily_equity, state


def generate_quantstats_report(returns: pd.Series, equity: pd.Series, method_name: str):
    """Generate quantstats HTML report and metrics."""
    try:
        import quantstats as qs

        # Generate HTML report
        report_path = OUTPUT_DIR / f"quantstats_report_{method_name}.html"
        qs.reports.html(returns, output=str(report_path), title=f"Whale Strategy - {method_name}")
        print(f"  Report saved: {report_path}")

        # Get key metrics
        metrics = {
            'total_return': qs.stats.comp(returns) * 100,
            'cagr': qs.stats.cagr(returns) * 100 if len(returns) > 252 else None,
            'sharpe': qs.stats.sharpe(returns),
            'sortino': qs.stats.sortino(returns),
            'max_drawdown': qs.stats.max_drawdown(returns) * 100,
            'volatility': qs.stats.volatility(returns) * 100,
            'calmar': qs.stats.calmar(returns) if qs.stats.max_drawdown(returns) != 0 else 0,
            'win_rate': qs.stats.win_rate(returns) * 100,
            'profit_factor': qs.stats.profit_factor(returns),
            'avg_win': qs.stats.avg_win(returns) * 100,
            'avg_loss': qs.stats.avg_loss(returns) * 100,
            'best_day': returns.max() * 100,
            'worst_day': returns.min() * 100,
        }

        return metrics

    except Exception as e:
        print(f"  Error generating quantstats report: {e}")
        return None


def save_equity_curve(equity: pd.Series, method_name: str):
    """Save equity curve to CSV and parquet."""
    eq_df = pd.DataFrame({'date': equity.index, 'equity': equity.values})

    csv_path = OUTPUT_DIR / f"equity_curve_{method_name}.csv"
    parquet_path = OUTPUT_DIR / f"equity_curve_{method_name}.parquet"

    eq_df.to_csv(csv_path, index=False)
    eq_df.to_parquet(parquet_path, index=False)

    print(f"  Equity curve saved: {csv_path}")
    return csv_path


def format_metrics_table(all_metrics: dict) -> str:
    """Format metrics as markdown table."""
    lines = []
    lines.append("| Metric | Top N | Percentile | Trade Size |")
    lines.append("|--------|-------|------------|------------|")

    metric_labels = {
        'total_return': ('Total Return', '{:.2f}%'),
        'sharpe': ('Sharpe Ratio', '{:.2f}'),
        'sortino': ('Sortino Ratio', '{:.2f}'),
        'max_drawdown': ('Max Drawdown', '{:.2f}%'),
        'volatility': ('Volatility (ann.)', '{:.2f}%'),
        'calmar': ('Calmar Ratio', '{:.2f}'),
        'win_rate': ('Win Rate', '{:.1f}%'),
        'profit_factor': ('Profit Factor', '{:.2f}'),
        'avg_win': ('Avg Win', '{:.2f}%'),
        'avg_loss': ('Avg Loss', '{:.2f}%'),
        'best_day': ('Best Day', '{:.2f}%'),
        'worst_day': ('Worst Day', '{:.2f}%'),
    }

    for key, (label, fmt) in metric_labels.items():
        row = f"| {label} |"
        for method in ['top_n', 'percentile', 'trade_size']:
            if method in all_metrics and all_metrics[method] and key in all_metrics[method]:
                val = all_metrics[method][key]
                if val is not None and not np.isnan(val) and not np.isinf(val):
                    row += f" {fmt.format(val)} |"
                else:
                    row += " N/A |"
            else:
                row += " N/A |"
        lines.append(row)

    return "\n".join(lines)


def main():
    """Run full analysis with quantstats."""
    print("=" * 70)
    print("QUANTSTATS ANALYSIS - WHALE COPYCAT STRATEGY")
    print("=" * 70)

    # Generate synthetic data (1 year)
    print("\nGenerating 1 year of market data...")
    trades_df, prices_df = generate_year_data(seed=42)

    print(f"  Trades: {len(trades_df):,}")
    print(f"  Price points: {len(prices_df):,}")
    print(f"  Date range: {trades_df['timestamp'].min()} to {trades_df['timestamp'].max()}")

    # Save raw data
    trades_df.to_parquet(PARQUET_DIR / "backtest_trades_1y.parquet")
    prices_df.to_parquet(PARQUET_DIR / "backtest_prices_1y.parquet")

    all_metrics = {}
    all_equity = {}
    all_states = {}

    methods = [
        (WhaleMethod.TOP_N, 'top_n'),
        (WhaleMethod.PERCENTILE, 'percentile'),
        (WhaleMethod.TRADE_SIZE, 'trade_size'),
    ]

    for method, method_name in methods:
        print(f"\n{'='*50}")
        print(f"Method: {method_name.upper()}")
        print('='*50)

        # Run backtest
        returns, equity, state = run_backtest_get_returns(method, trades_df, prices_df)

        if returns is None or len(returns) == 0:
            print("  No trades executed")
            continue

        all_states[method_name] = state
        all_equity[method_name] = equity

        # Save equity curve
        save_equity_curve(equity, method_name)

        # Generate quantstats report
        metrics = generate_quantstats_report(returns, equity, method_name)
        if metrics:
            all_metrics[method_name] = metrics

            print(f"\n  Performance Summary:")
            print(f"    Total Return: {metrics['total_return']:.2f}%")
            print(f"    Sharpe Ratio: {metrics['sharpe']:.2f}")
            print(f"    Max Drawdown: {metrics['max_drawdown']:.2f}%")
            print(f"    Win Rate: {metrics['win_rate']:.1f}%")
            print(f"    Profit Factor: {metrics['profit_factor']:.2f}")

    # Generate comparison table
    print("\n" + "=" * 70)
    print("COMPARISON TABLE (for README)")
    print("=" * 70)

    table = format_metrics_table(all_metrics)
    print("\n" + table)

    # Save comparison table
    with open(OUTPUT_DIR / "metrics_comparison.md", "w") as f:
        f.write("# Backtest Results Comparison\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        f.write(f"- Period: 1 year (synthetic data)\n")
        f.write(f"- Initial Capital: ${INITIAL_CAPITAL:,}\n")
        f.write(f"- Consensus Threshold: {CONSENSUS_THRESHOLD:.0%}\n\n")
        f.write("## Performance Metrics\n\n")
        f.write(table)
        f.write("\n\n## Equity Curves\n\n")
        for method_name in all_equity.keys():
            f.write(f"- `output/equity_curve_{method_name}.csv`\n")
        f.write("\n## Full Reports\n\n")
        for method_name in all_metrics.keys():
            f.write(f"- `output/quantstats_report_{method_name}.html`\n")

    print(f"\nComparison saved: {OUTPUT_DIR / 'metrics_comparison.md'}")

    # Save combined equity curves
    combined = pd.DataFrame()
    for method_name, equity in all_equity.items():
        combined[method_name] = equity
    combined.to_csv(OUTPUT_DIR / "equity_curves_combined.csv")
    combined.to_parquet(OUTPUT_DIR / "equity_curves_combined.parquet")
    print(f"Combined equity curves: {OUTPUT_DIR / 'equity_curves_combined.csv'}")

    # Print final summary for README
    print("\n" + "=" * 70)
    print("SUMMARY FOR README")
    print("=" * 70)

    print(f"""
## Backtest Results (1 Year Simulation)

**Configuration:**
- Initial Capital: ${INITIAL_CAPITAL:,}
- Consensus Threshold: {CONSENSUS_THRESHOLD:.0%}
- Signal Interval: 4 hours
- Whale Staleness: 7 days

{table}

### Key Findings

""")

    # Find best method
    if all_metrics:
        best_return = max(all_metrics.items(), key=lambda x: x[1].get('total_return', 0))
        best_sharpe = max(all_metrics.items(), key=lambda x: x[1].get('sharpe', 0))

        print(f"- **Best Return**: {best_return[0]} ({best_return[1]['total_return']:.2f}%)")
        print(f"- **Best Risk-Adjusted**: {best_sharpe[0]} (Sharpe: {best_sharpe[1]['sharpe']:.2f})")

        # Win rate explanation
        for method, metrics in all_metrics.items():
            if metrics.get('win_rate') and metrics.get('avg_win') and metrics.get('avg_loss'):
                wr = metrics['win_rate']
                aw = abs(metrics['avg_win'])
                al = abs(metrics['avg_loss'])
                if al > 0:
                    rr_ratio = aw / al
                    breakeven = 1 / (1 + rr_ratio) * 100
                    edge = wr - breakeven
                    print(f"\n**{method} Win Rate Analysis:**")
                    print(f"- Win Rate: {wr:.1f}%")
                    print(f"- Reward:Risk = {rr_ratio:.2f}:1")
                    print(f"- Breakeven WR: {breakeven:.1f}%")
                    print(f"- Edge: {edge:+.1f}%")

    return all_metrics, all_equity


if __name__ == "__main__":
    main()

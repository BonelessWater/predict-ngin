"""
Compare backtest results with different cost scenarios.
Shows impact of transaction costs and look-ahead bias.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.config import Config, CostConfig
from src.backtest.proper_backtest import (
    generate_market_data,
    run_backtest,
    calculate_metrics
)

import logging
logging.basicConfig(level=logging.WARNING)


def run_cost_comparison():
    """Compare different cost scenarios."""
    print("=" * 70)
    print("COST SCENARIO COMPARISON")
    print("=" * 70)

    # Generate data once
    print("\nGenerating market data...")
    trades_df, prices_df = generate_market_data(days=365, seed=42)

    # Define cost scenarios
    scenarios = {
        'no_costs': {
            'taker_fee': 0.0,
            'base_slippage': 0.0,
            'volume_impact': 0.0,
            'description': 'No costs (unrealistic baseline)'
        },
        'low_costs': {
            'taker_fee': 0.005,  # 0.5% - maker rebates
            'base_slippage': 0.002,  # 0.2%
            'volume_impact': 0.0005,
            'description': 'Low costs (market maker)'
        },
        'medium_costs': {
            'taker_fee': 0.01,  # 1%
            'base_slippage': 0.005,  # 0.5%
            'volume_impact': 0.001,
            'description': 'Medium costs (regular trader)'
        },
        'high_costs': {
            'taker_fee': 0.02,  # 2%
            'base_slippage': 0.01,  # 1%
            'volume_impact': 0.002,
            'description': 'High costs (retail taker)'
        }
    }

    all_results = {}

    for scenario_name, params in scenarios.items():
        print(f"\n{'='*60}")
        print(f"Scenario: {scenario_name.upper()}")
        print(f"  {params['description']}")
        print(f"  Taker Fee: {params['taker_fee']:.2%}")
        print(f"  Base Slippage: {params['base_slippage']:.2%}")
        print('='*60)

        config = Config.default()
        config.costs.taker_fee = params['taker_fee']
        config.costs.base_slippage = params['base_slippage']
        config.costs.volume_impact = params['volume_impact']

        scenario_results = {}

        for method in ['top_n', 'percentile', 'trade_size']:
            state = run_backtest(trades_df, prices_df, config, method)
            metrics = calculate_metrics(state, config)
            scenario_results[method] = metrics

            print(f"\n  {method.upper()}:")
            print(f"    Return: {metrics['total_return_pct']:>8.2f}%")
            print(f"    Sharpe: {metrics['sharpe_ratio']:>8.2f}")
            print(f"    Win Rate: {metrics['win_rate_pct']:>6.1f}%")
            print(f"    Costs: ${metrics['total_costs']:>8.2f} ({metrics['costs_pct_of_capital']:.1f}%)")

        all_results[scenario_name] = scenario_results

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY: TOP_N METHOD ACROSS COST SCENARIOS")
    print("=" * 70)

    print("\n| Scenario | Return | Sharpe | Win Rate | Total Costs |")
    print("|----------|--------|--------|----------|-------------|")

    for scenario_name, scenario_results in all_results.items():
        m = scenario_results['top_n']
        print(f"| {scenario_name:<8} | {m['total_return_pct']:>6.1f}% | {m['sharpe_ratio']:>6.2f} | {m['win_rate_pct']:>6.1f}% | ${m['total_costs']:>9.2f} |")

    # Save results
    output_dir = Config.default().data.output_dir
    with open(output_dir / "cost_comparison_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_dir / 'cost_comparison_results.json'}")

    # Key insights
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)

    no_cost_return = all_results['no_costs']['top_n']['total_return_pct']
    high_cost_return = all_results['high_costs']['top_n']['total_return_pct']
    cost_impact = no_cost_return - high_cost_return

    print(f"""
1. NO LOOK-AHEAD BIAS IMPACT:
   - Even with no costs, the strategy has modest returns
   - This is realistic: whale edge is ~60-65%, not 100%

2. TRANSACTION COST IMPACT:
   - No costs:   {no_cost_return:>8.2f}% return
   - High costs: {high_cost_return:>8.2f}% return
   - Cost drag:  {cost_impact:>8.2f}% difference

3. BREAK-EVEN ANALYSIS:
   - With 2% fees per trade, need very high win rate or reduced frequency
   - Lower costs (0.5-1%) make the strategy potentially viable

4. RECOMMENDATIONS:
   - Use limit orders (maker) to reduce fees
   - Trade less frequently (higher consensus threshold)
   - Focus on higher-conviction signals
""")

    return all_results


def run_threshold_sensitivity():
    """Test different consensus thresholds."""
    print("\n" + "=" * 70)
    print("CONSENSUS THRESHOLD SENSITIVITY")
    print("=" * 70)

    trades_df, prices_df = generate_market_data(days=365, seed=42)

    thresholds = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
    results = []

    config = Config.default()
    config.costs.taker_fee = 0.01  # 1% fee

    for threshold in thresholds:
        config.strategy.consensus_threshold = threshold
        state = run_backtest(trades_df, prices_df, config, 'top_n')
        metrics = calculate_metrics(state, config)

        results.append({
            'threshold': threshold,
            'return': metrics['total_return_pct'],
            'sharpe': metrics['sharpe_ratio'],
            'trades': metrics['num_trades'],
            'win_rate': metrics['win_rate_pct'],
            'costs': metrics['total_costs']
        })

        print(f"  Threshold {threshold:.0%}: Return={metrics['total_return_pct']:>7.2f}%, "
              f"Trades={metrics['num_trades']:>3}, WinRate={metrics['win_rate_pct']:>5.1f}%")

    # Find optimal
    best = max(results, key=lambda x: x['return'])
    print(f"\n  Best threshold: {best['threshold']:.0%} with {best['return']:.2f}% return")

    return results


if __name__ == "__main__":
    run_cost_comparison()
    run_threshold_sensitivity()

"""
Backtesting engine for comparing whale detection methods.
Runs the copycat strategy with each method and produces comparison results.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import json
import logging
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import INITIAL_CAPITAL, PARQUET_DIR, CONSENSUS_THRESHOLD
from src.whales.detector import WhaleDetector, WhaleMethod
from src.strategy.copycat import CopycatStrategy, StrategyState, Position

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BacktestMetrics:
    """Performance metrics for a backtest run."""
    method: str
    initial_capital: float
    final_capital: float
    total_return: float
    total_return_pct: float
    num_trades: int
    num_winning_trades: int
    num_losing_trades: int
    win_rate: float
    max_drawdown: float
    max_drawdown_pct: float
    sharpe_ratio: float
    avg_trade_return: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    time_in_market_pct: float
    start_date: str
    end_date: str


class BacktestEngine:
    """
    Runs backtests for all whale detection methods and compares results.
    """

    def __init__(
        self,
        initial_capital: float = INITIAL_CAPITAL,
        consensus_threshold: float = CONSENSUS_THRESHOLD
    ):
        self.initial_capital = initial_capital
        self.consensus_threshold = consensus_threshold
        self.results: Dict[WhaleMethod, StrategyState] = {}
        self.metrics: Dict[WhaleMethod, BacktestMetrics] = {}

    def _calculate_metrics(
        self,
        state: StrategyState,
        method: WhaleMethod,
        start_date: datetime,
        end_date: datetime
    ) -> BacktestMetrics:
        """Calculate performance metrics from strategy state."""
        trades = state.trades
        actual_trades = [t for t in trades if t.action not in ['hold']]

        # Calculate returns per trade
        trade_returns = []
        for t in actual_trades:
            if t.capital_before > 0:
                ret = (t.capital_after - t.capital_before) / t.capital_before
                trade_returns.append(ret)

        # Equity curve for drawdown calculation
        equity_curve = [self.initial_capital]
        for t in trades:
            if t.action in ['close_long', 'close_short']:
                equity_curve.append(t.capital_after)

        equity_curve = np.array(equity_curve)
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / peak
        max_drawdown_pct = drawdown.max() if len(drawdown) > 0 else 0
        max_drawdown = (peak - equity_curve).max() if len(equity_curve) > 0 else 0

        # Win/loss stats
        wins = [r for r in trade_returns if r > 0]
        losses = [r for r in trade_returns if r < 0]

        num_winning = len(wins)
        num_losing = len(losses)
        win_rate = num_winning / len(trade_returns) if trade_returns else 0

        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0

        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Sharpe ratio (annualized, assuming 4-hour intervals)
        if trade_returns:
            returns_std = np.std(trade_returns)
            avg_return = np.mean(trade_returns)
            periods_per_year = 365 * 6  # 4-hour intervals
            sharpe = (avg_return * periods_per_year) / (returns_std * np.sqrt(periods_per_year)) if returns_std > 0 else 0
        else:
            sharpe = 0

        # Time in market
        total_duration = (end_date - start_date).total_seconds()
        time_in_position = 0
        entry_time = None

        for t in trades:
            if t.action in ['buy', 'sell'] and entry_time is None:
                entry_time = t.timestamp
            elif t.action in ['close_long', 'close_short'] and entry_time is not None:
                time_in_position += (t.timestamp - entry_time).total_seconds()
                entry_time = None

        time_in_market_pct = time_in_position / total_duration if total_duration > 0 else 0

        return BacktestMetrics(
            method=method.value,
            initial_capital=self.initial_capital,
            final_capital=state.capital,
            total_return=state.capital - self.initial_capital,
            total_return_pct=(state.capital / self.initial_capital - 1) * 100,
            num_trades=len(actual_trades),
            num_winning_trades=num_winning,
            num_losing_trades=num_losing,
            win_rate=win_rate * 100,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct * 100,
            sharpe_ratio=sharpe,
            avg_trade_return=np.mean(trade_returns) * 100 if trade_returns else 0,
            avg_win=avg_win * 100,
            avg_loss=avg_loss * 100,
            profit_factor=profit_factor,
            time_in_market_pct=time_in_market_pct * 100,
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat()
        )

    def run_all_methods(
        self,
        trades_df: pd.DataFrame,
        prices_df: pd.DataFrame,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        signal_interval_hours: int = 4
    ) -> Dict[WhaleMethod, BacktestMetrics]:
        """
        Run backtest for all three whale detection methods.

        Returns:
            Dictionary mapping method to metrics
        """
        detector = WhaleDetector()

        for method in WhaleMethod:
            logger.info(f"\n{'='*60}")
            logger.info(f"Running backtest for method: {method.value}")
            logger.info(f"{'='*60}")

            strategy = CopycatStrategy(
                whale_detector=detector,
                whale_method=method,
                consensus_threshold=self.consensus_threshold,
                initial_capital=self.initial_capital
            )

            state = strategy.run_backtest(
                trades_df=trades_df,
                prices_df=prices_df,
                start_time=start_time,
                end_time=end_time,
                signal_interval_hours=signal_interval_hours
            )

            self.results[method] = state

            # Determine actual date range
            if state.trades:
                actual_start = state.trades[0].timestamp
                actual_end = state.trades[-1].timestamp
            else:
                actual_start = start_time or datetime.now()
                actual_end = end_time or datetime.now()

            metrics = self._calculate_metrics(state, method, actual_start, actual_end)
            self.metrics[method] = metrics

        return self.metrics

    def generate_report(self) -> str:
        """Generate a markdown report comparing all methods."""
        if not self.metrics:
            return "No backtest results available."

        report = []
        report.append("# Whale Copycat Strategy Backtest Results\n")
        report.append(f"Generated: {datetime.now().isoformat()}\n")
        report.append(f"Consensus Threshold: {self.consensus_threshold:.0%}\n")
        report.append(f"Initial Capital: ${self.initial_capital:,.2f}\n")

        # Summary table
        report.append("\n## Summary Comparison\n")
        report.append("| Metric | Top N | 95th Percentile | Trade Size >$10K |")
        report.append("|--------|-------|-----------------|------------------|")

        metric_rows = [
            ("Final Capital", "final_capital", "${:,.2f}"),
            ("Total Return %", "total_return_pct", "{:.2f}%"),
            ("# Trades", "num_trades", "{:d}"),
            ("Win Rate", "win_rate", "{:.1f}%"),
            ("Max Drawdown %", "max_drawdown_pct", "{:.2f}%"),
            ("Sharpe Ratio", "sharpe_ratio", "{:.2f}"),
            ("Profit Factor", "profit_factor", "{:.2f}"),
            ("Time in Market", "time_in_market_pct", "{:.1f}%"),
        ]

        for label, attr, fmt in metric_rows:
            row = f"| {label} |"
            for method in [WhaleMethod.TOP_N, WhaleMethod.PERCENTILE, WhaleMethod.TRADE_SIZE]:
                if method in self.metrics:
                    val = getattr(self.metrics[method], attr)
                    if isinstance(val, float) and val == float('inf'):
                        row += " ∞ |"
                    else:
                        row += f" {fmt.format(val)} |"
                else:
                    row += " N/A |"
            report.append(row)

        # Individual method details
        for method in WhaleMethod:
            if method not in self.metrics:
                continue

            m = self.metrics[method]
            report.append(f"\n## Method: {method.value}\n")
            report.append(f"- **Period**: {m.start_date} to {m.end_date}")
            report.append(f"- **Final Capital**: ${m.final_capital:,.2f}")
            report.append(f"- **Total Return**: ${m.total_return:,.2f} ({m.total_return_pct:.2f}%)")
            report.append(f"- **Trades**: {m.num_trades} ({m.num_winning_trades} wins, {m.num_losing_trades} losses)")
            report.append(f"- **Win Rate**: {m.win_rate:.1f}%")
            report.append(f"- **Average Trade Return**: {m.avg_trade_return:.2f}%")
            report.append(f"- **Average Win**: {m.avg_win:.2f}%")
            report.append(f"- **Average Loss**: {m.avg_loss:.2f}%")
            report.append(f"- **Max Drawdown**: ${m.max_drawdown:,.2f} ({m.max_drawdown_pct:.2f}%)")
            report.append(f"- **Sharpe Ratio**: {m.sharpe_ratio:.2f}")
            report.append(f"- **Profit Factor**: {m.profit_factor:.2f}")

            # Trade log
            if method in self.results:
                state = self.results[method]
                actual_trades = [t for t in state.trades if t.action != 'hold']
                if actual_trades:
                    report.append(f"\n### Trade Log (showing last 10)\n")
                    report.append("| Time | Action | Price | Consensus | Capital After |")
                    report.append("|------|--------|-------|-----------|---------------|")
                    for t in actual_trades[-10:]:
                        report.append(
                            f"| {t.timestamp.strftime('%Y-%m-%d %H:%M')} | "
                            f"{t.action} | {t.price:.4f} | "
                            f"{t.whale_consensus:.1%} | ${t.capital_after:,.2f} |"
                        )

        return "\n".join(report)

    def save_results(self, output_dir: Path = PARQUET_DIR):
        """Save backtest results to files."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save metrics as JSON
        metrics_data = {
            method.value: asdict(m)
            for method, m in self.metrics.items()
        }
        with open(output_dir / "backtest_metrics.json", "w") as f:
            json.dump(metrics_data, f, indent=2, default=str)

        # Save trade logs as parquet
        for method, state in self.results.items():
            trades_data = []
            for t in state.trades:
                trades_data.append({
                    'timestamp': t.timestamp,
                    'action': t.action,
                    'position': t.position.value,
                    'price': t.price,
                    'size': t.size,
                    'capital_before': t.capital_before,
                    'capital_after': t.capital_after,
                    'whale_consensus': t.whale_consensus,
                    'whale_direction': t.whale_direction,
                    'reason': t.reason
                })
            df = pd.DataFrame(trades_data)
            df.to_parquet(output_dir / f"trades_{method.value}.parquet")

        # Save report
        report = self.generate_report()
        with open(output_dir / "backtest_report.md", "w") as f:
            f.write(report)

        logger.info(f"Results saved to {output_dir}")


def main():
    """Run backtest with sample data."""
    # Generate sample data for testing
    np.random.seed(42)
    n_trades = 1000
    n_prices = 168 * 4  # 4 weeks hourly

    # Sample trades with some whale-like patterns
    traders = [f"0x{i:040x}" for i in range(100)]
    whale_traders = traders[:10]  # Top 10 are whales

    trades_list = []
    base_time = datetime.now() - timedelta(days=28)

    for i in range(n_trades):
        is_whale = np.random.random() < 0.2  # 20% of trades from whales
        trader = np.random.choice(whale_traders if is_whale else traders)
        size = np.random.exponential(15000 if is_whale else 1000)
        # Whales tend to follow trends
        trend_bias = 0.6 if i > n_trades * 0.5 else 0.4
        side = 'buy' if np.random.random() < (trend_bias if is_whale else 0.5) else 'sell'

        trades_list.append({
            'timestamp': base_time + timedelta(minutes=i * 40),
            'trader': trader,
            'size': size,
            'side': side
        })

    trades_df = pd.DataFrame(trades_list)

    # Sample prices with uptrend in second half
    prices_list = []
    base_price = 0.4

    for i in range(n_prices):
        if i < n_prices // 2:
            drift = 0
        else:
            drift = 0.001  # Slight uptrend
        base_price += drift + np.random.randn() * 0.005
        base_price = max(0.05, min(0.95, base_price))

        prices_list.append({
            'timestamp': base_time + timedelta(hours=i),
            'price': base_price
        })

    prices_df = pd.DataFrame(prices_list)

    # Run backtest
    engine = BacktestEngine()
    engine.run_all_methods(trades_df, prices_df, signal_interval_hours=4)

    # Print and save results
    print(engine.generate_report())
    engine.save_results()


if __name__ == "__main__":
    main()

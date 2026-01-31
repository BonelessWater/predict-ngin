"""
Bias-free backtesting engine for whale following strategies.

Addresses potential biases:
1. Look-ahead bias: Train/test split with temporal ordering
2. Survivorship bias: All markets included
3. Selection bias: Whales identified from training period only
4. Execution bias: Entry at probAfter (post-whale) price
5. Multiple entry bias: One position per market
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Set, List, Any, Optional
from scipy import stats

from .costs import CostModel, DEFAULT_COST_MODEL


@dataclass
class BacktestResult:
    """Results from a backtest run."""

    strategy_name: str
    total_trades: int
    unique_markets: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_gross_pnl: float
    total_net_pnl: float
    total_costs: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    roi_pct: float
    trades_df: pd.DataFrame = field(repr=False)
    daily_returns: pd.Series = field(repr=False)

    def statistical_significance(self) -> Dict[str, float]:
        """Calculate statistical significance metrics."""
        if len(self.daily_returns) < 10:
            return {"t_stat": 0, "p_value": 1, "binom_p": 1}

        # T-test for returns
        t_stat, p_value = stats.ttest_1samp(self.daily_returns, 0)

        # Binomial test for win rate
        binom_result = stats.binomtest(
            self.winning_trades,
            self.total_trades,
            0.5,
            alternative="greater"
        )

        return {
            "t_stat": t_stat,
            "p_value": p_value,
            "binom_p": binom_result.pvalue,
        }


def run_backtest(
    test_df: pd.DataFrame,
    whale_set: Set[str],
    resolution_data: Dict[str, Dict[str, Any]],
    strategy_name: str = "Whale Strategy",
    cost_model: Optional[CostModel] = None,
    position_size: float = 100,
    starting_capital: float = 10000,
    min_liquidity: float = 0,
    category_filter: Optional[str] = None,
) -> Optional[BacktestResult]:
    """
    Run bias-free backtest of whale following strategy.

    Args:
        test_df: Test period bets DataFrame
        whale_set: Set of whale user IDs
        resolution_data: Market resolution mapping
        strategy_name: Name for this strategy
        cost_model: Cost model (default: small retail)
        position_size: Fixed position size in dollars
        starting_capital: Starting capital for ROI calculation
        min_liquidity: Minimum market liquidity to trade
        category_filter: Only trade this category (if set)

    Returns:
        BacktestResult or None if no valid trades
    """
    if cost_model is None:
        cost_model = DEFAULT_COST_MODEL

    # Filter to whale trades
    whale_trades = test_df[test_df["userId"].isin(whale_set)].copy()
    whale_trades = whale_trades[whale_trades["outcome"].isin(["YES", "NO"])]
    whale_trades = whale_trades[whale_trades["probAfter"].notna()]
    whale_trades = whale_trades.sort_values("datetime")

    if len(whale_trades) == 0:
        return None

    # Track positions - one per market
    positions: Dict[str, bool] = {}
    trades: List[Dict[str, Any]] = []

    for _, trade in whale_trades.iterrows():
        contract_id = trade["contractId"]

        # One position per market
        if contract_id in positions:
            continue

        # Must be resolved
        if contract_id not in resolution_data:
            continue

        res_data = resolution_data[contract_id]
        resolution = res_data["resolution"]
        liquidity = res_data.get("liquidity", 1000)

        # Check liquidity
        if liquidity < min_liquidity:
            continue

        # Category filter
        if category_filter:
            category = res_data.get("category", "other")
            if category != category_filter:
                continue

        # Must trade before resolution
        trade_time = trade["createdTime"]
        res_time = res_data["resolved_time"]
        if res_time and trade_time >= res_time:
            continue

        outcome = trade["outcome"]
        prob_after = trade["probAfter"]

        # Calculate entry/exit prices
        if outcome == "YES":
            entry_price = prob_after
            exit_price = resolution
        else:
            entry_price = 1 - prob_after
            exit_price = 1 - resolution

        # Skip extreme prices
        if entry_price < 0.02 or entry_price > 0.98:
            continue

        # Apply costs
        entry_cost = cost_model.calculate_entry_price(entry_price, position_size, liquidity)
        exit_proceeds = cost_model.calculate_exit_price(exit_price, position_size, liquidity)

        gross_pnl = (exit_price - entry_price) * position_size
        net_pnl = (exit_proceeds - entry_cost) * position_size

        positions[contract_id] = True

        # Sanitize question for output
        question = res_data.get("question", "")[:60]
        question = question.encode("ascii", "replace").decode("ascii")

        trades.append({
            "datetime": trade["datetime"],
            "date": trade["date"],
            "contract_id": contract_id,
            "whale_id": trade["userId"],
            "outcome": outcome,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "gross_pnl": gross_pnl,
            "net_pnl": net_pnl,
            "question": question,
        })

    if not trades:
        return None

    trades_df = pd.DataFrame(trades)
    trades_df = trades_df.sort_values("datetime")
    trades_df["cumulative_pnl"] = trades_df["net_pnl"].cumsum()

    # Daily returns for Sharpe
    daily_pnl = trades_df.groupby("date")["net_pnl"].sum()
    daily_returns = daily_pnl / starting_capital

    # Calculate metrics
    total_trades = len(trades_df)
    winners = trades_df[trades_df["net_pnl"] > 0]
    losers = trades_df[trades_df["net_pnl"] <= 0]

    win_rate = len(winners) / total_trades if total_trades > 0 else 0
    total_gross = trades_df["gross_pnl"].sum()
    total_net = trades_df["net_pnl"].sum()
    total_costs = total_gross - total_net

    avg_win = winners["net_pnl"].mean() if len(winners) > 0 else 0
    avg_loss = losers["net_pnl"].mean() if len(losers) > 0 else 0

    # Profit factor
    gross_wins = winners["net_pnl"].sum() if len(winners) > 0 else 0
    gross_losses = abs(losers["net_pnl"].sum()) if len(losers) > 0 else 1
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf")

    # Sharpe ratio (annualized)
    if len(daily_returns) > 1 and daily_returns.std() > 0:
        sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
    else:
        sharpe = 0

    # Max drawdown
    cummax = trades_df["cumulative_pnl"].cummax()
    drawdown = cummax - trades_df["cumulative_pnl"]
    max_dd = drawdown.max()

    # ROI
    roi = total_net / starting_capital * 100

    return BacktestResult(
        strategy_name=strategy_name,
        total_trades=total_trades,
        unique_markets=trades_df["contract_id"].nunique(),
        winning_trades=len(winners),
        losing_trades=len(losers),
        win_rate=win_rate,
        total_gross_pnl=total_gross,
        total_net_pnl=total_net,
        total_costs=total_costs,
        avg_win=avg_win,
        avg_loss=avg_loss,
        profit_factor=profit_factor,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        roi_pct=roi,
        trades_df=trades_df,
        daily_returns=daily_returns,
    )


def print_result(result: BacktestResult) -> None:
    """Print formatted backtest results."""
    print(f"\n{'='*60}")
    print(f"STRATEGY: {result.strategy_name}")
    print(f"{'='*60}")
    print(f"Unique Markets Traded:  {result.unique_markets:,}")
    print(f"Total Trades:           {result.total_trades:,}")
    print(f"Win Rate:               {result.win_rate*100:.1f}%")
    print()
    print(f"Gross P&L:              ${result.total_gross_pnl:>12,.2f}")
    print(f"Trading Costs:          ${result.total_costs:>12,.2f}")
    print(f"Net P&L:                ${result.total_net_pnl:>12,.2f}")
    print(f"ROI:                    {result.roi_pct:>12.2f}%")
    print()
    print(f"Avg Winning Trade:      ${result.avg_win:>12,.2f}")
    print(f"Avg Losing Trade:       ${result.avg_loss:>12,.2f}")
    print(f"Profit Factor:          {result.profit_factor:>12.2f}")
    print()
    print(f"Sharpe Ratio:           {result.sharpe_ratio:>12.2f}")
    print(f"Max Drawdown:           ${result.max_drawdown:>12,.2f}")

    # Statistical significance
    sig = result.statistical_significance()
    print()
    print(f"T-statistic:            {sig['t_stat']:>12.3f}")
    print(f"P-value (returns):      {sig['p_value']:>12.4f}")
    print(f"P-value (win rate):     {sig['binom_p']:>12.4f}")

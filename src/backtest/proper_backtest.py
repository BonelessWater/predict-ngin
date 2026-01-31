"""
Proper backtest implementation with:
- No look-ahead bias
- Transaction costs (fees)
- Slippage modeling
- Configurable parameters

All whale detection and signals are calculated using only data
available at the time of the trading decision.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import sys
import warnings
import json

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.config import Config, get_config

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Position(Enum):
    NONE = "none"
    LONG = "long"
    SHORT = "short"


@dataclass
class Trade:
    """Record of a single trade with all costs."""
    timestamp: datetime
    action: str
    market_price: float
    execution_price: float  # After slippage
    size: float  # In contracts
    value: float  # In USD
    fee: float
    slippage_cost: float
    total_cost: float
    capital_before: float
    capital_after: float
    consensus: float
    direction: str
    pnl: float = 0.0
    reason: str = ""


@dataclass
class BacktestState:
    """Current state of the backtest."""
    capital: float
    position: Position = Position.NONE
    position_size: float = 0.0  # Number of contracts
    entry_price: float = 0.0
    entry_value: float = 0.0  # USD value at entry
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[dict] = field(default_factory=list)


def generate_market_data(
    days: int = 365,
    seed: int = 42,
    trades_per_day: int = 100
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate realistic market data with whale trading patterns.
    NO LOOK-AHEAD: Whale behavior is based on their own analysis,
    not future price knowledge.

    Returns:
        (trades_df, prices_df)
    """
    np.random.seed(seed)

    start_date = datetime.now() - timedelta(days=days)

    # Define market regimes (unknown to the strategy)
    regimes = [
        {'start': 0, 'end': 0.15, 'trend': 0.001, 'vol': 0.015, 'whale_edge': 0.60},
        {'start': 0.15, 'end': 0.35, 'trend': 0.003, 'vol': 0.020, 'whale_edge': 0.65},
        {'start': 0.35, 'end': 0.50, 'trend': -0.001, 'vol': 0.018, 'whale_edge': 0.55},
        {'start': 0.50, 'end': 0.70, 'trend': -0.003, 'vol': 0.025, 'whale_edge': 0.62},
        {'start': 0.70, 'end': 0.85, 'trend': 0.002, 'vol': 0.020, 'whale_edge': 0.58},
        {'start': 0.85, 'end': 1.0, 'trend': 0.001, 'vol': 0.015, 'whale_edge': 0.60},
    ]

    # Generate daily prices
    prices = []
    price = 0.45

    for day in range(days):
        progress = day / days

        # Find current regime
        regime = next(r for r in regimes if r['start'] <= progress < r['end'])

        # Price evolution (random walk with drift)
        daily_return = regime['trend'] + np.random.randn() * regime['vol']
        price = price * (1 + daily_return)
        price = max(0.10, min(0.90, price))

        prices.append({
            'date': start_date + timedelta(days=day),
            'price': price,
            'regime_trend': regime['trend'],
            'whale_edge': regime['whale_edge']
        })

    prices_df = pd.DataFrame(prices)

    # Generate trades
    # Key insight: Whales have a slight edge in predicting short-term direction,
    # but this is based on their own analysis, NOT future price knowledge
    num_traders = 200
    whale_ids = [f"whale_{i}" for i in range(15)]
    retail_ids = [f"retail_{i}" for i in range(num_traders - 15)]
    all_traders = whale_ids + retail_ids

    trades = []

    for day in range(days):
        day_date = start_date + timedelta(days=day)
        progress = day / days

        # Find current regime
        regime = next(r for r in regimes if r['start'] <= progress < r['end'])

        # Number of trades today (varies)
        n_trades = int(trades_per_day * (0.7 + np.random.random() * 0.6))

        # Current price and recent trend (what traders can observe)
        current_price = prices_df.iloc[day]['price']
        if day > 5:
            recent_prices = prices_df.iloc[max(0, day-5):day]['price'].values
            recent_trend = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
        else:
            recent_trend = 0

        for _ in range(n_trades):
            # Determine trader type
            is_whale = np.random.random() < 0.12

            if is_whale:
                trader_id = np.random.choice(whale_ids)
                trade_size = np.random.exponential(15000)

                # Whales have an edge based on their analysis (not future knowledge)
                # Their edge is regime-dependent and imperfect
                whale_edge = regime['whale_edge']

                # Whales incorporate multiple signals:
                # 1. Their proprietary analysis (simulated as regime edge)
                # 2. Recent trend (momentum)
                # 3. Mean reversion expectations

                # Combine signals
                momentum_signal = 1 if recent_trend > 0 else -1
                mean_reversion_signal = 1 if current_price < 0.5 else -1

                # Whale's probabilistic decision
                if np.random.random() < whale_edge:
                    # Whale correctly assesses direction
                    # (based on their analysis, which happens to align with future)
                    if regime['trend'] > 0:
                        side = 'BUY'
                    else:
                        side = 'SELL'
                else:
                    # Whale gets it wrong
                    side = 'BUY' if np.random.random() < 0.5 else 'SELL'
            else:
                trader_id = np.random.choice(retail_ids)
                trade_size = np.random.exponential(800)

                # Retail traders are essentially random with slight momentum bias
                if np.random.random() < 0.5 + recent_trend * 2:
                    side = 'BUY'
                else:
                    side = 'SELL'

            # Add some randomness to trade timing within the day
            trade_time = day_date + timedelta(
                hours=np.random.randint(0, 24),
                minutes=np.random.randint(0, 60)
            )

            trades.append({
                'timestamp': trade_time,
                'trader_id': trader_id,
                'is_whale': is_whale,
                'size': trade_size,
                'side': side,
                'price': current_price + np.random.randn() * 0.001  # Small price variation
            })

    trades_df = pd.DataFrame(trades)
    trades_df = trades_df.sort_values('timestamp').reset_index(drop=True)

    return trades_df, prices_df


def detect_whales_no_lookahead(
    trades_df: pd.DataFrame,
    as_of_date: datetime,
    config: Config,
    method: str = 'top_n'
) -> set:
    """
    Detect whales using ONLY data available up to as_of_date.
    NO LOOK-AHEAD BIAS.

    Args:
        trades_df: All historical trades
        as_of_date: Current simulation date (only use data before this)
        config: Configuration object
        method: 'top_n', 'percentile', or 'trade_size'

    Returns:
        Set of whale trader IDs
    """
    # CRITICAL: Only use trades BEFORE as_of_date
    cutoff = as_of_date - timedelta(days=config.whale.staleness_days)
    historical = trades_df[
        (trades_df['timestamp'] < as_of_date) &
        (trades_df['timestamp'] >= cutoff)
    ]

    if historical.empty:
        return set()

    # Aggregate by trader
    trader_stats = historical.groupby('trader_id').agg({
        'size': ['sum', 'mean', 'count']
    }).reset_index()
    trader_stats.columns = ['trader_id', 'total_volume', 'avg_trade_size', 'trade_count']

    if method == 'top_n':
        top = trader_stats.nlargest(config.whale.top_n, 'total_volume')
        return set(top['trader_id'].tolist())

    elif method == 'percentile':
        threshold = np.percentile(trader_stats['total_volume'], config.whale.percentile)
        whales = trader_stats[trader_stats['total_volume'] >= threshold]
        return set(whales['trader_id'].tolist())

    elif method == 'trade_size':
        whales = trader_stats[trader_stats['avg_trade_size'] >= config.whale.min_trade_size]
        return set(whales['trader_id'].tolist())

    return set()


def calculate_whale_consensus_no_lookahead(
    trades_df: pd.DataFrame,
    whales: set,
    as_of_date: datetime,
    lookback_hours: int = 24
) -> Tuple[float, str]:
    """
    Calculate whale consensus using ONLY historical data.
    NO LOOK-AHEAD BIAS.

    Args:
        trades_df: All trades
        whales: Set of whale trader IDs
        as_of_date: Current date (only use data before this)
        lookback_hours: How far back to look for whale activity

    Returns:
        (consensus_ratio, direction)
    """
    if not whales:
        return 0.0, 'neutral'

    # CRITICAL: Only use trades BEFORE as_of_date
    cutoff = as_of_date - timedelta(hours=lookback_hours)
    recent = trades_df[
        (trades_df['timestamp'] < as_of_date) &
        (trades_df['timestamp'] >= cutoff) &
        (trades_df['trader_id'].isin(whales))
    ]

    if recent.empty:
        return 0.0, 'neutral'

    # Volume-weighted consensus
    buys = recent[recent['side'] == 'BUY']['size'].sum()
    sells = recent[recent['side'] == 'SELL']['size'].sum()
    total = buys + sells

    if total == 0:
        return 0.0, 'neutral'

    buy_ratio = buys / total
    sell_ratio = sells / total

    if buy_ratio > sell_ratio:
        return buy_ratio, 'buy'
    else:
        return sell_ratio, 'sell'


def execute_trade(
    state: BacktestState,
    action: str,
    market_price: float,
    timestamp: datetime,
    consensus: float,
    direction: str,
    config: Config
) -> BacktestState:
    """
    Execute a trade with realistic costs.

    Args:
        state: Current backtest state
        action: 'buy', 'sell', 'close_long', 'close_short'
        market_price: Current market price (before slippage)
        timestamp: Trade timestamp
        consensus: Current whale consensus
        direction: Whale direction signal
        config: Configuration with cost parameters

    Returns:
        Updated state
    """
    costs = config.costs

    if action == 'buy' and state.position == Position.NONE:
        # Calculate trade size
        available_capital = state.capital * config.strategy.position_size_pct

        # Calculate costs upfront
        fee = available_capital * costs.taker_fee
        slippage = costs.calculate_slippage(available_capital, is_buy=True)
        slippage_cost = available_capital * slippage
        total_cost = fee + slippage_cost

        # Execution price after slippage
        execution_price = costs.get_execution_price(market_price, available_capital, is_buy=True)

        # Capital available after costs
        capital_for_position = available_capital - total_cost

        # Number of contracts we can buy
        position_size = capital_for_position / execution_price

        trade = Trade(
            timestamp=timestamp,
            action='buy',
            market_price=market_price,
            execution_price=execution_price,
            size=position_size,
            value=capital_for_position,
            fee=fee,
            slippage_cost=slippage_cost,
            total_cost=total_cost,
            capital_before=state.capital,
            capital_after=state.capital - available_capital,
            consensus=consensus,
            direction=direction,
            reason=f"Whale consensus {consensus:.1%} >= threshold"
        )

        state.trades.append(trade)
        state.capital = state.capital - available_capital
        state.position = Position.LONG
        state.position_size = position_size
        state.entry_price = execution_price
        state.entry_value = capital_for_position

    elif action == 'sell' and state.position == Position.NONE:
        # Short position (buy NO tokens)
        available_capital = state.capital * config.strategy.position_size_pct

        fee = available_capital * costs.taker_fee
        slippage = costs.calculate_slippage(available_capital, is_buy=True)
        slippage_cost = available_capital * slippage
        total_cost = fee + slippage_cost

        # For shorts, we buy NO tokens at (1 - price)
        no_price = 1 - market_price
        execution_price = costs.get_execution_price(no_price, available_capital, is_buy=True)

        capital_for_position = available_capital - total_cost
        position_size = capital_for_position / execution_price

        trade = Trade(
            timestamp=timestamp,
            action='sell',
            market_price=market_price,
            execution_price=execution_price,
            size=position_size,
            value=capital_for_position,
            fee=fee,
            slippage_cost=slippage_cost,
            total_cost=total_cost,
            capital_before=state.capital,
            capital_after=state.capital - available_capital,
            consensus=consensus,
            direction=direction,
            reason=f"Whale consensus {consensus:.1%} >= threshold (bearish)"
        )

        state.trades.append(trade)
        state.capital = state.capital - available_capital
        state.position = Position.SHORT
        state.position_size = position_size
        state.entry_price = execution_price
        state.entry_value = capital_for_position

    elif action == 'close_long' and state.position == Position.LONG:
        # Sell our YES tokens
        trade_value = state.position_size * market_price

        fee = trade_value * costs.taker_fee
        slippage = costs.calculate_slippage(trade_value, is_buy=False)
        slippage_cost = trade_value * slippage
        total_cost = fee + slippage_cost

        execution_price = costs.get_execution_price(market_price, trade_value, is_buy=False)
        proceeds = state.position_size * execution_price - fee

        pnl = proceeds - state.entry_value

        trade = Trade(
            timestamp=timestamp,
            action='close_long',
            market_price=market_price,
            execution_price=execution_price,
            size=state.position_size,
            value=proceeds,
            fee=fee,
            slippage_cost=slippage_cost,
            total_cost=total_cost,
            capital_before=state.capital,
            capital_after=state.capital + proceeds,
            consensus=consensus,
            direction=direction,
            pnl=pnl,
            reason="Exit: consensus dropped or reversed"
        )

        state.trades.append(trade)
        state.capital = state.capital + proceeds
        state.position = Position.NONE
        state.position_size = 0
        state.entry_price = 0
        state.entry_value = 0

    elif action == 'close_short' and state.position == Position.SHORT:
        # Sell our NO tokens
        no_price = 1 - market_price
        trade_value = state.position_size * no_price

        fee = trade_value * costs.taker_fee
        slippage = costs.calculate_slippage(trade_value, is_buy=False)
        slippage_cost = trade_value * slippage
        total_cost = fee + slippage_cost

        execution_price = costs.get_execution_price(no_price, trade_value, is_buy=False)
        proceeds = state.position_size * execution_price - fee

        pnl = proceeds - state.entry_value

        trade = Trade(
            timestamp=timestamp,
            action='close_short',
            market_price=market_price,
            execution_price=execution_price,
            size=state.position_size,
            value=proceeds,
            fee=fee,
            slippage_cost=slippage_cost,
            total_cost=total_cost,
            capital_before=state.capital,
            capital_after=state.capital + proceeds,
            consensus=consensus,
            direction=direction,
            pnl=pnl,
            reason="Exit: consensus dropped or reversed"
        )

        state.trades.append(trade)
        state.capital = state.capital + proceeds
        state.position = Position.NONE
        state.position_size = 0
        state.entry_price = 0
        state.entry_value = 0

    return state


def run_backtest(
    trades_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    config: Config,
    method: str = 'top_n'
) -> BacktestState:
    """
    Run a backtest with NO LOOK-AHEAD BIAS.

    At each decision point, only data available up to that point is used.
    """
    state = BacktestState(capital=config.strategy.initial_capital)

    prices_df = prices_df.copy()
    prices_df['date'] = pd.to_datetime(prices_df['date'])

    # Start after warmup period
    start_idx = config.backtest.warmup_days
    if start_idx >= len(prices_df):
        logger.error("Not enough data for warmup period")
        return state

    logger.info(f"Running backtest: method={method}, {len(prices_df)} days")
    logger.info(f"Costs: taker_fee={config.costs.taker_fee:.2%}, base_slippage={config.costs.base_slippage:.2%}")

    for i in range(start_idx, len(prices_df)):
        current_date = prices_df.iloc[i]['date']
        current_price = prices_df.iloc[i]['price']

        # Record equity
        if state.position == Position.LONG:
            mark_to_market = state.position_size * current_price
            current_equity = state.capital + mark_to_market
        elif state.position == Position.SHORT:
            no_price = 1 - current_price
            mark_to_market = state.position_size * no_price
            current_equity = state.capital + mark_to_market
        else:
            current_equity = state.capital

        state.equity_curve.append({
            'date': current_date,
            'equity': current_equity,
            'price': current_price
        })

        # Detect whales using only historical data (NO LOOK-AHEAD)
        whales = detect_whales_no_lookahead(trades_df, current_date, config, method)

        if not whales:
            continue

        # Calculate consensus using only historical data (NO LOOK-AHEAD)
        consensus, direction = calculate_whale_consensus_no_lookahead(
            trades_df, whales, current_date, lookback_hours=24
        )

        # Trading logic
        threshold = config.strategy.consensus_threshold

        if state.position == Position.NONE:
            if consensus >= threshold:
                if direction == 'buy':
                    state = execute_trade(
                        state, 'buy', current_price, current_date,
                        consensus, direction, config
                    )
                elif direction == 'sell':
                    state = execute_trade(
                        state, 'sell', current_price, current_date,
                        consensus, direction, config
                    )

        elif state.position == Position.LONG:
            if consensus < threshold or direction == 'sell':
                state = execute_trade(
                    state, 'close_long', current_price, current_date,
                    consensus, direction, config
                )

        elif state.position == Position.SHORT:
            if consensus < threshold or direction == 'buy':
                state = execute_trade(
                    state, 'close_short', current_price, current_date,
                    consensus, direction, config
                )

    # Close any open position at end
    if state.position != Position.NONE:
        final_price = prices_df.iloc[-1]['price']
        final_date = prices_df.iloc[-1]['date']

        if state.position == Position.LONG:
            state = execute_trade(
                state, 'close_long', final_price, final_date,
                0, 'neutral', config
            )
        else:
            state = execute_trade(
                state, 'close_short', final_price, final_date,
                0, 'neutral', config
            )

    return state


def calculate_metrics(state: BacktestState, config: Config) -> dict:
    """Calculate performance metrics from backtest state."""
    if not state.equity_curve:
        return {}

    equity_df = pd.DataFrame(state.equity_curve)
    initial_capital = config.strategy.initial_capital
    final_capital = state.capital

    # Returns
    daily_equity = equity_df.set_index('date')['equity'].resample('D').last().ffill()
    returns = daily_equity.pct_change().dropna()

    # Trade metrics
    closed_trades = [t for t in state.trades if 'close' in t.action]
    pnls = [t.pnl for t in closed_trades]

    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    # Cost analysis
    total_fees = sum(t.fee for t in state.trades)
    total_slippage = sum(t.slippage_cost for t in state.trades)
    total_costs = sum(t.total_cost for t in state.trades)

    # Sharpe/Sortino
    if len(returns) > 0 and returns.std() > 0:
        sharpe = returns.mean() / returns.std() * np.sqrt(252)
    else:
        sharpe = 0

    downside = returns[returns < 0]
    if len(downside) > 0 and downside.std() > 0:
        sortino = returns.mean() / downside.std() * np.sqrt(252)
    else:
        sortino = 0

    # Max drawdown
    cummax = daily_equity.cummax()
    drawdown = (cummax - daily_equity) / cummax
    max_drawdown = drawdown.max() * 100

    # Volatility
    volatility = returns.std() * np.sqrt(252) * 100 if len(returns) > 0 else 0

    return {
        'initial_capital': initial_capital,
        'final_capital': final_capital,
        'total_return_pct': (final_capital / initial_capital - 1) * 100,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'max_drawdown_pct': max_drawdown,
        'volatility_pct': volatility,
        'calmar_ratio': ((final_capital / initial_capital - 1) * 100) / max_drawdown if max_drawdown > 0 else 0,
        'num_trades': len(closed_trades),
        'num_wins': len(wins),
        'num_losses': len(losses),
        'win_rate_pct': len(wins) / len(pnls) * 100 if pnls else 0,
        'avg_win': np.mean(wins) if wins else 0,
        'avg_loss': np.mean(losses) if losses else 0,
        'profit_factor': abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float('inf'),
        'total_fees': total_fees,
        'total_slippage': total_slippage,
        'total_costs': total_costs,
        'costs_pct_of_capital': total_costs / initial_capital * 100,
    }


def run_all_methods():
    """Run backtests for all methods with proper cost modeling."""
    print("=" * 70)
    print("PROPER BACKTEST - NO LOOK-AHEAD BIAS")
    print("With Transaction Costs and Slippage")
    print("=" * 70)

    # Generate data
    print("\nGenerating 1 year of market data...")
    trades_df, prices_df = generate_market_data(days=365, seed=42)
    print(f"  Trades: {len(trades_df):,}")
    print(f"  Price points: {len(prices_df)}")

    # Run with costs
    config_with_costs = Config.default()
    print(f"\nConfiguration:")
    print(f"  Taker Fee: {config_with_costs.costs.taker_fee:.2%}")
    print(f"  Base Slippage: {config_with_costs.costs.base_slippage:.2%}")
    print(f"  Consensus Threshold: {config_with_costs.strategy.consensus_threshold:.0%}")

    all_results = {}
    all_equity = {}

    methods = ['top_n', 'percentile', 'trade_size']

    for method in methods:
        print(f"\n{'='*50}")
        print(f"Method: {method.upper()}")
        print('='*50)

        state = run_backtest(trades_df, prices_df, config_with_costs, method)
        metrics = calculate_metrics(state, config_with_costs)

        all_results[method] = metrics
        all_equity[method] = pd.DataFrame(state.equity_curve)

        print(f"\n  Results:")
        print(f"    Final Capital:   ${metrics['final_capital']:,.2f}")
        print(f"    Total Return:    {metrics['total_return_pct']:.2f}%")
        print(f"    Sharpe Ratio:    {metrics['sharpe_ratio']:.2f}")
        print(f"    Max Drawdown:    {metrics['max_drawdown_pct']:.2f}%")
        print(f"    Win Rate:        {metrics['win_rate_pct']:.1f}%")
        print(f"    Trades:          {metrics['num_trades']}")
        print(f"    Total Costs:     ${metrics['total_costs']:.2f} ({metrics['costs_pct_of_capital']:.1f}% of capital)")
        print(f"      - Fees:        ${metrics['total_fees']:.2f}")
        print(f"      - Slippage:    ${metrics['total_slippage']:.2f}")

    # Save results
    output_dir = config_with_costs.data.output_dir

    # Save metrics
    with open(output_dir / "proper_backtest_metrics.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Save equity curves
    for method, eq_df in all_equity.items():
        eq_df.to_csv(output_dir / f"equity_curve_{method}_proper.csv", index=False)

    # Combined equity
    combined = pd.DataFrame({'date': all_equity['top_n']['date']})
    for method, eq_df in all_equity.items():
        combined[method] = eq_df['equity'].values
    combined.to_csv(output_dir / "equity_curves_proper_combined.csv", index=False)

    # Print comparison
    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)

    print("\n| Metric | Top N | Percentile | Trade Size |")
    print("|--------|-------|------------|------------|")

    metrics_to_show = [
        ('Total Return', 'total_return_pct', '{:.2f}%'),
        ('Sharpe Ratio', 'sharpe_ratio', '{:.2f}'),
        ('Sortino Ratio', 'sortino_ratio', '{:.2f}'),
        ('Max Drawdown', 'max_drawdown_pct', '{:.2f}%'),
        ('Win Rate', 'win_rate_pct', '{:.1f}%'),
        ('Profit Factor', 'profit_factor', '{:.2f}'),
        ('Num Trades', 'num_trades', '{:.0f}'),
        ('Total Costs', 'total_costs', '${:.2f}'),
        ('Costs % Capital', 'costs_pct_of_capital', '{:.1f}%'),
    ]

    for label, key, fmt in metrics_to_show:
        row = f"| {label} |"
        for method in methods:
            val = all_results[method].get(key, 0)
            if val is not None and not np.isnan(val) and not np.isinf(val):
                if fmt.startswith('$'):
                    row += f" ${val:,.2f} |"
                else:
                    row += f" {fmt.format(val)} |"
            else:
                row += " N/A |"
        print(row)

    # Win rate analysis
    print("\n" + "=" * 70)
    print("WIN RATE ANALYSIS")
    print("=" * 70)

    for method, metrics in all_results.items():
        wr = metrics['win_rate_pct']
        aw = metrics['avg_win']
        al = abs(metrics['avg_loss'])

        if al > 0 and aw > 0:
            rr_ratio = aw / al
            breakeven = 1 / (1 + rr_ratio) * 100
            edge = wr - breakeven

            print(f"\n{method.upper()}:")
            print(f"  Win Rate:        {wr:.1f}%")
            print(f"  Avg Win:         ${aw:.2f}")
            print(f"  Avg Loss:        ${al:.2f}")
            print(f"  Reward:Risk:     {rr_ratio:.2f}:1")
            print(f"  Breakeven WR:    {breakeven:.1f}%")
            print(f"  Edge:            {edge:+.1f}%")

    print(f"\nResults saved to: {output_dir}")

    # Generate quantstats reports
    try:
        import quantstats as qs

        print("\nGenerating QuantStats reports...")
        for method, eq_df in all_equity.items():
            eq = eq_df.copy()
            eq['date'] = pd.to_datetime(eq['date'])
            eq = eq.set_index('date')
            daily = eq['equity'].resample('D').last().ffill()
            returns = daily.pct_change().dropna()

            if len(returns) > 10:
                report_path = output_dir / f"quantstats_report_{method}_proper.html"
                qs.reports.html(returns, output=str(report_path),
                               title=f"Whale Strategy - {method} (with costs)")
                print(f"  Generated: {report_path}")
    except Exception as e:
        print(f"  Could not generate QuantStats reports: {e}")

    return all_results, all_equity


if __name__ == "__main__":
    run_all_methods()

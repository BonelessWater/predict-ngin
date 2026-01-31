"""
Copy-cat whale strategy implementation.
Follows whale trades when 70%+ consensus is reached.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import CONSENSUS_THRESHOLD, INITIAL_CAPITAL
from src.whales.detector import WhaleDetector, WhaleInfo, WhaleMethod

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Position(Enum):
    """Current position state."""
    NONE = "none"
    LONG = "long"  # Bought YES
    SHORT = "short"  # Bought NO


@dataclass
class Trade:
    """Record of a single trade."""
    timestamp: datetime
    action: str  # 'buy', 'sell', 'hold'
    position: Position
    price: float
    size: float
    capital_before: float
    capital_after: float
    whale_consensus: float
    whale_direction: str
    reason: str


@dataclass
class StrategyState:
    """Current state of the strategy."""
    capital: float
    position: Position = Position.NONE
    position_size: float = 0.0
    entry_price: float = 0.0
    trades: List[Trade] = field(default_factory=list)


class CopycatStrategy:
    """
    Copy-cat whale strategy.

    Rules:
    - Enter position when whale consensus >= threshold (70%)
    - Use 100% of capital for the trade
    - Exit when consensus drops below threshold or reverses
    - Track whales using configurable detection method
    """

    def __init__(
        self,
        whale_detector: WhaleDetector,
        whale_method: WhaleMethod = WhaleMethod.TOP_N,
        consensus_threshold: float = CONSENSUS_THRESHOLD,
        initial_capital: float = INITIAL_CAPITAL
    ):
        self.detector = whale_detector
        self.whale_method = whale_method
        self.consensus_threshold = consensus_threshold
        self.initial_capital = initial_capital

    def _detect_whales(
        self,
        trades_df: pd.DataFrame,
        reference_time: datetime
    ) -> List[WhaleInfo]:
        """Detect whales using configured method."""
        if self.whale_method == WhaleMethod.TOP_N:
            return self.detector.detect_whales_top_n(trades_df, reference_time)
        elif self.whale_method == WhaleMethod.PERCENTILE:
            return self.detector.detect_whales_percentile(trades_df, reference_time)
        elif self.whale_method == WhaleMethod.TRADE_SIZE:
            return self.detector.detect_whales_trade_size(trades_df, reference_time)
        else:
            return self.detector.detect_whales_top_n(trades_df, reference_time)

    def _get_signal(
        self,
        whales: List[WhaleInfo],
        trades_df: pd.DataFrame,
        reference_time: datetime,
        lookback_hours: int = 24
    ) -> Tuple[str, float]:
        """
        Get trading signal based on whale consensus.

        Returns:
            (signal, consensus) where signal is 'buy', 'sell', or 'hold'
        """
        consensus, direction = self.detector.calculate_whale_consensus(
            whales, trades_df, reference_time, lookback_hours
        )

        if consensus >= self.consensus_threshold:
            return direction, consensus
        else:
            return 'hold', consensus

    def _execute_trade(
        self,
        state: StrategyState,
        signal: str,
        price: float,
        timestamp: datetime,
        consensus: float,
        whale_direction: str
    ) -> StrategyState:
        """
        Execute a trade based on signal.
        Uses 100% of capital for position.
        """
        if signal == 'hold' and state.position == Position.NONE:
            # No position and no signal - do nothing
            trade = Trade(
                timestamp=timestamp,
                action='hold',
                position=state.position,
                price=price,
                size=0,
                capital_before=state.capital,
                capital_after=state.capital,
                whale_consensus=consensus,
                whale_direction=whale_direction,
                reason="No consensus signal"
            )
            state.trades.append(trade)
            return state

        if signal == 'buy' and state.position == Position.NONE:
            # Enter long position
            position_size = state.capital / price
            trade = Trade(
                timestamp=timestamp,
                action='buy',
                position=Position.LONG,
                price=price,
                size=position_size,
                capital_before=state.capital,
                capital_after=state.capital,  # Capital unchanged until exit
                whale_consensus=consensus,
                whale_direction=whale_direction,
                reason=f"Whale consensus {consensus:.1%} >= {self.consensus_threshold:.1%}"
            )
            state.trades.append(trade)
            state.position = Position.LONG
            state.position_size = position_size
            state.entry_price = price
            return state

        if signal == 'sell' and state.position == Position.NONE:
            # Enter short position (buy NO tokens)
            position_size = state.capital / (1 - price)  # NO token price
            trade = Trade(
                timestamp=timestamp,
                action='sell',
                position=Position.SHORT,
                price=price,
                size=position_size,
                capital_before=state.capital,
                capital_after=state.capital,
                whale_consensus=consensus,
                whale_direction=whale_direction,
                reason=f"Whale consensus {consensus:.1%} >= {self.consensus_threshold:.1%} (bearish)"
            )
            state.trades.append(trade)
            state.position = Position.SHORT
            state.position_size = position_size
            state.entry_price = price
            return state

        if state.position == Position.LONG:
            if signal == 'sell' or signal == 'hold':
                # Exit long position
                pnl = state.position_size * (price - state.entry_price)
                new_capital = state.capital + pnl
                trade = Trade(
                    timestamp=timestamp,
                    action='close_long',
                    position=Position.NONE,
                    price=price,
                    size=state.position_size,
                    capital_before=state.capital,
                    capital_after=new_capital,
                    whale_consensus=consensus,
                    whale_direction=whale_direction,
                    reason=f"Exit: consensus dropped or reversed (was {consensus:.1%})"
                )
                state.trades.append(trade)
                state.capital = new_capital
                state.position = Position.NONE
                state.position_size = 0
                state.entry_price = 0

                # If signal is sell, also enter short
                if signal == 'sell':
                    return self._execute_trade(
                        state, signal, price, timestamp, consensus, whale_direction
                    )

        if state.position == Position.SHORT:
            if signal == 'buy' or signal == 'hold':
                # Exit short position
                # Short profit = entry_price - exit_price (on YES token)
                # Or equivalently: position_size * (exit_no_price - entry_no_price)
                entry_no_price = 1 - state.entry_price
                exit_no_price = 1 - price
                pnl = state.position_size * (exit_no_price - entry_no_price)
                new_capital = state.capital + pnl
                trade = Trade(
                    timestamp=timestamp,
                    action='close_short',
                    position=Position.NONE,
                    price=price,
                    size=state.position_size,
                    capital_before=state.capital,
                    capital_after=new_capital,
                    whale_consensus=consensus,
                    whale_direction=whale_direction,
                    reason=f"Exit: consensus dropped or reversed (was {consensus:.1%})"
                )
                state.trades.append(trade)
                state.capital = new_capital
                state.position = Position.NONE
                state.position_size = 0
                state.entry_price = 0

                # If signal is buy, also enter long
                if signal == 'buy':
                    return self._execute_trade(
                        state, signal, price, timestamp, consensus, whale_direction
                    )

        return state

    def run_backtest(
        self,
        trades_df: pd.DataFrame,
        prices_df: pd.DataFrame,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        signal_interval_hours: int = 4
    ) -> StrategyState:
        """
        Run backtest over historical data.

        Args:
            trades_df: Historical trades for whale detection
            prices_df: Price history for the market
            start_time: Backtest start (default: earliest price data)
            end_time: Backtest end (default: latest price data)
            signal_interval_hours: How often to check signals

        Returns:
            Final strategy state with all trades
        """
        if prices_df.empty:
            logger.warning("No price data for backtest")
            return StrategyState(capital=self.initial_capital)

        # Parse price timestamps
        prices = prices_df.copy()
        time_col = None
        for col in ['timestamp', 't', 'time', 'date']:
            if col in prices.columns:
                time_col = col
                break

        if time_col is None:
            logger.warning("No timestamp column in price data")
            return StrategyState(capital=self.initial_capital)

        prices['parsed_time'] = pd.to_datetime(prices[time_col])
        prices = prices.sort_values('parsed_time')

        # Get price column
        price_col = None
        for col in ['price', 'p', 'close', 'mid']:
            if col in prices.columns:
                price_col = col
                break

        if price_col is None:
            logger.warning("No price column in price data")
            return StrategyState(capital=self.initial_capital)

        # Set time bounds
        if start_time is None:
            start_time = prices['parsed_time'].min()
        if end_time is None:
            end_time = prices['parsed_time'].max()

        # Initialize state
        state = StrategyState(capital=self.initial_capital)

        # Generate signal check times
        current_time = start_time
        interval = timedelta(hours=signal_interval_hours)

        logger.info(f"Running backtest from {start_time} to {end_time}")
        logger.info(f"Method: {self.whale_method.value}, Consensus threshold: {self.consensus_threshold:.0%}")

        while current_time <= end_time:
            # Get current price (nearest price at or before current_time)
            mask = prices['parsed_time'] <= current_time
            if not mask.any():
                current_time += interval
                continue

            current_price_row = prices[mask].iloc[-1]
            current_price = float(current_price_row[price_col])

            # Detect whales at this point in time
            whales = self._detect_whales(trades_df, current_time)

            # Get signal
            signal, consensus = self._get_signal(
                whales, trades_df, current_time, lookback_hours=24
            )

            # Determine whale direction for logging
            if signal in ['buy', 'sell']:
                whale_direction = signal
            else:
                _, whale_direction = self.detector.calculate_whale_consensus(
                    whales, trades_df, current_time, lookback_hours=24
                )

            # Execute trade if needed
            state = self._execute_trade(
                state, signal, current_price, current_time, consensus, whale_direction
            )

            current_time += interval

        # Close any open position at end
        if state.position != Position.NONE:
            final_price = float(prices.iloc[-1][price_col])
            final_time = prices['parsed_time'].max()
            state = self._execute_trade(
                state, 'hold', final_price, final_time, 0.0, 'neutral'
            )

        logger.info(f"Backtest complete. Final capital: ${state.capital:,.2f}")
        logger.info(f"Total trades: {len(state.trades)}")

        return state


def main():
    """Test strategy with sample data."""
    from src.whales.detector import WhaleDetector

    # Create sample data
    np.random.seed(42)
    n_trades = 500
    n_prices = 168  # 1 week hourly

    # Sample trades
    trades = pd.DataFrame({
        'timestamp': pd.date_range(end=datetime.now(), periods=n_trades, freq='30min'),
        'trader': [f"0x{i:040x}" for i in np.random.choice(50, n_trades)],
        'size': np.random.exponential(2000, n_trades),
        'side': np.random.choice(['buy', 'sell'], n_trades, p=[0.6, 0.4])  # Slight buy bias
    })

    # Sample prices with trend
    base_price = 0.5
    prices = pd.DataFrame({
        'timestamp': pd.date_range(end=datetime.now(), periods=n_prices, freq='1H'),
        'price': base_price + np.cumsum(np.random.randn(n_prices) * 0.01)
    })
    prices['price'] = prices['price'].clip(0.01, 0.99)

    # Run backtest
    detector = WhaleDetector()
    strategy = CopycatStrategy(detector, whale_method=WhaleMethod.TOP_N)
    state = strategy.run_backtest(trades, prices)

    print(f"\nInitial capital: ${INITIAL_CAPITAL:,.2f}")
    print(f"Final capital: ${state.capital:,.2f}")
    print(f"Return: {(state.capital / INITIAL_CAPITAL - 1) * 100:.2f}%")
    print(f"Trades executed: {len([t for t in state.trades if t.action != 'hold'])}")


if __name__ == "__main__":
    main()

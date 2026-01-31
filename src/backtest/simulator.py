"""
Extended market simulator for year-long backtests.
Generates realistic synthetic data with multiple market regimes.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, List, Dict
from dataclasses import dataclass
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime types."""
    ACCUMULATION = "accumulation"  # Whales buying, price flat/rising slowly
    MARKUP = "markup"  # Strong uptrend, retail FOMO
    DISTRIBUTION = "distribution"  # Whales selling, price flat/falling slowly
    MARKDOWN = "markdown"  # Strong downtrend, panic selling
    CHOPPY = "choppy"  # No clear direction, lots of noise


@dataclass
class RegimeConfig:
    """Configuration for a market regime."""
    regime: MarketRegime
    duration_days: int
    whale_buy_prob: float  # Probability whales buy vs sell
    retail_buy_prob: float  # Probability retail buys vs sell
    price_drift: float  # Daily drift
    volatility: float  # Daily volatility


def generate_year_data(
    seed: int = 42,
    start_date: datetime = None,
    num_traders: int = 500,
    num_whales: int = 25,
    trades_per_day: int = 150
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate a full year of realistic market data.

    Simulates multiple market regimes:
    - Accumulation phases (whales buying quietly)
    - Markup phases (price rises, retail joins)
    - Distribution phases (whales selling into strength)
    - Markdown phases (price falls, panic)
    - Choppy phases (no clear direction)

    Returns:
        (trades_df, prices_df)
    """
    np.random.seed(seed)

    if start_date is None:
        start_date = datetime.now() - timedelta(days=365)

    # Define market regimes for the year
    regimes = [
        RegimeConfig(MarketRegime.CHOPPY, 30, 0.50, 0.50, 0.0, 0.015),
        RegimeConfig(MarketRegime.ACCUMULATION, 45, 0.75, 0.45, 0.001, 0.010),
        RegimeConfig(MarketRegime.MARKUP, 60, 0.60, 0.70, 0.003, 0.020),
        RegimeConfig(MarketRegime.DISTRIBUTION, 40, 0.25, 0.55, -0.001, 0.012),
        RegimeConfig(MarketRegime.MARKDOWN, 35, 0.30, 0.30, -0.004, 0.025),
        RegimeConfig(MarketRegime.CHOPPY, 25, 0.50, 0.50, 0.0, 0.018),
        RegimeConfig(MarketRegime.ACCUMULATION, 40, 0.80, 0.40, 0.0005, 0.008),
        RegimeConfig(MarketRegime.MARKUP, 50, 0.55, 0.75, 0.004, 0.022),
        RegimeConfig(MarketRegime.DISTRIBUTION, 30, 0.20, 0.60, 0.0, 0.015),
        RegimeConfig(MarketRegime.MARKDOWN, 25, 0.35, 0.25, -0.005, 0.030),
    ]

    # Create trader pool
    all_traders = [f"0x{i:040x}" for i in range(num_traders)]
    whale_traders = all_traders[:num_whales]
    retail_traders = all_traders[num_whales:]

    # Generate data
    trades_list = []
    prices_list = []

    current_date = start_date
    current_price = 0.40  # Starting price

    total_days = sum(r.duration_days for r in regimes)
    logger.info(f"Generating {total_days} days of data ({total_days // 30} months)")

    for regime_idx, regime in enumerate(regimes):
        logger.info(f"Generating regime {regime_idx + 1}/{len(regimes)}: {regime.regime.value} ({regime.duration_days} days)")

        for day in range(regime.duration_days):
            # Generate hourly prices for the day
            for hour in range(24):
                timestamp = current_date + timedelta(hours=hour)

                # Price movement
                noise = np.random.randn() * regime.volatility / np.sqrt(24)
                current_price += regime.price_drift / 24 + noise
                current_price = max(0.05, min(0.95, current_price))

                prices_list.append({
                    'timestamp': timestamp,
                    'price': current_price,
                    'regime': regime.regime.value
                })

            # Generate trades for the day
            num_trades_today = int(trades_per_day * (0.7 + np.random.random() * 0.6))

            for _ in range(num_trades_today):
                # Determine if whale or retail
                is_whale = np.random.random() < 0.12  # 12% of trades from whales

                if is_whale:
                    trader = np.random.choice(whale_traders)
                    # Whale trade sizes: larger, with some very large trades
                    base_size = np.random.exponential(15000)
                    if np.random.random() < 0.1:  # 10% chance of mega trade
                        base_size *= 3
                    buy_prob = regime.whale_buy_prob
                else:
                    trader = np.random.choice(retail_traders)
                    # Retail trade sizes: smaller, more uniform
                    base_size = np.random.exponential(800)
                    buy_prob = regime.retail_buy_prob

                # Add some noise to buy probability
                buy_prob = np.clip(buy_prob + np.random.randn() * 0.1, 0.1, 0.9)
                side = 'buy' if np.random.random() < buy_prob else 'sell'

                # Random time within the day
                trade_time = current_date + timedelta(
                    hours=np.random.randint(0, 24),
                    minutes=np.random.randint(0, 60)
                )

                trades_list.append({
                    'timestamp': trade_time,
                    'trader': trader,
                    'size': base_size,
                    'side': side,
                    'is_whale': is_whale,
                    'regime': regime.regime.value
                })

            current_date += timedelta(days=1)

    trades_df = pd.DataFrame(trades_list)
    trades_df = trades_df.sort_values('timestamp').reset_index(drop=True)

    prices_df = pd.DataFrame(prices_list)
    prices_df = prices_df.sort_values('timestamp').reset_index(drop=True)

    logger.info(f"Generated {len(trades_df):,} trades and {len(prices_df):,} price points")
    logger.info(f"Date range: {trades_df['timestamp'].min()} to {trades_df['timestamp'].max()}")
    logger.info(f"Price range: {prices_df['price'].min():.4f} to {prices_df['price'].max():.4f}")

    # Summary stats
    whale_trades = trades_df[trades_df['is_whale']]
    retail_trades = trades_df[~trades_df['is_whale']]

    logger.info(f"\nTrade breakdown:")
    logger.info(f"  Whale trades: {len(whale_trades):,} ({len(whale_trades)/len(trades_df)*100:.1f}%)")
    logger.info(f"  Retail trades: {len(retail_trades):,} ({len(retail_trades)/len(trades_df)*100:.1f}%)")
    logger.info(f"  Avg whale trade: ${whale_trades['size'].mean():,.0f}")
    logger.info(f"  Avg retail trade: ${retail_trades['size'].mean():,.0f}")

    return trades_df, prices_df


def analyze_regime_performance(
    trades_df: pd.DataFrame,
    strategy_trades: List,
    prices_df: pd.DataFrame
) -> pd.DataFrame:
    """Analyze strategy performance by market regime."""
    results = []

    for trade in strategy_trades:
        if trade.action in ['hold']:
            continue

        # Find regime at trade time
        price_at_time = prices_df[prices_df['timestamp'] <= trade.timestamp].iloc[-1]
        regime = price_at_time.get('regime', 'unknown')

        pnl = trade.capital_after - trade.capital_before

        results.append({
            'timestamp': trade.timestamp,
            'action': trade.action,
            'regime': regime,
            'pnl': pnl,
            'return_pct': (pnl / trade.capital_before * 100) if trade.capital_before > 0 else 0
        })

    return pd.DataFrame(results)


def main():
    """Generate and save year-long dataset."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from config.settings import PARQUET_DIR

    trades_df, prices_df = generate_year_data()

    # Save to parquet
    trades_df.to_parquet(PARQUET_DIR / "synthetic_trades_1y.parquet")
    prices_df.to_parquet(PARQUET_DIR / "synthetic_prices_1y.parquet")

    print(f"\nSaved to {PARQUET_DIR}")
    print(f"  - synthetic_trades_1y.parquet ({len(trades_df):,} trades)")
    print(f"  - synthetic_prices_1y.parquet ({len(prices_df):,} prices)")


if __name__ == "__main__":
    main()

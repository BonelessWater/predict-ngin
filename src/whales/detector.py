"""
Whale detection module.
Implements three methods for identifying whale traders:
1. Top N by total trading volume
2. 95th percentile by volume
3. Per-trade size threshold (>$10K mean trade size)
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Set, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
import logging

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import (
    WHALE_TOP_N, WHALE_PERCENTILE, WHALE_MIN_TRADE_SIZE,
    WHALE_STALENESS_DAYS
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WhaleMethod(Enum):
    """Whale detection methods."""
    TOP_N = "top_n"
    PERCENTILE = "percentile"
    TRADE_SIZE = "trade_size"


@dataclass
class WhaleInfo:
    """Information about a detected whale."""
    address: str
    method: WhaleMethod
    total_volume: float
    trade_count: int
    avg_trade_size: float
    last_trade_time: datetime
    direction_bias: float  # -1 to 1, negative = more sells, positive = more buys


class WhaleDetector:
    """
    Detects whale traders using multiple methods.
    """

    def __init__(
        self,
        top_n: int = WHALE_TOP_N,
        percentile: float = WHALE_PERCENTILE,
        min_trade_size: float = WHALE_MIN_TRADE_SIZE,
        staleness_days: int = WHALE_STALENESS_DAYS
    ):
        self.top_n = top_n
        self.percentile = percentile
        self.min_trade_size = min_trade_size
        self.staleness_days = staleness_days

    def _parse_timestamp(self, ts) -> Optional[datetime]:
        """Parse various timestamp formats."""
        if pd.isna(ts):
            return None
        if isinstance(ts, datetime):
            return ts
        if isinstance(ts, (int, float)):
            # Unix timestamp (seconds or milliseconds)
            if ts > 1e12:
                ts = ts / 1000
            return datetime.fromtimestamp(ts)
        if isinstance(ts, str):
            try:
                return pd.to_datetime(ts).to_pydatetime()
            except:
                return None
        return None

    def _get_trade_value(self, row: pd.Series) -> float:
        """Extract trade value from various column formats."""
        # Try different column names for trade value
        for col in ['size', 'amount', 'value', 'trade_size', 'usd_value', 'price']:
            if col in row.index and pd.notna(row[col]):
                try:
                    val = float(row[col])
                    # If this looks like a price (0-1 range), multiply by size
                    if col == 'price' and 0 < val <= 1:
                        for size_col in ['size', 'amount', 'quantity']:
                            if size_col in row.index and pd.notna(row[size_col]):
                                return val * float(row[size_col])
                    return val
                except (ValueError, TypeError):
                    continue
        return 0.0

    def _get_trader_address(self, row: pd.Series) -> Optional[str]:
        """Extract trader address from various column formats."""
        for col in ['maker', 'taker', 'user', 'trader', 'address', 'owner']:
            if col in row.index and pd.notna(row[col]):
                return str(row[col])
        return None

    def _get_trade_side(self, row: pd.Series) -> Optional[str]:
        """Extract trade side (buy/sell) from row."""
        for col in ['side', 'type', 'action', 'direction']:
            if col in row.index and pd.notna(row[col]):
                side = str(row[col]).lower()
                if 'buy' in side or 'bid' in side:
                    return 'buy'
                if 'sell' in side or 'ask' in side:
                    return 'sell'
        return None

    def _preprocess_trades(
        self,
        trades_df: pd.DataFrame,
        reference_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Preprocess trades dataframe to standardized format.
        Filters by staleness window.
        """
        if trades_df.empty:
            return trades_df

        df = trades_df.copy()

        # Parse timestamps
        timestamp_cols = ['timestamp', 'created_at', 'time', 'date', 'createdAt']
        for col in timestamp_cols:
            if col in df.columns:
                df['parsed_time'] = df[col].apply(self._parse_timestamp)
                break
        else:
            df['parsed_time'] = datetime.now()

        # Filter by staleness
        if reference_time is None:
            reference_time = datetime.now()
        cutoff = reference_time - timedelta(days=self.staleness_days)
        df = df[df['parsed_time'] >= cutoff].copy()

        # Extract standardized fields
        df['trade_value'] = df.apply(self._get_trade_value, axis=1)
        df['trader'] = df.apply(self._get_trader_address, axis=1)
        df['side'] = df.apply(self._get_trade_side, axis=1)

        # Remove rows without trader info
        df = df[df['trader'].notna()].copy()

        return df

    def _aggregate_by_trader(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate trade statistics by trader."""
        if trades_df.empty:
            return pd.DataFrame()

        agg = trades_df.groupby('trader').agg(
            total_volume=('trade_value', 'sum'),
            trade_count=('trade_value', 'count'),
            avg_trade_size=('trade_value', 'mean'),
            last_trade=('parsed_time', 'max'),
            buy_count=('side', lambda x: (x == 'buy').sum()),
            sell_count=('side', lambda x: (x == 'sell').sum())
        ).reset_index()

        # Calculate direction bias (-1 to 1)
        total_directional = agg['buy_count'] + agg['sell_count']
        agg['direction_bias'] = np.where(
            total_directional > 0,
            (agg['buy_count'] - agg['sell_count']) / total_directional,
            0
        )

        return agg

    def detect_whales_top_n(
        self,
        trades_df: pd.DataFrame,
        reference_time: Optional[datetime] = None
    ) -> List[WhaleInfo]:
        """
        Method 1: Detect whales as top N traders by total volume.
        """
        processed = self._preprocess_trades(trades_df, reference_time)
        if processed.empty:
            return []

        aggregated = self._aggregate_by_trader(processed)
        if aggregated.empty:
            return []

        # Sort by total volume and take top N
        top_traders = aggregated.nlargest(self.top_n, 'total_volume')

        whales = []
        for _, row in top_traders.iterrows():
            whale = WhaleInfo(
                address=row['trader'],
                method=WhaleMethod.TOP_N,
                total_volume=row['total_volume'],
                trade_count=int(row['trade_count']),
                avg_trade_size=row['avg_trade_size'],
                last_trade_time=row['last_trade'],
                direction_bias=row['direction_bias']
            )
            whales.append(whale)

        logger.info(f"Top {self.top_n} method: Found {len(whales)} whales")
        return whales

    def detect_whales_percentile(
        self,
        trades_df: pd.DataFrame,
        reference_time: Optional[datetime] = None
    ) -> List[WhaleInfo]:
        """
        Method 2: Detect whales as traders above the Nth percentile by volume.
        """
        processed = self._preprocess_trades(trades_df, reference_time)
        if processed.empty:
            return []

        aggregated = self._aggregate_by_trader(processed)
        if aggregated.empty:
            return []

        # Calculate percentile threshold
        threshold = np.percentile(aggregated['total_volume'], self.percentile)
        high_volume = aggregated[aggregated['total_volume'] >= threshold]

        whales = []
        for _, row in high_volume.iterrows():
            whale = WhaleInfo(
                address=row['trader'],
                method=WhaleMethod.PERCENTILE,
                total_volume=row['total_volume'],
                trade_count=int(row['trade_count']),
                avg_trade_size=row['avg_trade_size'],
                last_trade_time=row['last_trade'],
                direction_bias=row['direction_bias']
            )
            whales.append(whale)

        logger.info(f"Percentile ({self.percentile}%) method: Found {len(whales)} whales (threshold: ${threshold:,.2f})")
        return whales

    def detect_whales_trade_size(
        self,
        trades_df: pd.DataFrame,
        reference_time: Optional[datetime] = None
    ) -> List[WhaleInfo]:
        """
        Method 3: Detect whales based on average trade size > threshold.
        Only includes traders with mean trade size > $10K in the lookback period.
        """
        processed = self._preprocess_trades(trades_df, reference_time)
        if processed.empty:
            return []

        aggregated = self._aggregate_by_trader(processed)
        if aggregated.empty:
            return []

        # Filter by average trade size
        large_traders = aggregated[aggregated['avg_trade_size'] >= self.min_trade_size]

        whales = []
        for _, row in large_traders.iterrows():
            whale = WhaleInfo(
                address=row['trader'],
                method=WhaleMethod.TRADE_SIZE,
                total_volume=row['total_volume'],
                trade_count=int(row['trade_count']),
                avg_trade_size=row['avg_trade_size'],
                last_trade_time=row['last_trade'],
                direction_bias=row['direction_bias']
            )
            whales.append(whale)

        logger.info(f"Trade size (>${self.min_trade_size:,}) method: Found {len(whales)} whales")
        return whales

    def detect_all_methods(
        self,
        trades_df: pd.DataFrame,
        reference_time: Optional[datetime] = None
    ) -> Dict[WhaleMethod, List[WhaleInfo]]:
        """Run all three detection methods and return results."""
        return {
            WhaleMethod.TOP_N: self.detect_whales_top_n(trades_df, reference_time),
            WhaleMethod.PERCENTILE: self.detect_whales_percentile(trades_df, reference_time),
            WhaleMethod.TRADE_SIZE: self.detect_whales_trade_size(trades_df, reference_time)
        }

    def get_whale_addresses(
        self,
        whales: List[WhaleInfo]
    ) -> Set[str]:
        """Extract unique addresses from whale list."""
        return {w.address for w in whales}

    def calculate_whale_consensus(
        self,
        whales: List[WhaleInfo],
        trades_df: pd.DataFrame,
        reference_time: Optional[datetime] = None,
        lookback_hours: int = 24
    ) -> Tuple[float, str]:
        """
        Calculate whale consensus direction.
        Returns (consensus_ratio, direction) where:
        - consensus_ratio: 0.0 to 1.0 (fraction agreeing on dominant direction)
        - direction: 'buy', 'sell', or 'neutral'
        """
        if not whales:
            return 0.0, 'neutral'

        whale_addresses = self.get_whale_addresses(whales)

        # Filter recent trades by whales
        processed = self._preprocess_trades(trades_df, reference_time)
        if processed.empty:
            return 0.0, 'neutral'

        # Additional filter for lookback period
        if reference_time is None:
            reference_time = datetime.now()
        lookback_cutoff = reference_time - timedelta(hours=lookback_hours)
        recent = processed[
            (processed['trader'].isin(whale_addresses)) &
            (processed['parsed_time'] >= lookback_cutoff)
        ]

        if recent.empty:
            return 0.0, 'neutral'

        # Count buys and sells weighted by volume
        buys = recent[recent['side'] == 'buy']['trade_value'].sum()
        sells = recent[recent['side'] == 'sell']['trade_value'].sum()
        total = buys + sells

        if total == 0:
            return 0.0, 'neutral'

        buy_ratio = buys / total
        sell_ratio = sells / total

        if buy_ratio > sell_ratio:
            return buy_ratio, 'buy'
        else:
            return sell_ratio, 'sell'


def main():
    """Test whale detection."""
    # Create sample data for testing
    np.random.seed(42)
    n_trades = 1000

    sample_trades = pd.DataFrame({
        'timestamp': pd.date_range(
            end=datetime.now(),
            periods=n_trades,
            freq='1H'
        ),
        'trader': [f"0x{i:040x}" for i in np.random.choice(100, n_trades)],
        'size': np.random.exponential(1000, n_trades),
        'side': np.random.choice(['buy', 'sell'], n_trades)
    })

    detector = WhaleDetector()
    results = detector.detect_all_methods(sample_trades)

    for method, whales in results.items():
        print(f"\n{method.value}: {len(whales)} whales")
        for w in whales[:3]:
            print(f"  {w.address[:16]}... vol=${w.total_volume:,.0f} avg=${w.avg_trade_size:,.0f} bias={w.direction_bias:.2f}")


if __name__ == "__main__":
    main()

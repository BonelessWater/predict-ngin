"""
Configuration class for the whale tracking strategy.
All configurable parameters in one place.
"""
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


@dataclass
class WhaleConfig:
    """Configuration for whale detection."""
    top_n: int = 10  # Top N traders by volume
    percentile: float = 95.0  # Percentile threshold
    min_trade_size: float = 10_000  # USD threshold for trade size method
    staleness_days: int = 7  # Ignore activity older than this


@dataclass
class StrategyConfig:
    """Configuration for the copycat strategy."""
    consensus_threshold: float = 0.70  # 70% whale agreement required
    initial_capital: float = 10_000  # Starting capital
    position_size_pct: float = 1.0  # Use 100% of capital per trade


@dataclass
class CostConfig:
    """Transaction costs and slippage configuration."""
    # Polymarket fees
    maker_fee: float = 0.0  # Maker fee (currently 0 on Polymarket)
    taker_fee: float = 0.02  # Taker fee (2% on Polymarket for market orders)

    # Slippage model
    base_slippage: float = 0.005  # Base slippage (0.5%)
    volume_impact: float = 0.001  # Additional slippage per $1000 traded
    max_slippage: float = 0.05  # Cap slippage at 5%

    def calculate_slippage(self, trade_size: float, is_buy: bool) -> float:
        """
        Calculate slippage based on trade size.
        Larger trades have more market impact.

        Args:
            trade_size: Size of trade in USD
            is_buy: True if buying, False if selling

        Returns:
            Slippage as a decimal (e.g., 0.01 = 1%)
        """
        # Volume-based slippage
        volume_slippage = self.volume_impact * (trade_size / 1000)
        total_slippage = self.base_slippage + volume_slippage

        # Cap at max
        return min(total_slippage, self.max_slippage)

    def calculate_total_cost(self, trade_size: float, price: float, is_buy: bool) -> float:
        """
        Calculate total transaction cost including fees and slippage.

        Args:
            trade_size: Size of trade in USD
            price: Current price
            is_buy: True if buying, False if selling

        Returns:
            Total cost in USD
        """
        # Trading fee (taker fee for market orders)
        fee_cost = trade_size * self.taker_fee

        # Slippage cost
        slippage = self.calculate_slippage(trade_size, is_buy)
        slippage_cost = trade_size * slippage

        return fee_cost + slippage_cost

    def get_execution_price(self, market_price: float, trade_size: float, is_buy: bool) -> float:
        """
        Get the actual execution price after slippage.

        Args:
            market_price: Current market price
            trade_size: Size of trade in USD
            is_buy: True if buying, False if selling

        Returns:
            Actual execution price
        """
        slippage = self.calculate_slippage(trade_size, is_buy)

        if is_buy:
            # Buying pushes price up
            return market_price * (1 + slippage)
        else:
            # Selling pushes price down
            return market_price * (1 - slippage)


@dataclass
class DataConfig:
    """Configuration for data paths and caching."""
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    data_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data")
    parquet_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data" / "parquet")
    output_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data" / "output")

    # Cache settings
    cache_max_age_hours: int = 24  # Max age for cached data

    def __post_init__(self):
        """Ensure directories exist."""
        self.parquet_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class APIConfig:
    """Configuration for Polymarket APIs."""
    gamma_api: str = "https://gamma-api.polymarket.com"
    clob_api: str = "https://clob.polymarket.com"
    data_api: str = "https://data-api.polymarket.com"
    ws_url: str = "wss://ws-subscriptions-clob.polymarket.com"

    # Rate limiting
    rate_limit_delay: float = 0.5  # Seconds between API calls


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    signal_interval_hours: int = 4  # How often to check signals
    warmup_days: int = 7  # Days of data needed before first trade

    # Validation
    min_trades_for_stats: int = 10  # Minimum trades for meaningful statistics


@dataclass
class Config:
    """Master configuration combining all sub-configs."""
    whale: WhaleConfig = field(default_factory=WhaleConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    data: DataConfig = field(default_factory=DataConfig)
    api: APIConfig = field(default_factory=APIConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)

    # Target market
    target_asset: str = "XRP"

    @classmethod
    def default(cls) -> 'Config':
        """Create default configuration."""
        return cls()

    @classmethod
    def no_costs(cls) -> 'Config':
        """Create configuration with no transaction costs (for comparison)."""
        config = cls()
        config.costs.taker_fee = 0.0
        config.costs.base_slippage = 0.0
        config.costs.volume_impact = 0.0
        return config

    @classmethod
    def conservative(cls) -> 'Config':
        """Create conservative configuration with higher costs."""
        config = cls()
        config.costs.taker_fee = 0.03  # 3% fee
        config.costs.base_slippage = 0.01  # 1% base slippage
        config.costs.volume_impact = 0.002  # Higher impact
        config.strategy.consensus_threshold = 0.75  # Higher threshold
        return config


# Global default config instance
DEFAULT_CONFIG = Config.default()


def get_config() -> Config:
    """Get the default configuration."""
    return DEFAULT_CONFIG

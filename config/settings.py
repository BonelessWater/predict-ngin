"""
Configuration settings for the Polymarket whale tracking strategy.
"""
from pathlib import Path

# API Endpoints
GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"
DATA_API = "https://data-api.polymarket.com"
WS_URL = "wss://ws-subscriptions-clob.polymarket.com"

# Data paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PARQUET_DIR = DATA_DIR / "parquet"

# Ensure directories exist
PARQUET_DIR.mkdir(parents=True, exist_ok=True)

# Target market
TARGET_ASSET = "XRP"  # Filter for XRP-related markets

# Whale detection thresholds
WHALE_TOP_N = 10  # Method 1: Top N traders by volume
WHALE_PERCENTILE = 95  # Method 2: Percentile threshold
WHALE_MIN_TRADE_SIZE = 10_000  # Method 3: Minimum trade size in USD

# Strategy parameters
CONSENSUS_THRESHOLD = 0.70  # 70% whale agreement required
WHALE_STALENESS_DAYS = 7  # Ignore whale activity older than this
INITIAL_CAPITAL = 10_000  # Starting capital for backtesting

# Polling interval (for live trading - not used in backtest)
POLL_INTERVAL_SECONDS = 300  # 5 minutes

# Rate limiting
API_RATE_LIMIT_DELAY = 0.5  # seconds between API calls

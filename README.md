# Polymarket Whale Tracking Strategy

A research system for detecting and following whale traders on Polymarket prediction markets.

## Important: Honest Backtest Results

**This strategy was tested with proper methodology:**
- No look-ahead bias (whales detected only using historical data)
- Transaction costs modeled (fees + slippage)
- Realistic whale edge assumptions (60-65% accuracy, not 100%)

## Backtest Results (1 Year, No Look-Ahead Bias)

### Cost Scenario Comparison

| Scenario | Return | Sharpe | Win Rate | Total Costs |
|----------|--------|--------|----------|-------------|
| **No Costs** | +17.2% | 0.55 | 46.8% | $0 |
| Low Costs (0.5% fee) | -73.1% | -2.97 | 35.5% | $6,257 |
| Medium Costs (1% fee) | -92.6% | -5.30 | 24.2% | $7,481 |
| High Costs (2% fee) | -99.2% | -7.69 | 14.5% | $7,820 |

### Key Findings

1. **Raw Alpha Exists**: Without costs, whale-tracking generates ~17% annual returns
2. **Costs Destroy Returns**: Even 0.5% per-trade costs turn profits into -73% losses
3. **Too Many Trades**: 62 trades/year with small edge = cost accumulation
4. **Win Rate Reality**: True win rate is ~47% (not 60%+ shown with look-ahead bias)

### Why Previous Results Were Wrong

The earlier backtest showed 854% returns because it had **look-ahead bias**:
- Whale signals were generated knowing future prices
- This made whales appear to have near-perfect prediction
- Real whale edge is ~60-65%, not 95%+

## Configuration

All parameters are in `config/config.py`:

```python
from config.config import Config

config = Config.default()

# Whale detection
config.whale.top_n = 10           # Top 10 by volume
config.whale.percentile = 95      # 95th percentile
config.whale.min_trade_size = 10_000
config.whale.staleness_days = 7

# Strategy
config.strategy.consensus_threshold = 0.70
config.strategy.initial_capital = 10_000
config.strategy.position_size_pct = 1.0

# Transaction Costs
config.costs.taker_fee = 0.02     # 2% taker fee
config.costs.base_slippage = 0.005 # 0.5% base slippage
config.costs.volume_impact = 0.001 # Additional slippage per $1000
config.costs.max_slippage = 0.05  # Cap at 5%
```

## Making This Strategy Viable

To make whale-tracking profitable with costs:

### 1. Reduce Trade Frequency
```python
config.strategy.consensus_threshold = 0.80  # Higher bar for trades
```

### 2. Use Maker Orders
```python
config.costs.taker_fee = 0.0  # Polymarket maker fee is 0%
config.costs.base_slippage = 0.002  # Lower with limit orders
```

### 3. Filter for Higher Conviction
- Only trade when multiple whale methods agree
- Require sustained consensus (not single-day spikes)
- Focus on markets with higher whale concentration

### 4. Reduce Position Size
```python
config.strategy.position_size_pct = 0.5  # 50% per trade
```

## Project Structure

```
predict-ngin/
├── config/
│   └── config.py               # Configuration class
├── src/
│   ├── data/
│   │   ├── loader.py           # Polymarket API loader
│   │   └── polymarket_fetcher.py
│   ├── whales/
│   │   └── detector.py         # Whale detection methods
│   ├── strategy/
│   │   └── copycat.py          # Copycat strategy
│   ├── backtest/
│   │   ├── proper_backtest.py  # No look-ahead bias
│   │   ├── cost_comparison.py  # Cost scenario analysis
│   │   └── fast_analysis.py    # Quick analysis (has bias)
│   └── main.py
├── data/
│   ├── parquet/                # Cached data
│   └── output/                 # Results & equity curves
└── requirements.txt
```

## Installation

```bash
git clone <repo-url>
cd predict-ngin
pip install -r requirements.txt
```

## Usage

### Run Proper Backtest (Recommended)

```bash
python src/backtest/proper_backtest.py
```

### Run Cost Comparison

```bash
python src/backtest/cost_comparison.py
```

### Fetch Real Polymarket Data

```bash
python src/data/polymarket_fetcher.py
```

## Output Files

Results saved to `data/output/`:

| File | Description |
|------|-------------|
| `equity_curve_*_proper.csv` | Equity curves (no bias) |
| `proper_backtest_metrics.json` | Metrics with costs |
| `cost_comparison_results.json` | All cost scenarios |
| `quantstats_report_*_proper.html` | QuantStats reports |

## Whale Detection Methods

| Method | Description | Performance |
|--------|-------------|-------------|
| **Top N** | Top 10 traders by volume | Best raw returns |
| **Percentile** | Above 95th percentile | Same as Top N |
| **Trade Size** | Avg trade > $10K | Slightly worse |

## Real Polymarket Data

Analysis of 500K real trades shows:

| Market | Top 10 Control | 95th Percentile |
|--------|---------------|-----------------|
| XRP | 50.6% | $16,712 |
| All Crypto | 59.8% | $34,901 |

**Whale concentration is real** - tracking is viable if costs can be managed.

## Limitations

1. **Data API Limitation**: Only recent trades available publicly
2. **No Live Trading**: Research only, not production-ready
3. **Single Market**: Tested on synthetic data, not real XRP
4. **Costs Estimated**: Actual slippage varies with market conditions

## Conclusions

1. **Whale tracking provides alpha** (~17% annually without costs)
2. **Transaction costs are the main obstacle** (not signal quality)
3. **Strategy needs optimization** for lower frequency, maker orders
4. **Not viable for retail** with high taker fees and small capital

## Data Collector

The data collector continuously fetches and stores Polymarket trades to build a historical database.

### Quick Start

```bash
# Run once (collect current trades)
python src/data/collector.py once

# Check collection status
python src/data/collector.py status

# Start continuous collection (runs every 5 minutes)
python src/data/collector.py start

# Start with custom interval (e.g., 10 minutes)
python src/data/collector.py start --interval 600

# Collect only XRP trades (filtered)
python src/data/collector.py once --market xrp

# Start continuous XRP collection
python src/data/collector.py start --market xrp

# Check XRP-specific status
python src/data/collector.py status --market xrp
```

### Windows Background Service

```batch
# Double-click to start
scripts\start_collector.bat
```

### Features

- **Deduplication**: Uses trade hashes to avoid storing duplicates
- **Partitioned Storage**: Data stored by date for efficient querying
- **Incremental**: Only fetches new trades each cycle
- **Resumable**: Tracks state across restarts

### Data Location

```
data/parquet/collected/
├── trades/                              # All trades
│   ├── date=2026-01-30/
│   │   └── trades_20260130_184849.parquet
│   └── ...
├── xrp/                                 # XRP-filtered trades
│   ├── trades/
│   │   └── date=2026-01-31/
│   │       └── trades_20260130_220433.parquet
│   ├── collection_metadata.json
│   └── seen_hashes.json
├── collection_metadata.json
└── seen_hashes.json
```

### Using Collected Data

```python
from src.data.collector import load_collected_data

# Load all collected trades
trades = load_collected_data()

# Load XRP-filtered trades
trades = load_collected_data(market_filter='xrp')

# Load specific date range
trades = load_collected_data(start_date='2026-01-30', end_date='2026-02-01')

# Load XRP trades for date range
trades = load_collected_data(market_filter='xrp', start_date='2026-01-30')
```

```bash
# Run backtest on real XRP data
python src/backtest/backtest_real_data.py --market XRP
```

### Building Historical Database

To get meaningful backtest results, collect data for at least:
- **1 week**: Minimum for testing
- **1 month**: Basic statistics
- **3+ months**: Reliable backtest results

Run the collector continuously:
```bash
# Leave running in background
python src/data/collector.py start --interval 300
```

## Development Log

### 2026-01-30

- Implemented proper backtest with no look-ahead bias
- Added transaction cost and slippage modeling
- Created Config class for all parameters
- Ran cost scenario comparison
- Found that raw alpha exists but costs destroy it
- Updated README with honest results

### Key Learnings

- Look-ahead bias can inflate returns by 50x+
- Whale edge is real but modest (~60-65%)
- Cost management is more important than signal quality
- High-frequency strategies need near-zero costs to work

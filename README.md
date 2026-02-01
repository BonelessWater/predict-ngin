# Prediction Market Whale Following Strategy

A bias-free backtesting framework for whale-following strategies on prediction markets.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Fetch data (if not already downloaded, ~4GB)
python run.py --fetch-only

# Run full analysis (auto-fetches if no data)
python run.py

# Run with specific method
python run.py --method win_rate_60pct --position-size 250

# Run all methods
python run.py --all-methods

# Analyze different position sizes ($100-$10k)
python run.py --analyze-sizes --capital 100000

# Rolling monthly whale identification (recalculates whales each month)
python run.py --rolling --lookback 3

# Refresh data and run
python run.py --fetch --all-methods
```

## Key Findings

Based on 5M+ bets from Manifold Markets (Dec 2024 - Jan 2026):

| Metric | Value | Notes |
|--------|-------|-------|
| Data | 5,001,000 bets | 174,400 markets |
| Resolved Markets | 89,765 | YES/NO binary |
| Train/Test Split | 30%/70% | Temporal split |

### Strategy Performance (volume_pct95)

| Metric | Value |
|--------|-------|
| Whales Identified | 545 |
| Trades Executed | 8,402 |
| Win Rate | 68.5% |
| Gross P&L | $53,663 |

### Best Methods

| Method | Win Rate | Net P&L/Trade |
|--------|----------|---------------|
| `win_rate_60pct` | 69.8% | $4.69 |
| `volume_pct95` | 68.5% | $2.81 |
| `combo_vol_win` | 71.4% | $3.58 |

### Category Performance

| Category | Sharpe | Win Rate |
|----------|--------|----------|
| Geopolitics | 3.33 | 70.9% |
| AI/Tech | 3.31 | 72.4% |
| Crypto | 2.66 | 68.4% |
| Sports | 0.44 | 60.3% |

### Capital Allocation

| Metric | Value |
|--------|-------|
| Avg Holding Period | ~87 days |
| Max Concurrent Positions | ~1,500+ |
| Peak Capital Required | Varies by position size |

**Capital constraint:** With long holding periods (~87 days), capital gets locked in positions. With $100k capital and $500 positions, you can hold ~200 concurrent positions.

**Warning:** Positions above $10k are unprofitable due to market impact (110%+ cost ratio). Optimal position size is $250-500.

See [WHALE_STRATEGY_RESEARCH_REPORT.md](WHALE_STRATEGY_RESEARCH_REPORT.md) for complete analysis.

## Project Structure

```
predict-ngin/
├── run.py                          # Main entry point
├── requirements.txt                # Dependencies
├── README.md                       # This file
├── WHALE_STRATEGY_RESEARCH_REPORT.md  # Full research report
│
├── src/whale_strategy/             # Core library
│   ├── __init__.py                 # Package exports
│   ├── data.py                     # Data loading (Manifold JSON)
│   ├── fetcher.py                  # API data fetching
│   ├── whales.py                   # Whale identification methods
│   ├── backtest.py                 # Bias-free backtesting engine
│   ├── costs.py                    # Polymarket cost model
│   ├── categories.py               # Market categorization
│   └── reporting.py                # QuantStats HTML reports
│
└── data/
    ├── README.md                   # Data documentation
    ├── manifold/                   # Raw Manifold Markets data
    │   ├── bets_*.json             # 5M+ bets
    │   └── markets_*.json          # 174k markets
    ├── polymarket/                 # Polymarket market data
    │   └── markets_*.json          # 151k markets
    └── output/                     # Generated reports
        ├── summary.csv             # Strategy comparison
        ├── trades_*.csv            # Trade logs
        └── quantstats_*.html       # Interactive reports
```

## Whale Identification Methods

| Method | Description | Best For |
|--------|-------------|----------|
| `volume_top10` | Top 10 by total volume | Conservative, few signals |
| `volume_pct95` | 95th percentile volume | Balanced, many signals |
| `large_trades` | Avg trade > $1000 | Large traders only |
| `win_rate_60pct` | 60%+ historical win rate | **Best P&L per trade** |
| `combo_vol_win` | High volume + good accuracy | Quality filter |

## Cost Model

Uses square-root market impact (Almgren-Chriss style):

```
Total Cost = Base Spread + Slippage + Market Impact
Market Impact = impact_coef * base_slippage * sqrt(trade_size / liquidity)
```

| Category | Position Size | Base Spread | Slippage | Min Liquidity |
|----------|--------------|-------------|----------|---------------|
| Small | $100-500 | 2.0% | 0.5% | $500 |
| Medium | $1k-5k | 2.5% | 1.0% | $5,000 |
| Large | $10k-20k | 3.0% | 2.0% | $25,000 |

## Data Sources

- **Manifold Markets**: 5M+ bets, 174k markets (play money)
- **Polymarket**: 151k markets metadata (real money)

Total data: ~4.2 GB

## Bias Prevention

| Bias Type | Mitigation |
|-----------|------------|
| Look-ahead | Train/test split (30%/70%) |
| Survivorship | All markets included |
| Selection | Whales from training only |
| Execution | Entry at probAfter price |
| Multiple entry | One position per market |

## Usage Examples

```python
from src.whale_strategy import (
    load_manifold_data,
    load_markets,
    build_resolution_map,
    identify_whales,
    run_backtest,
    CostModel,
)
from src.whale_strategy.data import train_test_split

# Load data
df = load_manifold_data("data/manifold")
markets_df = load_markets("data/manifold")
resolution_data = build_resolution_map(markets_df)

# Split data
train_df, test_df = train_test_split(df, 0.3)

# Identify whales
whales = identify_whales(train_df, resolution_data, "win_rate_60pct")

# Run backtest
result = run_backtest(
    test_df=test_df,
    whale_set=whales,
    resolution_data=resolution_data,
    strategy_name="Win Rate 60%",
    cost_model=CostModel.from_assumptions("small"),
    position_size=250,
)

print(f"Win Rate: {result.win_rate*100:.1f}%")
print(f"Net P&L: ${result.total_net_pnl:,.0f}")
print(f"Sharpe: {result.sharpe_ratio:.2f}")
```

## Example Output

```
======================================================================
WHALE FOLLOWING STRATEGY BACKTEST
======================================================================

[1] Loading data...
    Loaded 5,001,000 bets
    Date range: 2024-12-13 to 2026-01-31
    Loaded 174,400 markets

[2] Building resolution map...
    Resolved YES/NO markets: 89,765

[3] Train/test split...
    Training: 1,500,300 bets (30%)
    Testing:  3,500,700 bets (70%)

[4] Cost model: Small Retail ($100-500)

[5] Running backtests...
    Method: volume_pct95 - 95th percentile by volume
    Identified 545 whales

============================================================
STRATEGY: volume_pct95
============================================================
Unique Markets Traded:  8,402
Total Trades:           8,402
Win Rate:               68.5%
Gross P&L:              $53,662.70
```

## Output Files

After running `python run.py --all-methods --analyze-sizes --rolling`:

- `data/output/summary.csv` - Comparison of all methods
- `data/output/trades_*.csv` - Individual trade logs
- `data/output/quantstats_*.html` - Interactive QuantStats reports
- `data/output/position_size_analysis.csv` - Position size comparison
- `data/output/rolling_monthly_results.csv` - Monthly rolling backtest

Open the HTML files in a browser for detailed performance metrics, drawdown charts, and return distributions.

## License

MIT

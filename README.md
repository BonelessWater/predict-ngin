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

# Refresh data and run
python run.py --fetch --all-methods
```

## Key Findings

| Metric | Best Value | Method |
|--------|------------|--------|
| Win Rate | 69.8% | win_rate_60pct |
| Net P&L/Trade | $4.69 | win_rate_60pct |
| Sharpe Ratio | 3.33 | Geopolitics category |
| Optimal Position | $250-500 | Small retail |

**Warning:** Positions above $10k are unprofitable due to market impact (110%+ cost ratio).

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

## Output Files

After running `python run.py --all-methods`:

- `data/output/summary.csv` - Comparison of all methods
- `data/output/trades_*.csv` - Individual trade logs
- `data/output/quantstats_*.html` - Interactive QuantStats reports

Open the HTML files in a browser for detailed performance metrics, drawdown charts, and return distributions.

## License

MIT

# Prediction Market Data

This directory contains prediction market data consolidated into a SQLite database.

## Data Sources

| Source | Records | Description |
|--------|---------|-------------|
| **Manifold Markets** | ~174K markets | Play-money prediction market |
| **Polymarket** | ~2K markets | USDC prediction market |
| **Polymarket Prices** | ~279M price points | Historical CLOB price data |
| **Polymarket Trades** | ~5.6M trades | Trade history |

## Directory Structure

```
data/
|-- prediction_markets.db    # SQLite database (32 GB)
|-- output/                   # Analysis outputs (CSV, HTML reports)
`-- poly_data/                # (Optional) Raw Polymarket trade data
```

## Database Schema

### Tables

| Table | Description |
|-------|-------------|
| `manifold_markets` | Manifold market metadata (JSON blob) |
| `manifold_bets` | Manifold bet/trade records |
| `polymarket_markets` | Polymarket market metadata |
| `polymarket_prices` | CLOB price history |
| `polymarket_trades` | Processed trade records |

### Usage Examples

```python
from src.whale_strategy import load_markets, load_polymarket_markets

# Load Manifold markets (from database)
markets = load_markets()
print(f"Loaded {len(markets):,} Manifold markets")

# Load Polymarket markets (from database)
poly_markets = load_polymarket_markets()
print(f"Loaded {len(poly_markets):,} Polymarket markets")
```

### Direct SQL Queries

```python
from src.whale_strategy import PredictionMarketDB

db = PredictionMarketDB()

# Get all Polymarket markets
markets_df = db.get_all_markets()

# Get price history for a market
prices_df = db.get_price_history(market_id="12345", outcome="YES")

# Find significant price movements
changes_df = db.get_price_changes(min_change=0.05, time_window_minutes=60)

# Custom SQL query
df = db.query("SELECT COUNT(*) FROM polymarket_trades WHERE price > 0.9")

db.close()
```

## Rebuilding the Database

If you need to rebuild from source JSON files:

```bash
python run.py --build-db
```

## APIs Used (Free, No Auth Required)

| Platform | Endpoint | Description |
|----------|----------|-------------|
| Manifold | `api.manifold.markets/v0/markets` | Market metadata |
| Manifold | `api.manifold.markets/v0/bets` | Trade/bet history |
| Polymarket Gamma | `gamma-api.polymarket.com/markets` | Market metadata |
| Polymarket CLOB | `clob.polymarket.com/prices-history` | Price history |

## Data Collection Date

Data collected: January 31, 2026

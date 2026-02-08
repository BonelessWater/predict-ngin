# Prediction Market Analysis Engine

A backtesting framework for prediction market trading strategies, with whale identification and signal generation for Manifold Markets and Polymarket.

## Features

- **Whale Identification**: Find profitable traders using multiple methods (win rate, volume, trade size)
- **Backtesting Engine**: Event-driven backtest with realistic cost modeling
- **Multi-Platform**: Supports Manifold Markets (play money) and Polymarket (USDC)
- **SQLite Database**: Fast queries on 280M+ price points and 5M+ trades
- **Real-Time Streaming**: WebSocket support for live Polymarket prices

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run Manifold whale strategy backtest
python run.py --method win_rate_60pct

# Run Polymarket whale analysis
python run.py --polymarket-whales

# Build/update database from API data
python run.py --build-db
```

## Project Structure

```
predict-ngin/
|-- run.py                      # Main CLI entry point
|-- requirements.txt            # Dependencies
|
|-- src/
|   |-- trading/                # Trading engine + shared modules
|   |   |-- data_modules/        # Data loading + DB + fetchers
|   |   |-- strategies/          # Strategy implementations
|   |   |-- risk.py              # Risk modules + manager
|   |   |-- engine.py            # Event-driven backtester
|   |   |-- portfolio.py         # Position sizing & constraints
|   |   |-- reporting.py         # Performance reports
|   |   |-- signals.py           # Polymarket signal research
|   |   `-- polymarket_backtest.py  # CLOB backtester
|   |
|   `-- whale_strategy/          # Whale-specific logic
|       |-- strategy.py          # Whale identification only
|       |-- backtest.py          # Manifold whale backtest
|       `-- polymarket_whales.py # Polymarket whale identification
|
|-- scripts/                    # Utility scripts
|   |-- fetch_clob_shard.py      # Parallel CLOB data fetching
|   |-- merge_clob_shards.py     # Merge sharded data
|   `-- realtime_prices.py       # WebSocket price streaming
|
|-- tests/                      # Test suite
|   |-- test_backtest.py
|   `-- test_data.py
|
|-- docs/                       # Documentation
|   |-- WHALE_STRATEGY_RESEARCH_REPORT.md
|   `-- BACKTEST_ROADMAP.md
|
`-- data/                       # Data directory
    |-- prediction_markets.db   # SQLite database (32GB)
    `-- output/                 # Generated reports
```

## Database

All data is consolidated into a single SQLite database for fast querying.

### Tables

| Table | Records | Description |
|-------|---------|-------------|
| `manifold_markets` | 174K | Manifold market metadata |
| `manifold_bets` | 5M+ | Manifold trade history |
| `polymarket_markets` | 2K | Polymarket market metadata |
| `polymarket_prices` | 279M | CLOB price history |
| `polymarket_trades` | 5.6M | Trade history |

### Usage

```python
from trading import PredictionMarketDB, load_markets, load_polymarket_markets

# Load data (auto-uses database)
manifold_markets = load_markets()
poly_markets = load_polymarket_markets()

# Direct database queries
db = PredictionMarketDB()
df = db.query("SELECT * FROM polymarket_trades WHERE usd_amount > 1000")
prices = db.get_price_history("market_id", outcome="YES")
db.close()
```

## Whale Strategies

### Manifold Markets

```python
from trading import load_manifold_data, load_markets, build_resolution_map, train_test_split
from whale_strategy import identify_whales, run_backtest

# Load and split data
df = load_manifold_data()
markets = load_markets()
resolution_data = build_resolution_map(markets)
train_df, test_df = train_test_split(df, 0.3)

# Identify whales and backtest
whales = identify_whales(train_df, resolution_data, method="win_rate_60pct")
result = run_backtest(test_df, whales, resolution_data)
print(f"Win Rate: {result.win_rate*100:.1f}%")
```

### Polymarket

```python
from whale_strategy import (
    load_polymarket_trades,
    identify_polymarket_whales,
    generate_whale_signals,
)

# Load trades and identify whales
trades = load_polymarket_trades()
whales = identify_polymarket_whales(trades, method="win_rate_60pct")
signals = generate_whale_signals(trades, whales)

print(f"Found {len(whales)} whales")
print(f"Generated {len(signals)} signals")
```

## Whale Identification Methods

| Method | Description | Best For |
|--------|-------------|----------|
| `volume_top10` | Top 10 by total volume | Conservative |
| `volume_pct95` | 95th percentile volume | Balanced |
| `large_trades` | Avg trade > $1000 | Large traders |
| `win_rate_60pct` | 60%+ win rate | **Best accuracy** |
| `combo_vol_win` | High volume + good win rate | Quality filter |

## Data Fetching

### Fetch from APIs

```bash
# Fetch all Manifold data
python run.py --fetch

# Fetch Polymarket CLOB price history
python run.py --fetch-clob --clob-min-volume 5000 --clob-max-markets 500

# Parallel fetching (4 terminals)
python scripts/fetch_clob_shard.py 0 4 5000 1000  # Terminal 1
python scripts/fetch_clob_shard.py 1 4 5000 1000  # Terminal 2
python scripts/fetch_clob_shard.py 2 4 5000 1000  # Terminal 3
python scripts/fetch_clob_shard.py 3 4 5000 1000  # Terminal 4
python scripts/merge_clob_shards.py 4             # Merge results
```

### Real-Time Streaming

```bash
pip install websockets
python scripts/data/realtime_prices.py
```

### Market Scanner

Scan Polymarket markets for opportunities, high-volume markets, or expiring soon.

```bash
# Top opportunity scores (default)
python -m src.tools.market_scanner --top 20

# Highest 24h volume markets
python -m src.tools.market_scanner --mode high-volume --top 15

# Markets expiring within 5 days
python -m src.tools.market_scanner --mode expiring --expiring-within 5 --top 20

# Run multiple reports in one call
python -m src.tools.market_scanner --mode opportunities --mode expiring --top 15
```

### Whale Trade Discord Alerts (Real-Time)

This uses the Polymarket market WebSocket and flags large last-trade events.
The market channel does not include trader addresses, so "whale" means large
notional size based on price * size.

Set these environment variables (or put them in `.env`):

```
DISCORD_BOT_TOKEN=...
DISCORD_CHANNEL_ID=...
```

Run a test message:

```bash
python scripts/monitoring/whale_trade_notifier.py --test
```

Start streaming with alerts:

```bash
python scripts/monitoring/whale_trade_notifier.py --min-usd 5000 --min-volume 20000
```

### Trade Alert Bot (Telegram/Discord)

Stream trade notifications from local logs (paper trading + execution logger).
Supports Telegram and Discord (webhook or bot token).

Set these environment variables (or put them in `.env`):

```
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
DISCORD_WEBHOOK_URL=...
# OR
DISCORD_BOT_TOKEN=...
DISCORD_CHANNEL_ID=...
```

Run with defaults (tails `data/paper_trading_log.jsonl` and `data/execution_log.jsonl`):

```bash
python scripts/monitoring/trade_alert_bot.py
```

Optional flags:

```bash
# Read existing logs from the start
python scripts/monitoring/trade_alert_bot.py --from-start

# Only alert on trades >= $500
python scripts/monitoring/trade_alert_bot.py --min-size 500

# Specify a custom log file
python scripts/monitoring/trade_alert_bot.py --log data/paper_trading_log.jsonl
```

### Live Whale Tracker (Address-Based)

Track known whale addresses in near-real time via the Polymarket CLOB
trades endpoint. This requires CLOB API credentials.

Set these environment variables (or put them in `.env`):

```
POLYMARKET_API_KEY=...
POLYMARKET_API_SECRET=...
POLYMARKET_PASSPHRASE=...
POLYMARKET_ADDRESS=...   # optional, if required by your API setup
```

Examples:

```bash
# Load whales from DB (top 25 by volume)
python scripts/monitoring/live_whale_tracker.py --db-path data/prediction_markets.db --max-whales 25

# Use a watchlist file (one address per line)
python scripts/monitoring/live_whale_tracker.py --whales-file data/whales.txt --min-usd 1000 --role taker
```

## Cost Model

Transaction costs use a square-root market impact model:

```
Total Cost = Spread + Slippage + Market Impact
Impact = coefficient * sqrt(trade_size / liquidity)
```

| Size | Spread | Slippage | Min Liquidity |
|------|--------|----------|---------------|
| Small ($100-500) | 2.0% | 0.5% | $500 |
| Medium ($1k-5k) | 2.5% | 1.0% | $5,000 |
| Large ($10k+) | 3.0% | 2.0% | $25,000 |

## CLI Reference

```bash
# Manifold backtest
python run.py --method win_rate_60pct --position-size 250
python run.py --all-methods --analyze-sizes
python run.py --rolling --lookback 3

# Data management
python run.py --fetch              # Fetch Manifold data
python run.py --fetch-clob         # Fetch Polymarket CLOB data
python run.py --build-db           # Build SQLite database

# Polymarket analysis
python run.py --polymarket-whales
```

## Key Findings

### Manifold Markets (5M bets, 174K markets)

| Method | Whales | Win Rate | Sharpe |
|--------|--------|----------|--------|
| win_rate_60pct | 583 | 69.8% | 2.4 |
| volume_pct95 | 545 | 68.5% | 2.1 |
| combo_vol_win | 312 | 71.4% | 2.6 |

### Polymarket (5.6M trades, 102K traders)

| Metric | Value |
|--------|-------|
| Whales (60%+ WR) | 1,423 |
| Avg Whale Win Rate | 77.8% |
| Total Whale Volume | $187M |

## Documentation

- [Whale Strategy Research Report](docs/WHALE_STRATEGY_RESEARCH_REPORT.md) - Full analysis
- [Backtest Roadmap](docs/BACKTEST_ROADMAP.md) - Building production backtester
- [Data README](data/README.md) - Database schema and usage

## License

MIT

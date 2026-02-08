# Scripts Directory

Organized collection of utility scripts for the prediction market backtesting engine.

## Directory Structure

```
scripts/
├── backtest/          # Backtesting and strategy evaluation
├── data/              # Data collection and fetching
├── analysis/           # Data analysis and research
├── monitoring/         # Live trading monitoring and alerts
└── README.md          # This file
```

## Categories

### 📊 Backtest (`backtest/`)

Scripts for running backtests, comparing results, and managing backtest runs.

- **`run_momentum_backtest.py`** - Run momentum strategy backtest with QuantStats reporting
- **`compare_backtests.py`** - Compare multiple backtest runs side-by-side
- **`list_backtests.py`** - Search and list past backtest runs

**Usage:**
```bash
# Run a backtest
python scripts/backtest/run_momentum_backtest.py --threshold 0.05

# List runs
python scripts/backtest/list_backtests.py --strategy momentum

# Compare runs
python scripts/backtest/compare_backtests.py run_id1 run_id2
```

### 📥 Data (`data/`)

Scripts for fetching and collecting market data from various sources.

- **`download_hf_polymarket.py`** - Download Polymarket datasets from HuggingFace
- **`fetch_polymarket_trades.py`** - Fetch recent trades from Polymarket API
- **`realtime_prices.py`** - Stream real-time prices via WebSocket

**Usage:**
```bash
# Download HuggingFace datasets
python scripts/data/download_hf_polymarket.py

# Fetch recent trades
python scripts/data/fetch_polymarket_trades.py --days 30

# Stream real-time prices
python scripts/data/realtime_prices.py
```

### 🔬 Analysis (`analysis/`)

Scripts for analyzing market data, categories, and generating research reports.

- **`analyze_categories.py`** - Analyze market categories and data volume per category
- **`analyze_category_details.py`** - Detailed category analysis with breakdowns

**Usage:**
```bash
# Analyze categories
python scripts/analysis/analyze_categories.py

# Detailed category analysis
python scripts/analysis/analyze_category_details.py --category politics
```

### 🔔 Monitoring (`monitoring/`)

Scripts for live trading monitoring, whale tracking, and trade alerts.

- **`live_whale_tracker.py`** - Track known whale addresses in real-time
- **`whale_trade_notifier.py`** - Discord alerts for large whale trades
- **`trade_alert_bot.py`** - Telegram/Discord bot for trade notifications

**Usage:**
```bash
# Track whales
python scripts/monitoring/live_whale_tracker.py --max-whales 25

# Whale trade notifications
python scripts/monitoring/whale_trade_notifier.py --min-usd 5000

# Trade alert bot
python scripts/monitoring/trade_alert_bot.py
```

## Quick Reference

| Category | Purpose | Key Scripts |
|----------|---------|-------------|
| **backtest** | Strategy evaluation | `run_momentum_backtest.py`, `compare_backtests.py` |
| **data** | Data collection | `fetch_polymarket_trades.py`, `realtime_prices.py` |
| **analysis** | Research & analysis | `analyze_categories.py` |
| **monitoring** | Live tracking | `live_whale_tracker.py`, `trade_alert_bot.py` |

## Adding New Scripts

When adding a new script:

1. **Choose the right category:**
   - Backtesting → `backtest/`
   - Data fetching → `data/`
   - Analysis/research → `analysis/`
   - Live monitoring → `monitoring/`

2. **Update path references:**
   ```python
   # Scripts are now in subdirectories, so use:
   _project_root = Path(__file__).resolve().parent.parent.parent
   ```

3. **Update this README** with the new script's purpose and usage.

## Path References

All scripts use a consistent pattern for importing project modules:

```python
from pathlib import Path
import sys

# Get project root (3 levels up from scripts/subdir/script.py)
_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "src"))

# Now you can import from src
from trading import ...
from src.backtest import ...
```

## See Also

- `docs/BACKTEST_ENGINE_TYPES.md` - Types of backtesting engines
- `backtests/README.md` - Backtest result storage
- `README.md` - Main project documentation

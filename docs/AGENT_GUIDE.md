# Agent Guide: Repository Structure & Components

**⚠️ IMPORTANT: Update this file whenever you make significant changes to the repository structure, add new components, or modify existing ones.**

**Last Updated:** 2026-02-07  
**Last Updated By:** AI Agent (Composer)  
**Update Reason:** Initial creation + scripts reorganization + performance analysis + trade-based backtesting

---

## Purpose

This document provides a comprehensive guide to the repository structure, explaining what each component is for and how it fits into the overall system. This is intended for AI agents and developers to quickly understand the codebase.

---

## Repository Overview

**Project Name:** predict-ngin  
**Type:** Prediction Market Backtesting & Research Engine  
**Primary Use Case:** Backtesting trading strategies on prediction markets (Polymarket, Manifold Markets)

---

## Directory Structure

### Root Level

```
predict-ngin/
├── src/                    # Source code (main application)
├── scripts/                # Utility scripts (organized by category)
├── tests/                  # Test suite
├── config/                 # Configuration files (YAML)
├── data/                   # Data storage (databases, parquet files)
├── backtests/              # Organized backtest results
├── docs/                   # Documentation
├── examples/               # Example usage scripts
├── logs/                   # Log files
├── memory/                 # Agent memory files (daily notes)
├── run.py                  # Main CLI entry point
├── requirements.txt        # Python dependencies
└── README.md              # Main project documentation
```

---

## Core Components

### `/src` - Source Code

Main application code organized by functionality:

#### `/src/trading/` - Trading Engine & Strategies

**Purpose:** Core trading logic, strategies, and execution

- **`engine.py`** - Event-driven trading engine (processes signals, applies risk management)
- **`polymarket_backtest.py`** - Signal-based backtest for Polymarket CLOB data
- **`strategies/`** - Strategy implementations:
  - `momentum.py` - Momentum-based signals
  - `whale.py` - Whale following strategy
  - `breakout.py` - Breakout detection
  - `mean_reversion.py` - Mean reversion signals
  - `sentiment.py` - Sentiment analysis
  - `time_decay.py` - Time decay strategies
  - `composite.py` - Composite/multi-strategy
  - `cross_market.py` - Cross-market correlation
  - `smart_money.py` - Smart money detection
- **`risk.py`** - Risk management modules
- **`portfolio.py`** - Position sizing and portfolio constraints
- **`reporting.py`** - Performance reporting and metrics
- **`data_modules/`** - Data access layer:
  - `database.py` - SQLite database interface
  - `parquet_store.py` - Parquet file storage
  - `trade_price_store.py` - Extract prices from trades (no price files needed)
  - `fetcher.py` - Data fetching utilities
  - `categories.py` - Market categorization

#### `/src/backtest/` - Backtesting Infrastructure

**Purpose:** Backtest engines, validation, and result management

- **`engine.py`** - Re-exports from trading engine (backward compatibility)
- **`walk_forward.py`** - Walk-forward validation for robustness testing
- **`costs.py`** - Cost model definitions
- **`storage.py`** - Backtest result storage (hierarchical, organized)
- **`catalog.py`** - Searchable catalog of all backtests
- **`comparison.py`** - Comparison utilities for multiple runs

#### `/src/whale_strategy/` - Whale Strategy Implementation

**Purpose:** Specific implementation for whale following strategies

- **`strategy.py`** - Whale identification logic
- **`backtest.py`** - Bias-free backtest for whale strategies
- **`polymarket_whales.py`** - Polymarket-specific whale identification
- **`polymarket_realistic_backtest.py`** - Realistic backtest with CLOB data
- **`correlation.py`** - Whale correlation analysis
- **`research_data_loader.py`** - Load trades/prices from data/research by category
- **`whale_scoring.py`** - Whale quality score (W) per strategy spec
- **`whale_following_strategy.py`** - Category-level limits, Kelly sizing, last-signal-wins

#### `/src/experiments/` - Experiment Tracking

**Purpose:** Research reproducibility and experiment management

- **`tracker.py`** - ExperimentTracker class (SQLite + filesystem)
- **`notebooks.py`** - Notebook archive management

#### `/src/research/` - Research Tools

**Purpose:** Research and analysis utilities

- **`alpha_discovery.py`** - Alpha signal discovery
- **`ensemble_research.py`** - Ensemble strategy research
- **`regime_detection.py`** - Market regime detection
- **`whale_features.py`** - Whale feature engineering

#### `/src/storage/` - Storage Abstractions

**Purpose:** Storage layer abstractions

- **`base.py`** - Base storage interface
- **`database.py`** - Database storage implementation
- **`parquet.py`** - Parquet storage implementation
- **`liquidity.py`** - Liquidity data storage

#### `/src/collection/` - Data Collection

**Purpose:** Data collection from external sources

- **`base.py`** - Base collector interface
- **`prices.py`** - Price data collection
- **`trades.py`** - Trade data collection
- **`liquidity.py`** - Liquidity data collection

#### `/src/analysis/` - Analysis Tools

**Purpose:** Statistical analysis and distributions

- **`distributions.py`** - Distribution analysis

#### `/src/validation/` - Data Validation

**Purpose:** Data quality and validation

- **`assertions.py`** - Data assertions
- **`quality.py`** - Quality monitoring

#### `/src/taxonomy/` - Market Taxonomy

**Purpose:** Market categorization and taxonomy

- **`markets.py`** - Market taxonomy definitions

#### `/src/tools/` - Utility Tools

**Purpose:** General utility tools

- **`market_scanner.py`** - Market opportunity scanner

---

### `/scripts` - Utility Scripts

**Purpose:** Organized collection of utility scripts for various tasks

**Structure:** Organized into categories to prevent clutter

#### `/scripts/backtest/` - Backtesting Scripts

- **`run_momentum_backtest.py`** - Run momentum strategy backtest with QuantStats
- **`run_whale_category_backtest.py`** - Whale-following backtest at category level (data/research)
- **`compare_backtests.py`** - Compare multiple backtest runs side-by-side
- **`list_backtests.py`** - Search and list past backtest runs

#### `/scripts/data/` - Data Collection Scripts

- **`fetch_research_trades_and_prices.py`** - Fetch trades/prices into data/research (primary)
- **`collect_research_by_market_list.py`** - Filter markets to top N, populate data/research
- **`fetch_user_activity_summary.py`** - User activity into data/research/users
- **`download_hf_polymarket.py`** - Download Polymarket datasets from HuggingFace
- **`fetch_polymarket_trades.py`** - Fetch recent trades from Polymarket API
- **`realtime_prices.py`** - Stream real-time prices via WebSocket

#### `/scripts/analysis/` - Analysis Scripts (data/research workflow)

- **`analyze_categories.py`** - Analyze market categories and data volume
- **`run_full_user_research.py`** - Full user-behavior research from data/research
- **`analyze_volume_and_trade_size.py`** - Volume and trade size analysis
- **`user_profile_summary.py`** - Per-user profile summaries
- **`analyze_users_per_market.py`** - User activity per market
- **`price_change_over_market_period.py`** - Price change over market lifecycle
- **`top500_volume_share.py`** - Top 500 volume share by category
- **`whale_category_research.py`** - Whale performance by category, score distribution
- **`volume_distribution.py`** - Volume distribution analysis
- **`holistic_market_research.py`** - Holistic market research

#### `/scripts/monitoring/` - Live Monitoring Scripts

- **`live_whale_tracker.py`** - Track known whale addresses in real-time
- **`whale_trade_notifier.py`** - Discord alerts for large whale trades
- **`trade_alert_bot.py`** - Telegram/Discord bot for trade notifications

**Note:** All scripts use `parent.parent.parent` for path resolution since they're in subdirectories.

---

### `/config` - Configuration Files

**Purpose:** YAML-based configuration for the application

- **`default.yaml`** - Default configuration (version controlled)
- **`local.yaml`** - Local overrides (gitignored, user-specific)
- **`taxonomy.yaml`** - Market taxonomy definitions

**Usage:** Configuration is loaded via `src/config.py` which merges default + local.

---

### `/data` - Data Storage

**Purpose:** Canonical data lives in `data/research`. DB generation is deprecated.

**Structure:**

```
data/
└── research/                # Canonical research data (ONLY source)
    ├── {category}/          # Tech, Politics, Finance, etc.
    │   ├── markets_filtered.csv
    │   ├── trades.parquet
    │   └── prices.parquet
    ├── users/               # User activity summaries
    ├── user_research/       # Analysis outputs
    ├── features/            # Market features
    ├── similarities/        # Market similarities
    └── correlations/        # Price correlations
```

**Note:** See `data/research/README.md` for how to populate. `prediction_markets.db` is deprecated.

---

### `/backtests` - Backtest Results

**Purpose:** Organized storage of backtest results with full reproducibility

**Structure:**

```
backtests/
├── {strategy_name}/
│   ├── {run_id}/            # e.g., momentum_20240207_143022_a1b2c3
│   │   ├── metadata.json    # Full run metadata
│   │   ├── config.yaml      # Config snapshot
│   │   ├── results/         # Trades, summary, reports
│   │   ├── signals/         # Input signals (if applicable)
│   │   └── logs/           # Run logs
│   └── latest -> {run_id}   # Symlink to most recent
├── catalog.db               # Searchable SQLite index
└── README.md               # Backtest storage guide
```

**Features:**

- Unique run IDs prevent overwrites
- Full metadata (git commit, config, environment)
- Searchable catalog for fast discovery
- Comparison tools for analyzing runs

---

### `/tests` - Test Suite

**Purpose:** Unit and integration tests

**Structure:**

- **`test_*.py`** - Test files matching source modules
- **`unit/`** - Unit tests for specific components
- **`conftest.py`** - Pytest configuration and fixtures

---

### `/docs` - Documentation

**Purpose:** Comprehensive documentation

**Key Documents:**

- **`AGENT_GUIDE.md`** - This file (repository structure guide)
- **`BACKTEST_ENGINE_TYPES.md`** - Types of backtesting engines and use cases
- **`BACKTEST_STRUCTURE_ANALYSIS.md`** - Backtest storage analysis
- **`DATA_STORAGE_PLAN_100GB.md`** - Data storage strategy for large datasets
- **`PERFORMANCE_OPTIMIZATIONS_100GB.md`** - Performance bottlenecks and optimizations
- **`PERFORMANCE_QUICK_FIXES.md`** - Quick fixes for faster backtests
- **`SCRIPTS_ORGANIZATION.md`** - Scripts organization guide
- **`IMPLEMENTATION_SUMMARY.md`** - Implementation details
- **`results/`** - Research results and reports

---

### `/examples` - Example Scripts

**Purpose:** Example usage scripts for common tasks

- **`run_strategy.py`** - Example of running a strategy

---

### `/memory` - Agent Memory

**Purpose:** Daily memory files for AI agents (per AGENTS.md rules)

- **`YYYY-MM-DD.md`** - Daily memory files
- Used for continuity across sessions

---

## Key Files

### `run.py` - Main CLI Entry Point

**Purpose:** Command-line interface for common operations

**Key Commands:**

- `--fetch` - Fetch Manifold data
- `--fetch-clob` - Fetch Polymarket CLOB data
- `--build-db` - Build SQLite database
- `--method` - Run whale strategy backtest
- `--polymarket-whales` - Analyze Polymarket whales

---

### `requirements.txt` - Dependencies

**Purpose:** Python package dependencies

**Key Dependencies:**

- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `sqlite3` - Database (built-in)
- `pyarrow` - Parquet file support
- `polars` - Fast data processing
- `quantstats` - Performance analytics
- `pyyaml` - Configuration files

---

## Backtesting Engines

The repository contains multiple backtesting engines for different use cases:

1. **Event-Driven Engine** (`TradingEngine`) - Multi-strategy, realistic simulation
2. **Signal-Based Backtest** (`run_polymarket_backtest`) - Fast signal research
3. **Whale Strategy Backtest** - Bias-free whale following
4. **Walk-Forward Validation** - Robustness testing

See `docs/BACKTEST_ENGINE_TYPES.md` for detailed explanations.

---

## Data Flow

1. **Collection** → Scripts fetch data from APIs → Store in SQLite/Parquet
2. **Processing** → Strategies generate signals → Backtest engines execute
3. **Storage** → Results saved to `/backtests` → Indexed in catalog
4. **Analysis** → Comparison tools → Research reports

---

## Configuration System

**Location:** `/config/`

**How it works:**

1. `default.yaml` provides base configuration
2. `local.yaml` (gitignored) provides user overrides
3. `src/config.py` merges them with local taking precedence

**Key Sections:**

- `database` - Database paths and settings
- `data` - Data collection settings
- `strategies` - Strategy parameters
- `backtest` - Backtest configuration
- `whale_strategy` - Whale strategy settings

---

## Experiment Tracking

**Location:** `src/experiments/tracker.py`

**Purpose:** Track experiments for reproducibility

**Features:**

- SQLite metadata storage
- Artifact management
- Git commit tracking
- Search and comparison

**Integration:** Automatically integrated into backtest scripts.

---

## Important Patterns

### Path Resolution

Scripts in subdirectories use:

```python
_project_root = Path(__file__).resolve().parent.parent.parent
```

### Import Patterns

```python
# From src modules
from trading import ...
from src.backtest import ...

# From project root
from config import get_config
```

### Data Access

- **SQLite:** Use `PredictionMarketDB` from `trading.data_modules.database`
- **Parquet:** Use `PriceStore`, `TradeStore` from `trading.data_modules.parquet_store`
- **Auto-detection:** Code auto-detects parquet vs SQLite based on file existence

---

## Common Tasks

### Running a Backtest

```bash
python scripts/backtest/run_momentum_backtest.py --threshold 0.05
```

### Fetching Data

```bash
python scripts/data/fetch_polymarket_trades.py --days 30
```

### Analyzing Categories

```bash
python scripts/analysis/analyze_categories.py
```

### Monitoring Whales

```bash
python scripts/monitoring/live_whale_tracker.py --max-whales 25
```

---

## Testing

**Location:** `/tests/`

**Run tests:**

```bash
pytest tests/
```

**Test structure matches source structure for easy navigation.**

---

## Documentation Standards

- **README.md** - Main project documentation
- **docs/** - Detailed documentation
- **Docstrings** - In-code documentation
- **Type hints** - Type annotations for clarity

---

## Git Structure

**Branches:**

- `main` - Main development branch

**Ignored:**

- `data/prediction_markets.db` - Large database file
- `data/parquet/` - Large parquet files
- `config/local.yaml` - User-specific config
- `logs/` - Log files
- `backtests/` - Backtest results (can be large)

---

## Agent-Specific Notes

### Memory Files

- **Location:** `/memory/`
- **Format:** `YYYY-MM-DD.md`
- **Purpose:** Daily notes for continuity
- **See:** `AGENTS.md` for rules

### Workspace Rules

- **Location:** `AGENTS.md` (root)
- **Purpose:** Agent behavior rules and conventions
- **Read:** Before starting work

---

## Update Instructions

**When to update this file:**

- ✅ Adding new major components or directories
- ✅ Reorganizing directory structure
- ✅ Adding new script categories
- ✅ Changing data storage structure
- ✅ Adding new backtesting engines
- ✅ Significant architectural changes

**How to update:**

1. Update the relevant section
2. Update "Last Updated" date at top
3. Add entry to "Update Reason" explaining what changed
4. Keep structure consistent with existing format

---

## Quick Reference

| Component        | Location                  | Purpose                           |
| ---------------- | ------------------------- | --------------------------------- |
| Trading Engine   | `src/trading/engine.py`   | Event-driven backtesting          |
| Strategies       | `src/trading/strategies/` | Strategy implementations          |
| Backtest Storage | `src/backtest/storage.py` | Result storage                    |
| Backtest Catalog | `src/backtest/catalog.py` | Searchable index                  |
| Data             | `data/research`           | Canonical research data (parquet) |
| Config           | `config/default.yaml`     | Configuration                     |
| Scripts          | `scripts/{category}/`     | Utility scripts                   |
| Tests            | `tests/`                  | Test suite                        |

---

## Questions?

If something is unclear or missing:

1. Check the relevant documentation in `/docs`
2. Check code docstrings
3. Update this file with the clarification

---

**Remember:** Keep this file updated as the repository evolves!

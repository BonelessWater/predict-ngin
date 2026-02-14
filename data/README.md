# Data Directory

**Canonical data source:** `data/research`

All analysis and research use data from `data/research`. Database generation (`prediction_markets.db`) is deprecated.

## Layout

```
data/
└── research/                    # Canonical research data
    ├── {category}/              # e.g. Tech, Politics, Finance
    │   ├── markets_filtered.csv
    │   ├── trades.parquet
    │   └── prices.parquet
    ├── users/
    │   ├── user_activity_summary.json
    │   ├── user_stats.csv
    │   └── activity_*.json
    ├── user_research/           # User behavior analysis outputs
    ├── user_profiles/           # Per-user profile summaries
    ├── features/                # Market features (extract_market_features)
    ├── similarities/           # Market similarities
    ├── correlations/            # Price correlations
    └── checkpoints/             # Research script checkpoints
```

## Populating data/research

1. **Market list:** Place `Polymarket/{Category}/markets.csv` at repo root.
2. **Filter to top 500:** `python scripts/data/collect_research_by_market_list.py --top-n 500 --markets-only`
3. **Fetch trades and prices:** `python scripts/data/fetch_research_trades_and_prices.py --research-dir data/research`
4. **User activity:** `python scripts/data/fetch_user_activity_summary.py --output-dir data/research/users`

See `data/research/README.md` for full details.

## Deprecated

- `prediction_markets.db` — Do not generate. Use parquet in `data/research` instead.
- `data/polymarket/`, `data/parquet/` — Legacy paths. All data lives in `data/research`.

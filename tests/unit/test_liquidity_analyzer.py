"""
Tests for liquidity analyzer.
"""

import pytest
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
from unittest.mock import patch, MagicMock
import sys

# Add project root to path
_project_root = Path(__file__).resolve().parent.parent.parent
SRC = _project_root / "src"
sys.path.insert(0, str(_project_root))

# Import directly from module file to avoid package import issues
import importlib.util
liquidity_path = SRC / "trading" / "data_modules" / "liquidity.py"
spec = importlib.util.spec_from_file_location("liquidity", liquidity_path)
liquidity_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(liquidity_module)
init_liquidity_schema = liquidity_module.init_liquidity_schema


@pytest.fixture
def sample_liquidity_db(tmp_path):
    """Create a sample database with liquidity data."""
    db_path = tmp_path / "test_liquidity.db"
    
    # Initialize schema
    init_liquidity_schema(str(db_path))
    
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # Insert sample snapshots
    base_time = datetime(2026, 1, 15, 12, 0, 0)
    for i in range(10):
        timestamp = int((base_time + timedelta(hours=i)).timestamp())
        cursor.execute("""
            INSERT INTO liquidity_snapshots
            (market_id, token_id, outcome, timestamp, datetime,
             bid_depth_1pct, bid_depth_5pct, bid_depth_10pct,
             ask_depth_1pct, ask_depth_5pct, ask_depth_10pct,
             spread, midpoint, best_bid, best_ask)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            f"market_{i % 3:03d}",
            f"token_{i}",
            "YES",
            timestamp,
            (base_time + timedelta(hours=i)).isoformat(),
            1000.0 + i * 100,
            5000.0 + i * 500,
            10000.0 + i * 1000,
            1000.0 + i * 100,
            5000.0 + i * 500,
            10000.0 + i * 1000,
            0.01 + i * 0.001,
            0.50,
            0.495,
            0.505,
        ))
    
    # Insert sample estimates
    for i in range(5):
        cursor.execute("""
            INSERT INTO liquidity_estimates
            (market_id, date, estimated_liquidity, avg_trade_impact,
             trade_count, total_volume, volatility)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            f"market_{i % 3:03d}",
            (base_time.date() + timedelta(days=i)).isoformat(),
            50000.0 + i * 10000,
            0.001 + i * 0.0001,
            100 + i * 10,
            100000.0 + i * 20000,
            0.02,
        ))
    
    conn.commit()
    conn.close()
    
    return str(db_path)


def test_load_liquidity_snapshots(sample_liquidity_db):
    """Test loading liquidity snapshots."""
    from scripts.analysis.liquidity_analyzer import load_liquidity_snapshots
    
    df = load_liquidity_snapshots(sample_liquidity_db)
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 10
    assert "market_id" in df.columns
    assert "bid_depth_10pct" in df.columns
    assert "spread" in df.columns


def test_load_liquidity_snapshots_filtered(sample_liquidity_db):
    """Test loading liquidity snapshots with filters."""
    from scripts.analysis.liquidity_analyzer import load_liquidity_snapshots
    
    # Filter by market
    df = load_liquidity_snapshots(sample_liquidity_db, market_id="market_000")
    
    assert len(df) > 0
    assert all(df["market_id"] == "market_000")
    
    # Filter by date
    start_date = datetime(2026, 1, 15, 14, 0, 0).isoformat()
    df = load_liquidity_snapshots(sample_liquidity_db, start_date=start_date)
    
    assert len(df) > 0


def test_load_liquidity_estimates(sample_liquidity_db):
    """Test loading liquidity estimates."""
    from scripts.analysis.liquidity_analyzer import load_liquidity_estimates
    
    df = load_liquidity_estimates(sample_liquidity_db)
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5
    assert "market_id" in df.columns
    assert "estimated_liquidity" in df.columns


def test_analyze_liquidity_trends(sample_liquidity_db):
    """Test liquidity trends analysis."""
    from scripts.analysis.liquidity_analyzer import (
        load_liquidity_snapshots,
        analyze_liquidity_trends,
    )
    
    snapshots_df = load_liquidity_snapshots(sample_liquidity_db)
    trends = analyze_liquidity_trends(snapshots_df)
    
    assert isinstance(trends, dict)
    assert len(trends) > 0
    
    # Check trend structure
    for market_id, trend_data in trends.items():
        assert "avg_bid_depth" in trend_data
        assert "avg_ask_depth" in trend_data
        assert "avg_spread" in trend_data
        assert "snapshot_count" in trend_data


def test_analyze_market_liquidity(sample_liquidity_db):
    """Test market liquidity analysis."""
    from scripts.analysis.liquidity_analyzer import (
        load_liquidity_snapshots,
        load_liquidity_estimates,
        analyze_market_liquidity,
    )
    
    snapshots_df = load_liquidity_snapshots(sample_liquidity_db)
    estimates_df = load_liquidity_estimates(sample_liquidity_db)
    
    market_df = analyze_market_liquidity(snapshots_df, estimates_df)
    
    assert isinstance(market_df, pd.DataFrame)
    assert len(market_df) > 0
    assert "market_id" in market_df.columns
    assert "avg_spread" in market_df.columns
    assert "snapshot_count" in market_df.columns


def test_identify_liquidity_issues(sample_liquidity_db):
    """Test liquidity issue identification."""
    from scripts.analysis.liquidity_analyzer import (
        load_liquidity_snapshots,
        identify_liquidity_issues,
    )
    
    snapshots_df = load_liquidity_snapshots(sample_liquidity_db)
    issues = identify_liquidity_issues(
        snapshots_df,
        min_depth_threshold=5000.0,
        max_spread_threshold=0.02,
    )
    
    assert isinstance(issues, pd.DataFrame)
    # Should have columns for issues
    if len(issues) > 0:
        assert "market_id" in issues.columns
        assert "low_depth_snapshots" in issues.columns


def test_identify_liquidity_issues_empty():
    """Test liquidity issue identification with empty data."""
    from scripts.analysis.liquidity_analyzer import identify_liquidity_issues
    
    empty_df = pd.DataFrame()
    issues = identify_liquidity_issues(empty_df)
    
    assert isinstance(issues, pd.DataFrame)
    assert len(issues) == 0


@patch("scripts.analysis.liquidity_analyzer.get_config")
def test_generate_liquidity_report(mock_config, sample_liquidity_db, tmp_path, capsys):
    """Test liquidity report generation."""
    from scripts.analysis.liquidity_analyzer import generate_liquidity_report
    
    mock_config.return_value.database.path = sample_liquidity_db
    
    output_path = tmp_path / "liquidity_report.txt"
    
    generate_liquidity_report(
        db_path=sample_liquidity_db,
        output_path=str(output_path),
    )
    
    # Check output file was created
    assert output_path.exists()
    
    # Check console output
    captured = capsys.readouterr()
    assert "liquidity" in captured.out.lower() or "snapshot" in captured.out.lower()


def test_generate_liquidity_report_empty_db(tmp_path, capsys):
    """Test report generation with empty database."""
    from scripts.analysis.liquidity_analyzer import generate_liquidity_report
    
    # Create empty database
    empty_db = tmp_path / "empty.db"
    init_liquidity_schema(str(empty_db))
    
    generate_liquidity_report(
        db_path=str(empty_db),
    )
    
    captured = capsys.readouterr()
    assert "no liquidity" in captured.out.lower() or "not found" in captured.out.lower()

"""
Reporting utilities including QuantStats HTML report generation.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, List
import warnings


def generate_quantstats_report(
    daily_returns: pd.Series,
    output_path: str,
    title: str = "Whale Strategy Report",
    benchmark: Optional[pd.Series] = None,
) -> bool:
    """
    Generate an interactive QuantStats HTML report.

    Args:
        daily_returns: Series of daily returns indexed by date
        output_path: Path to save HTML report
        title: Report title
        benchmark: Optional benchmark returns for comparison

    Returns:
        True if successful, False otherwise
    """
    try:
        import quantstats as qs
    except ImportError:
        warnings.warn(
            "quantstats not installed. Run: pip install quantstats"
        )
        return False

    if len(daily_returns) < 5:
        warnings.warn("Not enough data points for QuantStats report")
        return False

    # Ensure proper datetime index
    returns = pd.Series(
        daily_returns.values,
        index=pd.to_datetime(daily_returns.index)
    ).sort_index()

    try:
        qs.reports.html(
            returns,
            output=output_path,
            title=title,
            benchmark=benchmark,
        )
        return True
    except Exception as e:
        warnings.warn(f"Failed to generate QuantStats report: {e}")
        return False


def save_trades_csv(
    trades_df: pd.DataFrame,
    output_path: str
) -> None:
    """Save trades DataFrame to CSV."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    trades_df.to_csv(output_path, index=False)


def save_summary_csv(
    results: List[dict],
    output_path: str
) -> None:
    """Save summary results to CSV."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(output_path, index=False)


def generate_all_reports(
    results: dict,
    output_dir: str = "data/output",
) -> None:
    """
    Generate all reports for backtest results.

    Args:
        results: Dictionary mapping strategy name to BacktestResult
        output_dir: Output directory for reports
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary = []

    for name, result in results.items():
        if result is None:
            continue

        # Save trades
        safe_name = name.replace(" ", "_").lower()
        result.trades_df.to_csv(
            output_path / f"trades_{safe_name}.csv",
            index=False
        )

        # Generate QuantStats report
        generate_quantstats_report(
            result.daily_returns,
            str(output_path / f"quantstats_{safe_name}.html"),
            title=f"Whale Strategy: {name}"
        )

        # Add to summary
        summary.append({
            "strategy": name,
            "trades": result.total_trades,
            "win_rate": f"{result.win_rate*100:.1f}%",
            "gross_pnl": result.total_gross_pnl,
            "net_pnl": result.total_net_pnl,
            "costs": result.total_costs,
            "sharpe": result.sharpe_ratio,
            "max_dd": result.max_drawdown,
            "profit_factor": result.profit_factor,
        })

    # Save summary
    pd.DataFrame(summary).to_csv(
        output_path / "summary.csv",
        index=False
    )

    print(f"\nReports saved to {output_dir}/")
    print(f"  - summary.csv")
    for name in results.keys():
        if results[name] is not None:
            safe_name = name.replace(" ", "_").lower()
            print(f"  - trades_{safe_name}.csv")
            print(f"  - quantstats_{safe_name}.html")

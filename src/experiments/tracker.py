"""
Experiment tracking for research reproducibility.

Tracks experiment runs, parameters, metrics, and artifacts.
Uses SQLite for metadata and filesystem for artifacts.
"""

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Union
import hashlib
import json
import logging
import shutil
import sqlite3
import subprocess
import uuid

import pandas as pd


@dataclass
class ExperimentRun:
    """
    Represents a single experiment run.

    Attributes:
        run_id: Unique identifier for this run
        name: Human-readable experiment name
        timestamp: When the run started
        parameters: Experiment parameters/config
        git_commit: Git commit hash (if available)
        metrics: Logged metrics (e.g., sharpe, win_rate)
        artifacts: Mapping of artifact names to file paths
        tags: Categorical tags for filtering
        status: running, completed, or failed
    """

    run_id: str
    name: str
    timestamp: datetime
    parameters: Dict[str, Any] = field(default_factory=dict)
    git_commit: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    status: str = "running"  # running, completed, failed
    end_time: Optional[datetime] = None
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "run_id": self.run_id,
            "name": self.name,
            "timestamp": self.timestamp.isoformat(),
            "parameters": json.dumps(self.parameters),
            "git_commit": self.git_commit,
            "metrics": json.dumps(self.metrics),
            "artifacts": json.dumps(self.artifacts),
            "tags": json.dumps(self.tags),
            "status": self.status,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "notes": self.notes,
        }

    @classmethod
    def from_row(cls, row: dict) -> "ExperimentRun":
        """Create from database row."""
        return cls(
            run_id=row["run_id"],
            name=row["name"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            parameters=json.loads(row["parameters"] or "{}"),
            git_commit=row.get("git_commit"),
            metrics=json.loads(row["metrics"] or "{}"),
            artifacts=json.loads(row["artifacts"] or "{}"),
            tags=json.loads(row["tags"] or "[]"),
            status=row["status"],
            end_time=datetime.fromisoformat(row["end_time"]) if row.get("end_time") else None,
            notes=row.get("notes", ""),
        )


class ExperimentTracker:
    """
    Tracks research experiments for reproducibility.

    Stores experiment metadata in SQLite and artifacts on filesystem.

    Example:
        tracker = ExperimentTracker()
        with tracker.run("whale_strategy_v2", {"lookback": 3}) as run:
            # Run experiment
            result = run_backtest(...)
            tracker.log_metrics(run.run_id, {"sharpe": result.sharpe_ratio})
            tracker.log_artifact(run.run_id, "trades", "trades.csv")

        # Later: search and compare
        runs = tracker.search_runs(name_pattern="whale_*", min_sharpe=1.0)
        comparison = tracker.compare_runs([r.run_id for r in runs])
    """

    def __init__(
        self,
        experiments_dir: str = "research/experiments",
        logger: Optional[logging.Logger] = None,
    ):
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)

        self.db_path = self.experiments_dir / "experiments.db"
        self.artifacts_dir = self.experiments_dir / "artifacts"
        self.artifacts_dir.mkdir(exist_ok=True)

        self.logger = logger or logging.getLogger("experiments.tracker")

        self._init_db()

    def _init_db(self) -> None:
        """Initialize SQLite database schema."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                parameters TEXT,
                git_commit TEXT,
                metrics TEXT,
                artifacts TEXT,
                tags TEXT,
                status TEXT DEFAULT 'running',
                end_time TEXT,
                notes TEXT
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_runs_name ON runs(name)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_runs_timestamp ON runs(timestamp)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status)
        """)

        conn.commit()
        conn.close()

    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()[:12]
        except Exception:
            pass
        return None

    def _generate_run_id(self, name: str) -> str:
        """Generate a unique run ID."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        unique = uuid.uuid4().hex[:6]
        return f"{name}_{timestamp}_{unique}"

    @contextmanager
    def run(
        self,
        name: str,
        parameters: Dict[str, Any],
        tags: Optional[List[str]] = None,
        notes: str = "",
    ) -> Iterator[ExperimentRun]:
        """
        Context manager for tracking an experiment run.

        Args:
            name: Experiment name
            parameters: Experiment parameters
            tags: Optional tags for categorization
            notes: Optional notes about the run

        Yields:
            ExperimentRun instance

        Example:
            with tracker.run("my_experiment", {"lr": 0.01}) as run:
                # Run experiment
                tracker.log_metrics(run.run_id, {"accuracy": 0.95})
        """
        run_id = self._generate_run_id(name)
        git_commit = self._get_git_commit()

        experiment = ExperimentRun(
            run_id=run_id,
            name=name,
            timestamp=datetime.utcnow(),
            parameters=parameters,
            git_commit=git_commit,
            tags=tags or [],
            status="running",
            notes=notes,
        )

        # Save initial run
        self._save_run(experiment)
        self.logger.info(f"Started experiment run: {run_id}")

        try:
            yield experiment
            # Reload to get any logged metrics/artifacts before updating status
            current = self.get_run(experiment.run_id)
            if current:
                current.status = "completed"
                current.end_time = datetime.utcnow()
                self._save_run(current)
            else:
                experiment.status = "completed"
                experiment.end_time = datetime.utcnow()
                self._save_run(experiment)
            self.logger.info(f"Completed experiment run: {run_id}")
        except Exception as e:
            # Reload to preserve logged metrics/artifacts on failure
            current = self.get_run(experiment.run_id)
            if current:
                current.status = "failed"
                current.end_time = datetime.utcnow()
                current.notes = f"{current.notes}\nError: {str(e)}".strip()
                self._save_run(current)
            else:
                experiment.status = "failed"
                experiment.end_time = datetime.utcnow()
                experiment.notes = f"{experiment.notes}\nError: {str(e)}".strip()
                self._save_run(experiment)
            self.logger.error(f"Failed experiment run: {run_id} - {e}")
            raise

    def _save_run(self, run: ExperimentRun) -> None:
        """Save or update an experiment run."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        data = run.to_dict()
        cursor.execute("""
            INSERT OR REPLACE INTO runs
            (run_id, name, timestamp, parameters, git_commit, metrics,
             artifacts, tags, status, end_time, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data["run_id"],
            data["name"],
            data["timestamp"],
            data["parameters"],
            data["git_commit"],
            data["metrics"],
            data["artifacts"],
            data["tags"],
            data["status"],
            data["end_time"],
            data["notes"],
        ))

        conn.commit()
        conn.close()

    def log_metrics(
        self,
        run_id: str,
        metrics: Dict[str, float],
    ) -> None:
        """
        Log metrics for a run.

        Args:
            run_id: Run identifier
            metrics: Metrics to log
        """
        run = self.get_run(run_id)
        if not run:
            raise ValueError(f"Run not found: {run_id}")

        run.metrics.update(metrics)
        self._save_run(run)
        self.logger.debug(f"Logged metrics for {run_id}: {metrics}")

    def log_artifact(
        self,
        run_id: str,
        name: str,
        source_path: str,
        copy: bool = True,
    ) -> str:
        """
        Log an artifact for a run.

        Args:
            run_id: Run identifier
            name: Artifact name
            source_path: Path to artifact file
            copy: If True, copy file to artifacts dir

        Returns:
            Path to stored artifact
        """
        run = self.get_run(run_id)
        if not run:
            raise ValueError(f"Run not found: {run_id}")

        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(f"Artifact not found: {source_path}")

        # Create run-specific artifact directory
        run_artifacts_dir = self.artifacts_dir / run_id
        run_artifacts_dir.mkdir(exist_ok=True)

        # Determine destination
        dest = run_artifacts_dir / f"{name}{source.suffix}"

        if copy:
            shutil.copy2(source, dest)
            artifact_path = str(dest)
        else:
            artifact_path = str(source.absolute())

        run.artifacts[name] = artifact_path
        self._save_run(run)
        self.logger.debug(f"Logged artifact for {run_id}: {name} -> {artifact_path}")

        return artifact_path

    def get_run(self, run_id: str) -> Optional[ExperimentRun]:
        """
        Get a specific run by ID.

        Args:
            run_id: Run identifier

        Returns:
            ExperimentRun or None if not found
        """
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,))
        row = cursor.fetchone()
        conn.close()

        if row:
            return ExperimentRun.from_row(dict(row))
        return None

    def search_runs(
        self,
        name_pattern: Optional[str] = None,
        min_sharpe: Optional[float] = None,
        min_win_rate: Optional[float] = None,
        tags: Optional[List[str]] = None,
        status: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 100,
    ) -> List[ExperimentRun]:
        """
        Search for experiment runs.

        Args:
            name_pattern: SQL LIKE pattern for name
            min_sharpe: Minimum Sharpe ratio
            min_win_rate: Minimum win rate
            tags: Filter by tags (any match)
            status: Filter by status
            start_date: Filter by start date (YYYY-MM-DD)
            end_date: Filter by end date
            limit: Maximum runs to return

        Returns:
            List of matching ExperimentRun objects
        """
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query = "SELECT * FROM runs WHERE 1=1"
        params: List[Any] = []

        if name_pattern:
            query += " AND name LIKE ?"
            params.append(name_pattern.replace("*", "%"))

        if status:
            query += " AND status = ?"
            params.append(status)

        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)

        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        runs = [ExperimentRun.from_row(dict(row)) for row in rows]

        # Filter by metrics (in Python since metrics are JSON)
        if min_sharpe is not None:
            runs = [r for r in runs if r.metrics.get("sharpe", 0) >= min_sharpe]

        if min_win_rate is not None:
            runs = [r for r in runs if r.metrics.get("win_rate", 0) >= min_win_rate]

        # Filter by tags
        if tags:
            runs = [r for r in runs if any(t in r.tags for t in tags)]

        return runs

    def compare_runs(
        self,
        run_ids: List[str],
        metrics: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Compare multiple runs side by side.

        Args:
            run_ids: List of run IDs to compare
            metrics: Specific metrics to compare (default: all)

        Returns:
            DataFrame with runs as rows and metrics as columns
        """
        runs = [self.get_run(rid) for rid in run_ids]
        runs = [r for r in runs if r is not None]

        if not runs:
            return pd.DataFrame()

        data = []
        for run in runs:
            row = {
                "run_id": run.run_id,
                "name": run.name,
                "timestamp": run.timestamp,
                "status": run.status,
                "git_commit": run.git_commit,
            }

            # Add parameters
            for key, value in run.parameters.items():
                row[f"param_{key}"] = value

            # Add metrics
            for key, value in run.metrics.items():
                if metrics is None or key in metrics:
                    row[key] = value

            data.append(row)

        df = pd.DataFrame(data)

        # Reorder columns
        first_cols = ["run_id", "name", "timestamp", "status"]
        param_cols = [c for c in df.columns if c.startswith("param_")]
        metric_cols = [c for c in df.columns if c not in first_cols + param_cols + ["git_commit"]]

        ordered_cols = first_cols + sorted(param_cols) + sorted(metric_cols)
        if "git_commit" in df.columns:
            ordered_cols.append("git_commit")

        return df[[c for c in ordered_cols if c in df.columns]]

    def get_best_run(
        self,
        name_pattern: Optional[str] = None,
        metric: str = "sharpe",
        minimize: bool = False,
    ) -> Optional[ExperimentRun]:
        """
        Get the best run by a specific metric.

        Args:
            name_pattern: Filter by name pattern
            metric: Metric to optimize
            minimize: If True, find minimum instead of maximum

        Returns:
            Best ExperimentRun or None
        """
        runs = self.search_runs(name_pattern=name_pattern, status="completed")

        if not runs:
            return None

        # Filter to runs with the metric
        runs = [r for r in runs if metric in r.metrics]

        if not runs:
            return None

        if minimize:
            return min(runs, key=lambda r: r.metrics[metric])
        return max(runs, key=lambda r: r.metrics[metric])

    def delete_run(self, run_id: str, delete_artifacts: bool = True) -> bool:
        """
        Delete an experiment run.

        Args:
            run_id: Run identifier
            delete_artifacts: If True, also delete artifacts

        Returns:
            True if deleted, False if not found
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
        deleted = cursor.rowcount > 0

        conn.commit()
        conn.close()

        if deleted and delete_artifacts:
            artifacts_path = self.artifacts_dir / run_id
            if artifacts_path.exists():
                shutil.rmtree(artifacts_path)

        return deleted

    def export_run(
        self,
        run_id: str,
        output_path: str,
    ) -> str:
        """
        Export a run and its artifacts to a zip file.

        Args:
            run_id: Run identifier
            output_path: Output zip file path

        Returns:
            Path to exported file
        """
        run = self.get_run(run_id)
        if not run:
            raise ValueError(f"Run not found: {run_id}")

        output = Path(output_path)

        # Create temporary directory for export
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            # Save metadata
            metadata_path = tmp_path / "metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(run.to_dict(), f, indent=2)

            # Copy artifacts
            if run.artifacts:
                artifacts_tmp = tmp_path / "artifacts"
                artifacts_tmp.mkdir()
                for name, path in run.artifacts.items():
                    src = Path(path)
                    if src.exists():
                        shutil.copy2(src, artifacts_tmp / src.name)

            # Create zip
            shutil.make_archive(
                str(output.with_suffix("")),
                "zip",
                tmp_path,
            )

        return str(output)


__all__ = ["ExperimentRun", "ExperimentTracker"]

"""
Notebook archive for research reproducibility.

Manages Jupyter notebooks for experiments, including:
- Archiving executed notebooks
- Parameterized notebook execution
- Search and retrieval
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import logging
import shutil

try:
    import papermill as pm
    PAPERMILL_AVAILABLE = True
except ImportError:
    PAPERMILL_AVAILABLE = False

try:
    import nbformat
    NBFORMAT_AVAILABLE = True
except ImportError:
    NBFORMAT_AVAILABLE = False


@dataclass
class ArchivedNotebook:
    """Represents an archived notebook."""

    notebook_id: str
    name: str
    path: str
    timestamp: datetime
    experiment_id: Optional[str]
    parameters: Dict[str, Any]
    tags: List[str]
    notes: str


class NotebookArchive:
    """
    Archive for research notebooks.

    Provides:
    - Notebook archiving with metadata
    - Parameterized execution via papermill
    - Search and retrieval

    Example:
        archive = NotebookArchive()

        # Archive an existing notebook
        archive.archive_notebook("analysis.ipynb", experiment_id="exp123")

        # Execute with parameters and archive
        archive.execute_and_archive(
            "templates/backtest_report.ipynb",
            parameters={"strategy": "whale_following", "lookback": 3}
        )

        # Search archived notebooks
        results = archive.search_notebooks(query="whale", tags=["backtest"])
    """

    def __init__(
        self,
        archive_dir: str = "research/notebooks",
        templates_dir: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.archive_dir = Path(archive_dir)
        self.archive_dir.mkdir(parents=True, exist_ok=True)

        self.templates_dir = Path(templates_dir) if templates_dir else self.archive_dir / "templates"
        self.templates_dir.mkdir(exist_ok=True)

        self.archived_dir = self.archive_dir / "archived"
        self.archived_dir.mkdir(exist_ok=True)

        self.index_path = self.archive_dir / "notebook_index.json"
        self.logger = logger or logging.getLogger("experiments.notebooks")

        self._index: Dict[str, Dict[str, Any]] = self._load_index()

    def _load_index(self) -> Dict[str, Dict[str, Any]]:
        """Load notebook index from disk."""
        if self.index_path.exists():
            with open(self.index_path, "r") as f:
                return json.load(f)
        return {}

    def _save_index(self) -> None:
        """Save notebook index to disk."""
        with open(self.index_path, "w") as f:
            json.dump(self._index, f, indent=2, default=str)

    def _generate_notebook_id(self, name: str) -> str:
        """Generate unique notebook ID."""
        import uuid
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        unique = uuid.uuid4().hex[:6]
        safe_name = "".join(c if c.isalnum() else "_" for c in name)[:30]
        return f"{safe_name}_{timestamp}_{unique}"

    def archive_notebook(
        self,
        notebook_path: str,
        experiment_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        notes: str = "",
        parameters: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Archive a notebook.

        Args:
            notebook_path: Path to notebook to archive
            experiment_id: Associated experiment ID
            tags: Tags for categorization
            notes: Notes about this notebook
            parameters: Parameters used (if executed)

        Returns:
            Notebook ID in archive
        """
        source = Path(notebook_path)
        if not source.exists():
            raise FileNotFoundError(f"Notebook not found: {notebook_path}")

        name = source.stem
        notebook_id = self._generate_notebook_id(name)

        # Create date-based directory
        date_dir = self.archived_dir / datetime.utcnow().strftime("%Y-%m")
        date_dir.mkdir(exist_ok=True)

        # Copy notebook
        dest = date_dir / f"{notebook_id}.ipynb"
        shutil.copy2(source, dest)

        # Update index
        self._index[notebook_id] = {
            "name": name,
            "path": str(dest),
            "timestamp": datetime.utcnow().isoformat(),
            "experiment_id": experiment_id,
            "parameters": parameters or {},
            "tags": tags or [],
            "notes": notes,
        }
        self._save_index()

        self.logger.info(f"Archived notebook: {notebook_id}")
        return notebook_id

    def execute_and_archive(
        self,
        notebook_path: str,
        parameters: Dict[str, Any],
        experiment_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        notes: str = "",
        kernel_name: Optional[str] = None,
    ) -> str:
        """
        Execute a notebook with parameters and archive the result.

        Requires papermill to be installed.

        Args:
            notebook_path: Path to notebook (or template)
            parameters: Parameters to inject
            experiment_id: Associated experiment ID
            tags: Tags for categorization
            notes: Notes about this execution
            kernel_name: Jupyter kernel to use

        Returns:
            Notebook ID in archive
        """
        if not PAPERMILL_AVAILABLE:
            raise ImportError("papermill required for notebook execution. Install with: pip install papermill")

        source = Path(notebook_path)
        if not source.exists():
            # Check templates directory
            template_path = self.templates_dir / source.name
            if template_path.exists():
                source = template_path
            else:
                raise FileNotFoundError(f"Notebook not found: {notebook_path}")

        name = source.stem
        notebook_id = self._generate_notebook_id(name)

        # Create date-based directory
        date_dir = self.archived_dir / datetime.utcnow().strftime("%Y-%m")
        date_dir.mkdir(exist_ok=True)

        # Output path
        output_path = date_dir / f"{notebook_id}.ipynb"

        # Execute with papermill
        self.logger.info(f"Executing notebook: {source.name} with parameters {parameters}")

        try:
            pm.execute_notebook(
                str(source),
                str(output_path),
                parameters=parameters,
                kernel_name=kernel_name,
            )
        except Exception as e:
            self.logger.error(f"Notebook execution failed: {e}")
            # Still archive the failed notebook for debugging
            if not output_path.exists():
                shutil.copy2(source, output_path)
            notes = f"{notes}\nExecution failed: {str(e)}".strip()

        # Update index
        self._index[notebook_id] = {
            "name": name,
            "path": str(output_path),
            "timestamp": datetime.utcnow().isoformat(),
            "experiment_id": experiment_id,
            "parameters": parameters,
            "tags": tags or [],
            "notes": notes,
        }
        self._save_index()

        self.logger.info(f"Archived executed notebook: {notebook_id}")
        return notebook_id

    def search_notebooks(
        self,
        query: Optional[str] = None,
        tags: Optional[List[str]] = None,
        experiment_id: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Search archived notebooks.

        Args:
            query: Text search in name and notes
            tags: Filter by tags (any match)
            experiment_id: Filter by experiment
            start_date: Filter by date (YYYY-MM-DD)
            end_date: Filter by date
            limit: Maximum results

        Returns:
            List of notebook metadata dicts
        """
        results = []

        for notebook_id, data in self._index.items():
            # Apply filters
            if experiment_id and data.get("experiment_id") != experiment_id:
                continue

            if tags:
                notebook_tags = data.get("tags", [])
                if not any(t in notebook_tags for t in tags):
                    continue

            if start_date:
                if data.get("timestamp", "") < start_date:
                    continue

            if end_date:
                if data.get("timestamp", "") > end_date:
                    continue

            if query:
                query_lower = query.lower()
                name = data.get("name", "").lower()
                notes = data.get("notes", "").lower()
                if query_lower not in name and query_lower not in notes:
                    continue

            results.append({
                "notebook_id": notebook_id,
                **data,
            })

        # Sort by timestamp descending
        results.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

        return results[:limit]

    def get_notebook(self, notebook_id: str) -> Optional[Dict[str, Any]]:
        """
        Get notebook metadata by ID.

        Args:
            notebook_id: Notebook identifier

        Returns:
            Metadata dict or None
        """
        if notebook_id in self._index:
            return {
                "notebook_id": notebook_id,
                **self._index[notebook_id],
            }
        return None

    def get_notebook_path(self, notebook_id: str) -> Optional[Path]:
        """
        Get path to archived notebook file.

        Args:
            notebook_id: Notebook identifier

        Returns:
            Path object or None
        """
        data = self._index.get(notebook_id)
        if data:
            path = Path(data["path"])
            if path.exists():
                return path
        return None

    def create_from_template(
        self,
        template_name: str,
        parameters: Dict[str, Any],
        output_name: Optional[str] = None,
    ) -> str:
        """
        Create a new notebook from a template.

        Args:
            template_name: Name of template (without path)
            parameters: Parameters to inject
            output_name: Output filename (default: template name)

        Returns:
            Path to created notebook
        """
        if not NBFORMAT_AVAILABLE:
            raise ImportError("nbformat required. Install with: pip install nbformat")

        template_path = self.templates_dir / template_name
        if not template_path.exists():
            template_path = self.templates_dir / f"{template_name}.ipynb"
        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_name}")

        # Read template
        with open(template_path, "r") as f:
            notebook = nbformat.read(f, as_version=4)

        # Inject parameters as first cell
        param_cell = nbformat.v4.new_code_cell(
            "# Parameters\n" +
            "\n".join(f"{k} = {repr(v)}" for k, v in parameters.items())
        )
        param_cell.metadata["tags"] = ["parameters"]
        notebook.cells.insert(0, param_cell)

        # Write output
        output_name = output_name or template_path.stem
        output_path = self.archive_dir / f"{output_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.ipynb"

        with open(output_path, "w") as f:
            nbformat.write(notebook, f)

        self.logger.info(f"Created notebook from template: {output_path}")
        return str(output_path)

    def list_templates(self) -> List[str]:
        """
        List available templates.

        Returns:
            List of template names
        """
        templates = []
        for f in self.templates_dir.glob("*.ipynb"):
            templates.append(f.stem)
        return sorted(templates)

    def delete_notebook(self, notebook_id: str) -> bool:
        """
        Delete an archived notebook.

        Args:
            notebook_id: Notebook identifier

        Returns:
            True if deleted, False if not found
        """
        if notebook_id not in self._index:
            return False

        data = self._index[notebook_id]
        path = Path(data["path"])
        if path.exists():
            path.unlink()

        del self._index[notebook_id]
        self._save_index()

        return True


__all__ = ["NotebookArchive", "ArchivedNotebook"]

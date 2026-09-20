"""
ChurnGuard AI — MLOps Experiment Tracking & Model Registry Module.
Integrates with MLflow for tracking parameters, evaluation metrics, model artifacts,
and calibration curves. Includes seamless offline fallback to local JSON audit trails
if an MLflow tracking server is not reachable.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("churn.mlops")

LOCAL_RUNS_DIR = Path("reports/runs")
LOCAL_RUNS_FILE = Path("reports/mlflow_runs.json")


class MLOpsTracker:
    """Manages MLflow experiment runs and artifact logging with graceful offline fallback."""

    def __init__(
        self,
        experiment_name: str = "Customer_Churn_Prediction",
        tracking_uri: Optional[str] = None,
    ):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
        self.mlflow_available = False
        self._init_mlflow()

    def _init_mlflow(self) -> None:
        """Attempt to configure MLflow client; falls back to local logging if unavailable."""
        try:
            import mlflow

            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment_name)
            self.mlflow = mlflow
            self.mlflow_available = True
            logger.info("MLflow tracking initialized: URI=%s, Experiment=%s", self.tracking_uri, self.experiment_name)
        except Exception as e:
            self.mlflow_available = False
            self.mlflow = None
            logger.warning(
                "MLflow tracking unavailable (%s). Falling back to local offline logging at %s",
                e,
                LOCAL_RUNS_FILE,
            )

    def log_run(
        self,
        run_name: str,
        model_name: str,
        params: Dict[str, Any],
        metrics: Dict[str, float],
        artifacts: Optional[List[Path]] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Log hyperparameters, validation metrics, and model artifacts.
        Logs to MLflow if active, and always records a local JSON ledger.
        """
        run_record = {
            "run_name": run_name,
            "model_name": model_name,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "params": params,
            "metrics": metrics,
            "artifacts": [str(p) for p in (artifacts or [])],
            "tags": tags or {},
        }

        # 1. Log to MLflow if connected
        if self.mlflow_available and self.mlflow:
            try:
                with self.mlflow.start_run(run_name=run_name):
                    # Set tags
                    self.mlflow.set_tag("model_type", model_name)
                    if tags:
                        for k, v in tags.items():
                            self.mlflow.set_tag(k, v)

                    # Log hyperparameters
                    for param_name, param_val in params.items():
                        self.mlflow.log_param(param_name, str(param_val))

                    # Log metrics
                    for metric_name, metric_val in metrics.items():
                        if isinstance(metric_val, (int, float)):
                            self.mlflow.log_metric(metric_name, float(metric_val))

                    # Log artifact files
                    if artifacts:
                        for art_path in artifacts:
                            if Path(art_path).exists():
                                self.mlflow.log_artifact(str(art_path))

                logger.info("Successfully logged run '%s' to MLflow server.", run_name)
            except Exception as ex:
                logger.warning("Failed to push to MLflow server (%s). Logging locally.", ex)

        # 2. Always persist locally for offline transparency and reproducibility
        self._record_local_run(run_record)
        return run_record

    def _record_local_run(self, record: Dict[str, Any]) -> None:
        """Appends run record to local JSON ledger."""
        LOCAL_RUNS_DIR.mkdir(parents=True, exist_ok=True)
        runs: List[Dict[str, Any]] = []

        if LOCAL_RUNS_FILE.exists():
            try:
                with open(LOCAL_RUNS_FILE, "r", encoding="utf-8") as f:
                    runs = json.load(f)
            except Exception:
                runs = []

        runs.append(record)
        with open(LOCAL_RUNS_FILE, "w", encoding="utf-8") as f:
            json.dump(runs, f, indent=2)

        # Also write individual run snapshot
        slug = f"{record['model_name']}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        snap_file = LOCAL_RUNS_DIR / f"{slug}.json"
        with open(snap_file, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

    def get_runs_history(self) -> List[Dict[str, Any]]:
        """Retrieve historical runs from local ledger."""
        if LOCAL_RUNS_FILE.exists():
            try:
                with open(LOCAL_RUNS_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return []
        return []


tracker = MLOpsTracker()

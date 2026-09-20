"""
ChurnGuard AI — SHAP Explainable AI Module.
Provides local (individual customer) and global feature attribution explanations.
Converts model weights and interaction scores into human-interpretable percentage
impacts (e.g. "Contract: Month-to-month (+31%)", "Tenure: 65 months (-18%)").
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

logger = logging.getLogger("churn.explain")
FIGURES_DIR = Path("reports/figures")


class ChurnExplainer:
    """Encapsulates SHAP explanation generation across the pipeline."""

    def __init__(self, pipeline: Pipeline, background_data: pd.DataFrame):
        self.pipeline = pipeline
        self.preprocessor = pipeline.named_steps["preprocessor"]
        self.classifier = pipeline.named_steps["classifier"]

        # Transform background samples for explainer
        logger.info("Initializing SHAP explainer with %d background instances...", len(background_data))
        X_bg_trans = self.preprocessor.transform(background_data)
        self.feature_names = list(self.preprocessor.get_feature_names_out())

        # Determine appropriate explainer based on classifier type
        classifier_type = type(self.classifier).__name__
        if "Forest" in classifier_type or "Tree" in classifier_type or "GBM" in classifier_type:
            self.explainer = shap.TreeExplainer(self.classifier, X_bg_trans)
        elif "Logistic" in classifier_type:
            self.explainer = shap.LinearExplainer(self.classifier, X_bg_trans)
        else:
            self.explainer = shap.Explainer(self.classifier, X_bg_trans)

        logger.info("SHAP explainer ready for %s across %d features.", classifier_type, len(self.feature_names))

    def explain_instance(
        self,
        customer_df: pd.DataFrame,
        top_k: int = 4,
    ) -> Dict[str, Any]:
        """
        Compute SHAP values for a single customer instance, returning
        top positive drivers (increasing churn) and top negative drivers (reducing churn).
        """
        X_trans = self.preprocessor.transform(customer_df)
        shap_values = self.explainer(X_trans)

        # Handle binary classification shapes
        raw_values = shap_values.values
        if len(raw_values.shape) == 3:  # (n_samples, n_features, n_classes)
            values = raw_values[0, :, 1]
        elif len(raw_values.shape) == 2:  # (n_samples, n_features)
            values = raw_values[0, :]
        else:
            values = np.array(raw_values).flatten()

        impacts: List[Tuple[str, float]] = []
        for feat_name, val in zip(self.feature_names, values):
            # Clean feature display name
            display_name = feat_name.replace("cat__", "").replace("num__", "").replace("_", " ")
            impacts.append((display_name, float(val)))

        # Sort into positive churn pressures vs negative retention buffers
        positive_drivers = [item for item in impacts if item[1] > 0]
        negative_drivers = [item for item in impacts if item[1] < 0]

        positive_drivers.sort(key=lambda x: x[1], reverse=True)
        negative_drivers.sort(key=lambda x: x[1])  # Most negative first

        # Normalize relative impact percentages for clean executive reporting
        pos_total = sum(v for _, v in positive_drivers) or 1.0
        neg_total = abs(sum(v for _, v in negative_drivers)) or 1.0

        top_pos = [
            {"factor": name.title(), "impact_score": round(val, 4), "relative_pct": f"+{round((val / pos_total) * 100)}%"}
            for name, val in positive_drivers[:top_k]
        ]
        top_neg = [
            {"factor": name.title(), "impact_score": round(val, 4), "relative_pct": f"-{round((abs(val) / neg_total) * 100)}%"}
            for name, val in negative_drivers[:top_k]
        ]

        return {
            "top_churn_drivers": top_pos,
            "top_retention_factors": top_neg,
            "total_features_evaluated": len(self.feature_names),
        }

    def generate_global_summary_plot(self, X_sample: pd.DataFrame, output_path: Path) -> None:
        """Export publication-quality global SHAP summary bar plot."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        X_trans = self.preprocessor.transform(X_sample)
        shap_vals = self.explainer(X_trans)

        raw_vals = shap_vals.values
        if len(raw_vals.shape) == 3:
            vals = raw_vals[:, :, 1]
        else:
            vals = raw_vals

        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            vals,
            features=X_trans,
            feature_names=self.feature_names,
            plot_type="bar",
            max_display=12,
            show=False,
        )
        plt.title("ChurnGuard AI — Global Feature Importance (SHAP)", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        logger.info("Saved global SHAP summary plot to %s", output_path)


def initialize_explainer(
    pipeline_path: Path = Path("models/pipeline.joblib"),
    data_path: Path = Path("data/customer_churn.csv"),
) -> ChurnExplainer:
    """Factory helper to load pipeline and initialize explainer with background sample."""
    from src.train import load_and_preprocess_raw_data

    pipeline = joblib.load(pipeline_path)
    X, _ = load_and_preprocess_raw_data(data_path)
    # Use 200 background instances for low latency inference
    bg_sample = X.sample(n=min(200, len(X)), random_state=42)
    return ChurnExplainer(pipeline, bg_sample)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    explainer = initialize_explainer()
    from src.train import load_and_preprocess_raw_data
    X, _ = load_and_preprocess_raw_data(Path("data/customer_churn.csv"))
    test_instance = X.iloc[[0]]
    res = explainer.explain_instance(test_instance)
    print("Sample Individual SHAP Attribution:")
    print("Top Churn Drivers:", res["top_churn_drivers"])
    print("Top Retention Factors:", res["top_retention_factors"])
    explainer.generate_global_summary_plot(X.sample(300, random_state=42), FIGURES_DIR / "shap_summary.png")

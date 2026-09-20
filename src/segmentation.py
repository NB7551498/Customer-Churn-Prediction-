"""
ChurnGuard AI — Customer Segmentation Module.
Applies K-Means clustering (k=4) on customer tenure, financial spend, and
protection service adoption to identify actionable customer personas:
1. High-Value Loyalists
2. High-Value At-Risk
3. Budget Consumers
4. Unsettled Onboarders
"""

import logging
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("churn.segmentation")

SEGMENT_METADATA = {
    0: {
        "name": "High-Value Loyalists",
        "description": "Long-tenure subscribers with high recurring spend and extensive service adoption. Extremely low churn probability.",
        "risk_profile": "Very Low",
        "icon": "shield-check",
    },
    1: {
        "name": "High-Value At-Risk",
        "description": "Subscribers with high monthly charges (often premium fiber) but low tenure and few support add-ons. High churn vulnerability.",
        "risk_profile": "High",
        "icon": "alert-triangle",
    },
    2: {
        "name": "Budget Consumers",
        "description": "Price-sensitive subscribers on basic phone or DSL tiers with low monthly spend and consistent tenure.",
        "risk_profile": "Low",
        "icon": "wallet",
    },
    3: {
        "name": "Unsettled Onboarders",
        "description": "New subscribers in their first 1–6 months on flexible month-to-month contracts. High early drop-off rate.",
        "risk_profile": "Very High",
        "icon": "user-plus",
    },
}

SEGMENTATION_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "support_protection_index"]
DEFAULT_SEGMENT_MODEL_PATH = Path("models/kmeans_segmentation.joblib")


class CustomerSegmenter:
    """Encapsulates K-Means clustering and persona mapping."""

    def __init__(self, n_clusters: int = 4, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.cluster_order_map: Dict[int, int] = {}
        self.is_fitted: bool = False

    def fit(self, df: pd.DataFrame) -> "CustomerSegmenter":
        """Fit scaler and KMeans clusterer on customer features."""
        logger.info("Fitting K-Means segmentation (k=%d) on %d records...", self.n_clusters, len(df))
        X_seg = df[SEGMENTATION_FEATURES].copy()
        X_scaled = self.scaler.fit_transform(X_seg)
        self.kmeans.fit(X_scaled)
        self.is_fitted = True

        # Map cluster centers deterministically based on tenure and monthly spend
        centers = self.kmeans.cluster_centers_
        tenure_idx = SEGMENTATION_FEATURES.index("tenure")
        charges_idx = SEGMENTATION_FEATURES.index("MonthlyCharges")

        raw_centers = self.scaler.inverse_transform(centers)

        ranked = []
        for cid, center in enumerate(raw_centers):
            t = center[tenure_idx]
            m = center[charges_idx]
            ranked.append((cid, t, m))

        # Rank into canonical personas:
        # Highest tenure + highest charges -> High-Value Loyalists (0)
        # Lowest tenure + highest charges -> High-Value At-Risk (1)
        # Lowest charges -> Budget Consumers (2)
        # Lowest tenure + moderate charges -> Unsettled Onboarders (3)
        sorted_by_tenure = sorted(ranked, key=lambda x: x[1], reverse=True)
        loyalist_cid = sorted_by_tenure[0][0]

        remaining = [r for r in ranked if r[0] != loyalist_cid]
        sorted_by_charges = sorted(remaining, key=lambda x: x[2], reverse=True)
        high_val_at_risk_cid = sorted_by_charges[0][0]

        remaining2 = [r for r in remaining if r[0] != high_val_at_risk_cid]
        budget_cid = sorted(remaining2, key=lambda x: x[2])[0][0]

        remaining3 = [r for r in remaining2 if r[0] != budget_cid]
        onboarder_cid = remaining3[0][0]

        self.cluster_order_map = {
            loyalist_cid: 0,
            high_val_at_risk_cid: 1,
            budget_cid: 2,
            onboarder_cid: 3,
        }
        logger.info("Canonical persona mapping established: %s", self.cluster_order_map)
        return self

    def predict_segment(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Predict segment ID and metadata for a single customer or first row."""
        X_seg = df[SEGMENTATION_FEATURES].copy()
        X_scaled = self.scaler.transform(X_seg)
        raw_cid = int(self.kmeans.predict(X_scaled)[0])
        canonical_cid = self.cluster_order_map.get(raw_cid, raw_cid)
        meta = SEGMENT_METADATA.get(canonical_cid, SEGMENT_METADATA[0])

        return {
            "segment_id": canonical_cid,
            "segment_name": meta["name"],
            "description": meta["description"],
            "risk_profile": meta["risk_profile"],
            "icon": meta["icon"],
        }

    def assign_personas(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assign segment ID and persona name to batch DataFrame."""
        df_out = df.copy()
        X_seg = df[SEGMENTATION_FEATURES].copy()
        X_scaled = self.scaler.transform(X_seg)
        raw_cids = self.kmeans.predict(X_scaled)
        canonical_cids = [self.cluster_order_map.get(int(cid), int(cid)) for cid in raw_cids]
        personas = [SEGMENT_METADATA.get(cid, SEGMENT_METADATA[0])["name"] for cid in canonical_cids]
        df_out["segment_id"] = canonical_cids
        df_out["persona"] = personas
        return df_out


def train_and_save_segmenter(
    data_path: Path = Path("data/customer_churn.csv"),
    output_path: Path = DEFAULT_SEGMENT_MODEL_PATH,
) -> CustomerSegmenter:
    """Train segmentation model on raw dataset and save artifact."""
    from src.train import engineer_features

    df = pd.read_csv(data_path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce").fillna(0.0)
    df = engineer_features(df)

    segmenter = CustomerSegmenter(n_clusters=4, random_state=42)
    segmenter.fit(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(segmenter, output_path)
    logger.info("Saved CustomerSegmenter to %s", output_path)
    return segmenter


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_and_save_segmenter()

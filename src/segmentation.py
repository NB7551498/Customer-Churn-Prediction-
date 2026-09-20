"""
ChurnGuard AI — Customer Segmentation Module.
Applies K-Means clustering (k=5) on customer tenure, financial spend, and
service adoption to identify 5 business personas:
1. Loyal Customers
2. Price Sensitive
3. High Value / High Risk
4. New Customers
5. At-Risk Customers
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
        "name": "Loyal Customers",
        "description": "Long-tenure subscribers with steady recurring spend and extensive service adoption. Extremely low churn probability.",
        "risk_profile": "Very Low",
        "icon": "shield-check",
    },
    1: {
        "name": "Price Sensitive",
        "description": "Subscribers on lower or basic tiers, highly sensitive to price increases and bill fluctuations.",
        "risk_profile": "Medium",
        "icon": "wallet",
    },
    2: {
        "name": "High Value / High Risk",
        "description": "Subscribers with high monthly charges (often premium fiber) on flexible month-to-month contracts. High revenue at risk.",
        "risk_profile": "High",
        "icon": "alert-triangle",
    },
    3: {
        "name": "New Customers",
        "description": "Subscribers in their first 1–6 months of onboarding. Vulnerable to early cancellation if onboarding is friction-heavy.",
        "risk_profile": "High",
        "icon": "user-plus",
    },
    4: {
        "name": "At-Risk Customers",
        "description": "Subscribers exhibiting high churn signals: low support protection, manual payment issues, and escalating dissatisfaction.",
        "risk_profile": "Critical",
        "icon": "flame",
    },
}

SEGMENTATION_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "support_protection_index"]
DEFAULT_SEGMENT_MODEL_PATH = Path("models/kmeans_segmentation.joblib")


class CustomerSegmenter:
    """Encapsulates K-Means clustering and persona mapping."""

    def __init__(self, n_clusters: int = 5, random_state: int = 42):
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

        # Map cluster centers deterministically
        centers = self.kmeans.cluster_centers_
        raw_centers = self.scaler.inverse_transform(centers)

        tenure_idx = SEGMENTATION_FEATURES.index("tenure")
        charges_idx = SEGMENTATION_FEATURES.index("MonthlyCharges")
        support_idx = SEGMENTATION_FEATURES.index("support_protection_index")

        ranked = []
        for cid, center in enumerate(raw_centers):
            t = center[tenure_idx]
            m = center[charges_idx]
            s = center[support_idx]
            ranked.append((cid, t, m, s))

        # 0: Loyal: Max tenure
        sorted_by_tenure = sorted(ranked, key=lambda x: x[1], reverse=True)
        loyal_cid = sorted_by_tenure[0][0]

        # 3: New: Min tenure
        remaining1 = [r for r in ranked if r[0] != loyal_cid]
        sorted_by_min_tenure = sorted(remaining1, key=lambda x: x[1])
        new_cid = sorted_by_min_tenure[0][0]

        # 2: High Value / High Risk: highest monthly charges among remaining
        remaining2 = [r for r in remaining1 if r[0] != new_cid]
        sorted_by_charges = sorted(remaining2, key=lambda x: x[2], reverse=True)
        high_val_cid = sorted_by_charges[0][0]

        # 1: Price Sensitive: lowest monthly charges among remaining
        remaining3 = [r for r in remaining2 if r[0] != high_val_cid]
        sorted_by_low_charges = sorted(remaining3, key=lambda x: x[2])
        price_sens_cid = sorted_by_low_charges[0][0]

        # 4: At-Risk: the remaining cluster
        remaining4 = [r for r in remaining3 if r[0] != price_sens_cid]
        at_risk_cid = remaining4[0][0] if remaining4 else 4

        self.cluster_order_map = {
            loyal_cid: 0,
            price_sens_cid: 1,
            high_val_cid: 2,
            new_cid: 3,
            at_risk_cid: 4,
        }
        logger.info("Canonical 5-persona mapping established: %s", self.cluster_order_map)
        return self

    def predict_segment(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Predict segment ID and metadata for a single customer or first row."""
        X_seg = df[SEGMENTATION_FEATURES].copy()
        X_scaled = self.scaler.transform(X_seg)
        raw_cid = int(self.kmeans.predict(X_scaled)[0])
        canonical_cid = self.cluster_order_map.get(raw_cid, raw_cid % 5)
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
        canonical_cids = [self.cluster_order_map.get(int(cid), int(cid) % 5) for cid in raw_cids]
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

    # If data/customer_churn.csv doesn't exist, check parent or alternative
    if not data_path.exists():
        if Path("../telco_customer_churn.csv").exists():
            data_path = Path("../telco_customer_churn.csv")
        elif Path("telco_customer_churn.csv").exists():
            data_path = Path("telco_customer_churn.csv")

    df = pd.read_csv(data_path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce").fillna(0.0)
    df = engineer_features(df)

    segmenter = CustomerSegmenter(n_clusters=5, random_state=42)
    segmenter.fit(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(segmenter, output_path)
    logger.info("Saved 5-cluster CustomerSegmenter to %s", output_path)
    return segmenter


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_and_save_segmenter()

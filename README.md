# 🛡️ ChurnGuard AI — Enterprise Customer Retention Intelligence Platform

> **A production-grade machine learning system** transforming customer churn prediction into an end-to-end retention intelligence platform. Combines leak-proof scikit-learn pipelines, multi-model benchmarking (5 algorithms), SHAP explainability, unsupervised K-Means customer segmentation ($k=4$ personas), automated retention playbook generation with projected CLV ROI, JWT security with RBAC, a modern Tailwind CSS + Chart.js executive dashboard, and a Streamlit interactive workbench.

[![CI Status](https://img.shields.io/badge/CI%20Tests-39%2F39%20Passing-brightgreen)](tests/)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-2.0.0-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4+-F7931E?logo=scikit-learn)](https://scikit-learn.org/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-red)](https://github.com/slundberg/shap)
[![Security](https://img.shields.io/badge/Security-JWT%20%2B%20Bcrypt%20RBAC-purple)](app/auth.py)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker)](docker-compose.yml)

---

## 🏗️ System Architecture

```text
                                 ┌──────────────────────────┐
                                 │   Customer Data Stream   │
                                 │  (CSV / API / Database)  │
                                 └────────────┬─────────────┘
                                              │
                                              ▼
                                 ┌──────────────────────────┐
                                 │ Domain Feature Engine    │
                                 │ • tenure_monthly_ratio   │
                                 │ • avg_monthly_spend      │
                                 │ • support_protect_idx    │
                                 │ • high_risk_combo        │
                                 └────────────┬─────────────┘
                                              │
                                              ▼
                                 ┌──────────────────────────┐
                                 │ Leak-Proof Preprocessor  │
                                 │ ColumnTransformer        │
                                 │ StandardScaler + OneHot  │
                                 └────────────┬─────────────┘
                                              │
                    ┌─────────────────────────┴─────────────────────────┐
                    ▼                                                   ▼
       ┌─────────────────────────┐                         ┌─────────────────────────┐
       │ Multi-Model Benchmark   │                         │ Unsupervised Personas   │
       │ 5-Fold Stratified CV    │                         │ K-Means (k=4 clusters)  │
       │ (LR, RF, GB, HGB, LGBM) │                         │ Behavioral Profiling    │
       └────────────┬────────────┘                         └────────────┬────────────┘
                    │                                                   │
                    ▼                                                   ▼
       ┌─────────────────────────┐                         ┌─────────────────────────┐
       │ Production Ensemble     │                         │ Persona Classifier      │
       │ Calibrated Risk Scoring │                         │ Loyalists, At-Risk,     │
       │ Cost-Benefit Threshold  │                         │ Budget, Onboarders      │
       └────────────┬────────────┘                         └────────────┬────────────┘
                    │                                                   │
                    └─────────────────────────┬─────────────────────────┘
                                              │
                                              ▼
                                 ┌──────────────────────────┐
                                 │ SHAP Attribution Engine  │
                                 │ Local +/- Feature Drivers│
                                 │ TreeExplainer Attribs    │
                                 └────────────┬─────────────┘
                                              │
                                              ▼
                                 ┌──────────────────────────┐
                                 │ AI Retention Engine      │
                                 │ • Primary Vulnerability  │
                                 │ • Preserved CLV ROI      │
                                 │ • Personalized Outreach  │
                                 └────────────┬─────────────┘
                                              │
                    ┌─────────────────────────┴─────────────────────────┐
                    ▼                                                   ▼
       ┌─────────────────────────┐                         ┌─────────────────────────┐
       │ FastAPI REST Service    │                         │ Dual-UI Presentation    │
       │ • JWT Authentication    │                         │ • Tailwind CSS SPA      │
       │ • Role-Based Access     │                         │ • Streamlit Workbench   │
       │ • Batch CSV Scoring     │                         │ • Real-time Radar & Bar │
       └─────────────────────────┘                         └─────────────────────────┘
```

---

## 🚀 Key Features

1. **Domain Feature Engineering (`src/train.py`):**
   - `tenure_monthly_ratio`: Non-linear interaction between contract tenure and monthly pricing pressure.
   - `avg_monthly_spend`: Normalized historical rate of expenditure ($TotalCharges / (tenure + 1)$).
   - `support_protection_index`: Sum of proactive support security adoptions (0 to 4).
   - `high_risk_combo`: Fast identification of month-to-month contracts coupled with electronic check settlement.

2. **5-Model Stratified Cross-Validation Benchmark:**
   - 5 algorithms systematically benchmarked with 5-fold cross-validation on 5,634 training instances across 6 metrics (Accuracy, Precision, Recall, F1, ROC-AUC, and PR-AUC).

3. **Explainable AI with SHAP (`src/explain.py`):**
   - Individual local feature attributions quantifying exact positive and negative drivers of churn probability.
   - Global TreeExplainer summary charts saved directly to `reports/figures/shap_summary.png`.

4. **Customer Behavioral Segmentation (`src/segmentation.py`):**
   - Unsupervised K-Means clustering ($k=4$) mapping accounts into 4 business personas:
     - 🛡️ **High-Value Loyalists**: High tenure, premium spend, multi-service adoption (lowest risk).
     - ⚠️ **High-Value At-Risk**: High monthly charges, short tenure, missing technical support (highest revenue vulnerability).
     - 💼 **Budget Consumers**: Consistent tenure, price-sensitive baseline tiers.
     - 🚀 **Unsettled Onboarders**: Early subscribers in their first 1–6 months on flexible plans.

5. **AI Retention Playbook Engine (`src/recommendation.py`):**
   - Prescribes concrete mitigation actions, calculates projected preserved CLV revenue ($), and drafts ready-to-send customer success outreach scripts.

6. **Production FastAPI Service (`app/main.py`):**
   - JWT authentication (`pyjwt`, `bcrypt`) with Role-Based Access Control (`admin`, `analyst`, `viewer`).
   - Single prediction (`POST /predict`), bulk batch scoring (`POST /predict/batch`), portfolio analytics (`GET /analytics`), benchmark metrics (`GET /model/metrics`), and latency telemetry (`GET /monitoring`).

7. **Dual Interface Architecture:**
   - **Executive SPA (`app/static/dashboard.html`):** Dark-mode responsive interface built with Tailwind CSS and Chart.js featuring single subscriber scoring, live KPI cards, and drag-and-drop CSV batch evaluation.
   - **Interactive Workbench (`app/app.py`):** 6-tab Streamlit dashboard with sensitivity testing, SHAP waterfall charts, fairness analysis, and API documentation.

---

## 📊 Model Benchmark Comparison

Evaluated on 5,634 stratified samples using 5-Fold Stratified Cross-Validation:

| Model Architecture | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| **Logistic Regression (Balanced)** | 74.05% | 52.52% | **79.20%** | 0.6317 | **0.8480** | **0.6689** |
| **Random Forest (Balanced)** | **78.01%** | 54.51% | 76.52% | **0.6366** | **0.8458** | 0.6611 |
| **Gradient Boosting** | 80.33% | **64.93%** | 52.37% | 0.5795 | 0.8437 | 0.6539 |
| **HistGradientBoosting** | 76.86% | 53.98% | 74.05% | 0.6246 | 0.8383 | 0.6460 |
| **LightGBM** | 76.84% | 54.12% | 73.38% | 0.6231 | 0.8374 | 0.6486 |

> **Selection Rationale:** While Gradient Boosting achieves high nominal accuracy by under-predicting the minority churn class, **Random Forest (Balanced)** and **Logistic Regression (Balanced)** deliver superior recall (~77–79%) and ROC-AUC (~0.846–0.848), successfully catching 4 out of 5 churning subscribers before revenue loss occurs.

---

## 👥 Customer Personas (K-Means Clustering)

| Persona ID | Segment Name | Typical Tenure | Spend Profile | Support Adoption | Retention Strategy |
|---|---|---|---|---|---|
| **0** | **High-Value Loyalists** | 40–72 mo | High ($85–$120/mo) | High (3–4 services) | Exclusive loyalty perks, VIP account management |
| **1** | **High-Value At-Risk** | 1–12 mo | High ($80–$115/mo) | Low (0–1 services) | Priority 24/7 tech support bundle, 1-yr term discount |
| **2** | **Budget Consumers** | 20–60 mo | Low ($20–$45/mo) | Low–Moderate | Price lock guarantee, automated billing discount |
| **3** | **Unsettled Onboarders** | 1–6 mo | Moderate ($50–$75/mo) | Low (0 services) | Proactive customer success check-in, onboarding guidance |

---

## 🔐 Security & RBAC Specification

ChurnGuard AI implements standard Bearer token JWT authentication signed with HS256:

| Role | Permissions | Available Endpoints |
|---|---|---|
| **Admin** | Full access | `/predict`, `/predict/batch`, `/analytics`, `/model/metrics`, `/monitoring`, `/dashboard` |
| **Analyst** | Operational & Scoring | `/predict`, `/predict/batch`, `/analytics`, `/model/metrics`, `/dashboard` |
| **Viewer** | Read-only analytics | `/analytics`, `/model/metrics`, `/dashboard` |

### Default Test Credentials

| Username | Password | Role |
|---|---|---|
| `admin@churnguard.ai` | `AdminPass123!` | Admin |
| `analyst@churnguard.ai` | `AnalystPass123!` | Analyst |
| `viewer@churnguard.ai` | `ViewerPass123!` | Viewer |

*(Note: Guest mode is enabled by default for frictionless local evaluation when no Authorization header is provided).*

---

## 🔌 API Reference

| Method | Endpoint | Access Tier | Description |
|---|---|---|---|
| `POST` | `/auth/login` | Public | Authenticate with credentials and receive signed JWT |
| `GET` | `/dashboard` | Public / Guest | Executive Tailwind CSS + Chart.js web dashboard |
| `POST` | `/predict` | Guest / Any Role | Real-time churn scoring, persona mapping, SHAP drivers & playbook |
| `POST` | `/predict/batch` | Analyst / Admin | Upload CSV of subscribers; returns prioritized scored CSV stream |
| `POST` | `/explain` | Guest / Any Role | Direct SHAP feature attributions and impact percentages |
| `GET` | `/analytics` | Viewer / Analyst / Admin | Executive portfolio overview (MRR at risk, segment distributions) |
| `GET` | `/model/metrics` | Viewer / Analyst / Admin | 5-model stratified cross-validation comparison table |
| `GET` | `/monitoring` | Viewer / Analyst / Admin | Live latency, request throughput, and data drift indicators |
| `GET` | `/health` | Public | Liveness probe verifying ML pipeline, segmenter, and SHAP explainer |
| `GET` | `/model-info` | Public | Model version, algorithm specifications, and metadata |

### Sample Request (`POST /predict`):

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female",
    "SeniorCitizen": "0",
    "Partner": "No",
    "Dependents": "No",
    "tenure": 2,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 89.50,
    "TotalCharges": 179.00
  }'
```

### Sample Response:

```json
{
  "customer_id": "C-SUBSCRIBER",
  "churn_probability": 0.8642,
  "is_churn": true,
  "risk_tier": "High",
  "recommendation": "High volatility due to flexible month-to-month commitment: Present 15% discount incentive on a 1-year loyalty agreement.",
  "decision_threshold": 0.10,
  "segmentation": {
    "segment_id": 1,
    "segment_name": "High-Value At-Risk",
    "description": "Subscribers with high monthly charges but low tenure and few support add-ons.",
    "risk_profile": "High",
    "icon": "alert-triangle"
  },
  "top_churn_drivers": [
    {"factor": "Contract Commitment", "impact_score": 0.35, "relative_pct": "+35%"},
    {"factor": "Monthly Charges", "impact_score": 0.20, "relative_pct": "+20%"}
  ],
  "top_retention_factors": [
    {"factor": "Tenure Duration", "impact_score": -0.25, "relative_pct": "-25%"}
  ],
  "retention_playbook": {
    "primary_issue": "High volatility due to flexible month-to-month commitment.",
    "segment": "High-Value At-Risk",
    "recommended_actions": [
      "Present 15% discount incentive on a 1-year loyalty agreement.",
      "Offer a one-time $10 bill credit to enroll in automated bank transfer.",
      "Bundle complimentary priority 24/7 TechSupport for 6 months."
    ],
    "projected_risk_reduction": "-35% churn probability",
    "estimated_arr_at_risk": "$1,074.00",
    "projected_clv_preserved": "$751.80 preserved recurring ARR",
    "llm_outreach_draft": {
      "channel": "Email / Relationship Manager",
      "subject": "Special appreciation offer for your Fiber optic account",
      "body": "Hi there,\n\nWe noticed you've been with us for 2 months..."
    }
  }
}
```

---

## 🧪 Testing & Verification

The suite includes 39 unit and integration tests covering preprocessing, model serialization, SHAP explainers, K-Means clustering, recommendation generation, API routes, JWT security, and RBAC:

```bash
# Run complete test suite
pytest -v

# Run linter checks
ruff check .
```

---

## ⚡ Quickstart & Local Execution

### 1. Environment Setup

```bash
git clone https://github.com/NB7551498/Customer-Churn-Prediction-.git
cd Customer-Churn-Prediction-

python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Model Training & Pipeline Serialization

```bash
# Execute multi-model benchmark, feature engineering, and pipeline export
python -m src.train

# Fit and export K-Means customer segmentation clusterer
python -m src.segmentation

# Generate global SHAP explainability visualizations
python -m src.explain
```

### 3. Launch the Platform

```bash
# Terminal 1: Launch FastAPI Backend (Port 8000)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Launch Streamlit Workbench (Port 8501)
streamlit run app/app.py
```

- **Executive Tailwind Dashboard:** Open [http://localhost:8000/dashboard](http://localhost:8000/dashboard)
- **Interactive Swagger Docs:** Open [http://localhost:8000/docs](http://localhost:8000/docs)
- **Streamlit Workbench:** Open [http://localhost:8501](http://localhost:8501)

### 4. Run with Docker Compose

```bash
docker-compose up --build
```

---

## 📁 Repository Structure

```text
customer-churn-prediction/
├── app/
│   ├── auth.py                  # JWT authentication, bcrypt & RBAC guards
│   ├── main.py                  # FastAPI server with 10 enterprise endpoints
│   ├── schemas.py               # Pydantic v2 validation models
│   ├── explainer.py             # Inference-time SHAP wrapper
│   ├── app.py                   # 6-tab Streamlit workbench
│   └── static/
│       └── dashboard.html       # Responsive Tailwind CSS + Chart.js executive SPA
├── data/
│   └── customer_churn.csv       # Telco customer churn dataset
├── models/
│   ├── pipeline.joblib          # Serialized production pipeline
│   ├── kmeans_segmentation.joblib # Serialized K-Means segmenter
│   └── optimal_threshold.json   # Business cost-benefit threshold configuration
├── reports/
│   ├── model_benchmark_comparison.csv # 5-model cross-validation scorecard
│   └── figures/
│       ├── shap_summary.png     # Global SHAP feature attributions
│       ├── roc_curves.png       # ROC curve evaluation
│       └── confusion_matrix.png # Confusion matrix at optimal threshold
├── src/
│   ├── train.py                 # Multi-model benchmarking & feature engineering
│   ├── segmentation.py          # K-Means customer persona clustering
│   ├── explain.py               # SHAP explainability pipeline
│   ├── recommendation.py        # Automated AI retention playbook engine
│   └── fairness.py              # Demographic fairness & bias mitigation
├── tests/
│   ├── test_api.py              # FastAPI endpoint & batch tests
│   ├── test_auth.py             # JWT, password hashing & RBAC tests
│   ├── test_explainer.py        # SHAP TreeExplainer unit tests
│   ├── test_pipeline.py         # Preprocessing & pipeline tests
│   ├── test_recommendation.py   # AI retention playbook tests
│   └── test_segmentation.py     # K-Means persona clustering tests
├── Dockerfile                   # Multi-stage container build
├── docker-compose.yml           # Multi-service composition
├── requirements.txt             # Locked production dependencies
├── pytest.ini                   # Test configuration
└── README.md                    # Platform documentation
```

---

## 📜 License

Distributed under the MIT License. Developed for enterprise customer retention operations and portfolio demonstration.

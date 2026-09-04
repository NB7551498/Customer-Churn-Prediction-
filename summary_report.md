# Customer Churn Prediction — Executive Summary Report

**Task:** Supervised Machine Learning Classification (InternSpark ML Assignment)  
**Deliverable:** Short Technical Summary Report  

---

## 1. Problem Statement
Customer attrition ("churn") is a major driver of revenue loss in subscription businesses. Because acquiring a new subscriber is significantly more expensive than retaining an existing one, the goal of this project is to build a supervised binary classification model that accurately predicts whether an active customer will leave:
$$\text{Input: Customer Features } (X) \longrightarrow \text{Output: Churn } (y \in \{0, 1\})$$
Where:
- $0 = \text{Retained}$
- $1 = \text{Churned}$

---

## 2. Dataset Selection
- **Dataset:** IBM Telco Customer Churn (`telco_customer_churn.csv`).
- **Dimensions:** 7,043 customer records, 20 features + 1 target (`Churn`).
- **Class Distribution:** 5,174 Retained (73.46%) vs. 1,869 Churned (26.54%).
- **Data Preprocessing:**
  1. Handled 11 blank space strings in `TotalCharges` for new accounts (`tenure == 0`) by imputing `0.0`.
  2. Dropped arbitrary identifier `customerID`.
  3. Formatted target `Churn` as binary numeric (`1` for Yes, `0` for No).
  4. Encapsulated feature scaling (`StandardScaler` for numeric) and encoding (`OneHotEncoder(drop='first')` for categorical) inside a leak-proof `ColumnTransformer`.
  5. Applied an **80/20 Stratified Train/Test Split** (`stratify=y`, `random_state=42`) yielding 5,634 training and 1,409 testing samples.

---

## 3. Why These Two Models Were Chosen

1. **Logistic Regression (Baseline Model):**
   - **Why:** Serves as the industry-standard linear classification benchmark. It is computationally fast, resistant to overfitting on low-dimensional data, and provides direct probability estimates with transparent, interpretable log-odds coefficients.
   - **Configuration:** `class_weight='balanced'`, `max_iter=1000`, `random_state=42`.

2. **Random Forest Classifier (Ensemble Model):**
   - **Why:** Non-linear ensemble model based on bagging (bootstrap aggregation). It automatically captures non-linear interactions and threshold effects (e.g., high monthly charges combined with specific contract terms) without requiring manual interaction engineering.
   - **Configuration:** `n_estimators=150`, `max_depth=8`, `class_weight='balanced'`, `random_state=42`.

---

## 4. Cross-Validation & Final Evaluation Metrics

### A. 5-Fold Stratified Cross-Validation (Training Partition, $N = 5,634$)
Validation was conducted strictly on the training set using `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`:

| Model | Accuracy (Mean ± Std) | Precision (Mean ± Std) | Recall (Mean ± Std) | F1-Score (Mean ± Std) | ROC-AUC (Mean ± Std) |
|---|---|---|---|---|---|
| **Logistic Regression** | 74.97% ± 1.46% | 51.85% ± 1.81% | **80.20% ± 3.79%** | 62.96% ± 2.26% | 0.8459 ± 0.0124 |
| **Random Forest** | **76.64% ± 1.04%** | **54.27% ± 1.51%** | 76.52% ± 2.18% | **63.49% ± 1.46%** | **0.8463 ± 0.0106** |

---

### B. Final Benchmark on Untouched Test Set ($N = 1,409$)
Evaluated once on the isolated 20% test partition (374 true churners, 1,035 retained customers):

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | 73.81% | 50.43% | 78.34% | 61.36% | 0.8417 |
| **Random Forest (Baseline)** | **76.22%** | **53.54%** | **78.88%** | **63.78%** | **0.8432** |
| **Tuned Random Forest (`GridSearchCV`)** | 75.87% | 53.09% | 78.07% | 63.20% | 0.8415 |

---

## 5. Winner Declaration & Data-Backed Justification

### **Declared Winner:** **Random Forest Classifier**

### Data-Backed Justification:
1. **Higher Discriminative Power (ROC-AUC: 0.8432 vs. 0.8417):** The Random Forest model demonstrates higher separation between churners and non-churners across all classification thresholds.
2. **Superior Precision-Recall Balance (F1-Score: 63.78% vs. 61.36%):** Random Forest achieves an outstanding **78.88% Recall** (catching 295 out of 374 actual churners) while maintaining **53.54% Precision** (compared to 50.43% for Logistic Regression).
3. **Fewer False Positives:** By reducing false alarms by over 30 customers on the test sample alone, Random Forest avoids wasting retention budget and customer outreach on customers who were already loyal.
4. **Captures Non-Linear Feature Interactions:** Telecommunications churn is driven by coupled features (e.g., Fiber Optic subscribers with high monthly bills who lack technical support), which tree splits capture far better than additive linear hyperplanes.

---

## 6. Key Churn Drivers (Feature Importance)
1. **Contract Type (Month-to-month):** Customers on flexible monthly contracts churn at **42.7%**, compared to only **2.8%** on 2-year contracts.
2. **Tenure:** Attrition is heavily concentrated in the first 12 months (median churn tenure = 10 months).
3. **Monthly Charges & Fiber Optic:** High monthly bills without bundled services significantly elevate churn propensity.
4. **Payment Method:** Electronic check users exhibit a **45.3%** churn rate vs. <18% for automatic bank or credit card billing.
5. **Technical Support:** Subscribers without `TechSupport` and `OnlineSecurity` churn at more than triple the rate of subscribers with these protections.

---

## 7. Submission Checklist
- [x] Cleaned dataset without data leakage.
- [x] Stratified 80/20 train/test split.
- [x] 5-Fold Stratified Cross-Validation on training data.
- [x] Two algorithms compared (Logistic Regression vs. Random Forest).
- [x] All 5 required metrics reported in clean comparison tables (Accuracy, Precision, Recall, F1, ROC-AUC).
- [x] Fully executed Jupyter Notebook with plots (`churn_classification_assignment.ipynb`).
- [x] Short, direct summary report with declared winner (`summary_report.md`).

# Presentation Slides: Customer Churn Prediction ML Project

**Project Title:** Customer Churn Prediction Using Machine Learning Classification Algorithms  
**Target Audience:** Evaluators, Data Science Instructors, Business Stakeholders  
**Format:** 12-Slide Executive Walkthrough  

---

### Slide 1: Title Slide
- **Title:** Customer Churn Prediction Using Machine Learning Classification Algorithms
- **Subtitle:** An End-to-End Supervised ML Pipeline for Proactive Subscriber Retention
- **Presenter:** Machine Learning Engineering Intern / Candidate
- **Core Technologies:** Python, Scikit-Learn, Pandas, Matplotlib/Seaborn, Joblib, Streamlit
- **Visual:** Project logo / Telecommunications churn infographic.

---

### Slide 2: Executive Summary & Problem Statement
- **The Business Challenge:** Customer Acquisition Cost (CAC) is 5x to 7x higher than Customer Retention Cost (CRC).
- **Goal:** Predict whether an individual subscriber is likely to leave (`Churn = Yes/No`) before they terminate service.
- **Impact:** Empower retention teams to deliver targeted, automated incentives (discounts, contract lock-ins, support bundles) to safeguard high-margin recurring revenue.
- **Visual:** Value realization diagram (Raw Data -> ML Classifier -> Early Risk Alert -> Retention Action).

---

### Slide 3: Project Objectives & Requirements
- Build a supervised classification system adhering to strict ML engineering principles:
  1. Automated, leak-proof preprocessing using `ColumnTransformer`.
  2. Multi-algorithm benchmarking: Logistic Regression vs. Random Forest Ensemble.
  3. 5-Fold Stratified Cross-Validation on training data.
  4. Balanced evaluation across Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
  5. Hyperparameter tuning using `GridSearchCV`.
  6. Live deployment via interactive Streamlit Web App.
- **Visual:** Flowchart of 5 development phases (Foundation -> Modeling -> Improvement -> Product -> Submission).

---

### Slide 4: Dataset Overview & Quality Audit
- **Dataset:** IBM Telco Customer Churn (7,043 rows, 21 attributes).
- **Feature Categories:** Demographics (Gender, Senior Citizen), Account Profile (Contract, Tenure, Payment Method), Subscriptions (Internet, Tech Support, Security), Financials (Monthly Charges, Total Charges).
- **Target Distribution:** 73.5% Retained ($N = 5,174$) vs. 26.5% Churned ($N = 1,869$).
- **Data Quality Remediation:** Detected 11 whitespace strings in `TotalCharges` for subscribers with `tenure == 0`; converted to numeric float and imputed $0.0$. Dropped non-predictive `customerID`.
- **Recommended Chart:** Screenshot of `reports/figures/01_churn_distribution.png`.

---

### Slide 5: Exploratory Data Analysis (EDA) Insights
- **Key Insight 1 (Contract Type):** Month-to-month contracts have a **42.7% churn rate** vs. **2.8%** for two-year contracts.
- **Key Insight 2 (Tenure Curve):** High attrition concentrated in the first 12 months (median churn tenure = 10 months).
- **Key Insight 3 (Fiber Optic Disconnect):** Fiber subscribers exhibit a **41.9% churn rate**, driven by high fees and lack of technical assistance.
- **Key Insight 4 (Payment Friction):** Electronic check users churn at **45.3%** vs. ~16% for automated payment methods.
- **Recommended Chart:** Screenshot of `reports/figures/03_categorical_churn_rates.png`.

---

### Slide 6: Preprocessing & Anti-Leakage Architecture
- **Stratified Train/Test Split:** 80% Training ($N = 5,634$) and 20% Untouched Testing ($N = 1,409$). Stratification preserves the 26.54% class balance.
- **Scikit-Learn ColumnTransformer:**
  - `StandardScaler()` applied to continuous variables (`tenure`, `MonthlyCharges`, `TotalCharges`).
  - `OneHotEncoder(drop='first', handle_unknown='ignore')` applied to categorical features.
- **Zero Data Leakage:** Preprocessing pipeline is fit strictly on training folds and applied downstream.
- **Recommended Diagram:** Scikit-Learn Pipeline architecture diagram showing ColumnTransformer + Estimator.

---

### Slide 7: Model Selection & Cross-Validation Methodology
- **Evaluated Algorithms:**
  1. *Logistic Regression:* Linear probabilistic baseline with log-odds transparency (`class_weight='balanced'`).
  2. *Random Forest:* Non-linear ensemble aggregating decorrelated decision trees (`class_weight='balanced'`).
- **Validation Scheme:** 5-Fold Stratified Cross-Validation on the training partition ($K = 5$).
- **Primary Metric:** ROC-AUC and Recall (missing a churner is far costlier than sending an offer to a retained subscriber).
- **Recommended Table:** 5-Fold CV metrics comparison table.

---

### Slide 8: Model Comparison & Benchmarking
- **Test Set Performance ($N = 1,409$):**
  - **Random Forest:** Accuracy: **76.22%** | Precision: **53.54%** | Recall: **78.88%** | F1: **63.78%** | ROC-AUC: **0.8432**
  - **Logistic Regression:** Accuracy: **73.81%** | Precision: **50.43%** | Recall: **78.34%** | F1: **61.36%** | ROC-AUC: **0.8417**
- **Decision Rationale:** Random Forest outperforms Logistic Regression across every single evaluation metric, capturing 295 out of 374 churners with significantly fewer false alarms.
- **Recommended Chart:** Screenshot of `reports/figures/06_roc_curves.png` and `reports/figures/confusion_matrices.png`.

---

### Slide 9: Hyperparameter Optimization
- **GridSearchCV Optimization:** 5-fold stratified CV across 24 configurations (120 model fits).
- **Tuned Hyperparameters:** Tree depth (`max_depth = 8`), Number of estimators (`n_estimators = 200`), Leaf sample limits (`min_samples_split = 5`, `min_samples_leaf = 1`).
- **Result:** Stabilized decision boundaries, regularized variance, achieving 0.8471 mean CV ROC-AUC.
- **Recommended Visual:** Parameter grid heatmap or cross-validation score trajectory.

---

### Slide 10: Model Explainability & Key Churn Drivers
- **Top Predictive Features:**
  1. *Contract (Month-to-month):* Most dominant indicator of attrition.
  2. *Tenure:* Longitudinal loyalty exponentially suppresses churn risk.
  3. *Monthly Charges:* High bills increase price sensitivity.
  4. *Internet Service (Fiber Optic):* Creates dissatisfaction without tech support.
  5. *Payment Method (Electronic Check):* Correlates with lack of commitment and high friction.
- **Recommended Chart:** Screenshot of `reports/figures/feature_importance.png` and `reports/figures/logistic_coefficients.png`.

---

### Slide 11: Production Deployment & Streamlit Web App
- **End-to-End Pipeline Artifact:** Serialized `best_model.pkl` encapsulating preprocessor + tuned forest.
- **Streamlit Interactive Application:**
  - Real-time customer profile input sliders and selectors.
  - Automated Churn Probability score (%) and Risk Tiers (Low, Medium, High).
  - Prescriptive, automated retention intervention playbooks.
- **Standalone Prediction CLI:** `src/predict.py` for automated batch/API inference.
- **Recommended Visual:** Screenshot of Streamlit dashboard interface (`app/app.py`).

---

### Slide 12: Business Impact, Conclusion & Future Scope
- **Business Playbook:**
  1. Convert month-to-month subscribers with tailored 1-year contract discounts.
  2. Offer billing incentives to migrate electronic check payers to autopay.
  3. Bundle free Tech Support with high-speed Fiber Optic packages.
- **Future Enhancements:** Survival analysis (Kaplan-Meier), SHAP explainability, and REST API deployment.
- **Final Verdict:** All project requirements satisfied with rigorous ML methodology and reproducible code.

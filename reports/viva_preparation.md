# Viva Preparation & Technical Interview Guide: Customer Churn Prediction

This document provides 30 technical viva / interview questions and comprehensive, model-grade answers covering foundational to advanced machine learning concepts evaluated in this project.

---

### Q1: What is classification?
**Answer:**  
Classification is a supervised machine learning task where the model learns a mapping function from input features $X$ to discrete categorical output labels $y$. Unlike regression, where the target is continuous, classification assigns each instance into one of two (binary) or more (multiclass) predefined categories. In this project, it is binary classification: predicting whether a subscriber will churn (`1`) or remain (`0`).

---

### Q2: What is supervised learning?
**Answer:**  
Supervised learning is a machine learning paradigm in which algorithms are trained using labeled datasets containing both input features $X$ and ground-truth target labels $y$. The algorithm adjusts its internal parameters during training to minimize the error between its predictions $\hat{y}$ and actual labels $y$.

---

### Q3: Why did you choose churn prediction?
**Answer:**  
Customer churn prediction is a high-value, classic business problem where machine learning provides immediate financial ROI. In subscription industries (telecom, SaaS, fintech), Customer Acquisition Cost (CAC) is 5x to 7x higher than retention cost. A 5% increase in retention can boost enterprise profits by 25% to 95%. Churn prediction also presents realistic data science challenges, including mixed data types, class imbalance, and non-linear feature interactions.

---

### Q4: What is Logistic Regression and how does it work mathematically?
**Answer:**  
Logistic Regression is a linear classification model that estimates the probability that an observation belongs to a particular class. It applies the linear combination of inputs $z = \beta_0 + \sum \beta_i x_i$ to the standard logistic (sigmoid) function:
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$
This squashes any real-valued number into a valid probability between $0$ and $1$. It uses maximum likelihood estimation (log-loss / binary cross-entropy) to solve for optimal weights $\beta$.

---

### Q5: Why use Logistic Regression as a baseline?
**Answer:**  
Logistic Regression is fast to train, computationally lightweight, and mathematically interpretable through its coefficients (log-odds). It provides a strong, transparent linear benchmark against which more complex non-linear ensemble models (like Random Forest or Gradient Boosting) must justify their added complexity.

---

### Q6: What is Random Forest and how does it work?
**Answer:**  
Random Forest is an ensemble learning method based on **Bagging (Bootstrap Aggregation)**. It constructs a collection of decorrelated decision trees during training. Each tree is trained on a random bootstrap sample of the training data (sampling with replacement), and at each split, only a random subset of features ($\approx \sqrt{p}$) is considered. For classification, the forest aggregates predictions via majority voting or average class probability across all trees, which drastically reduces variance without increasing bias.

---

### Q7: Why compare two distinct models instead of training just one?
**Answer:**  
The **"No Free Lunch" theorem** in machine learning asserts that no single algorithm is universally superior across all problem domains. Comparing Logistic Regression (a linear, parametric model) with Random Forest (a non-linear, non-parametric ensemble) allows us to evaluate whether the underlying data relationships are predominantly linear or contain high-order non-linear feature interactions, verifying whether ensemble complexity yields genuine performance gains.

---

### Q8: What is train/test split?
**Answer:**  
Train/test split is the practice of partitioning available historical data into mutually exclusive subsets: a training set used to fit the model parameters and a testing set that remains strictly isolated. Evaluating on the unseen test set estimates how well the trained model generalizes to new, real-world data.

---

### Q9: Why use an 80/20 split ratio?
**Answer:**  
An 80/20 split provides an optimal empirical balance between sample size for model fitting and statistical power for validation. With 7,043 samples, 80% (5,634 rows) gives sufficient volume for robust 5-fold cross-validation, while 20% (1,409 rows) provides a large enough test sample to calculate tight confidence bounds on evaluation metrics without high variance.

---

### Q10: What is stratification and why is it essential for classification?
**Answer:**  
Stratification ensures that the relative frequency of each target class in the original dataset is preserved identically across every split (both train/test splits and cross-validation folds). If random splitting were used on an imbalanced dataset (e.g., 26.5% churn), a test fold might randomly receive an unrepresentative proportion of churners (e.g., 15% or 35%), biasing evaluation metrics.

---

### Q11: What is cross-validation?
**Answer:**  
Cross-validation is a statistical resampling procedure used to evaluate machine learning models on limited data. In $K$-fold cross-validation, the training set is partitioned into $K$ equal subsets (folds). The model is trained on $K-1$ folds and validated on the remaining fold, repeating this process $K$ times so every fold serves as the validation set once. The average metric and standard deviation across all $K$ iterations provide a realistic estimate of model generalization performance.

---

### Q12: Why must cross-validation be performed only on training data, keeping the test set untouched?
**Answer:**  
To prevent **data snooping / data leakage**. If the test set were included during model selection, feature engineering, or hyperparameter tuning, the final architecture would be optimized to fit the idiosyncrasies of that test set. Keeping the test set completely untouched until the final model is locked guarantees an unbiased measurement of real-world generalization.

---

### Q13: What is overfitting?
**Answer:**  
Overfitting occurs when a model learns not only the underlying patterns but also the random noise and idiosyncrasies present in the training data. An overfitted model demonstrates near-perfect performance on training data but fails to generalize, exhibiting high error on unseen test data (high variance). In decision trees, this occurs when trees grow to excessive depth without regularization.

---

### Q14: What is underfitting?
**Answer:**  
Underfitting occurs when a model is too simplistic to capture the true underlying data patterns (high bias). An underfitted model exhibits poor performance on both training and testing datasets. For instance, using a strictly linear model when the feature boundaries are strongly non-linear can cause underfitting.

---

### Q15: What is Precision and when is it critical?
**Answer:**  
$$\text{Precision} = \frac{TP}{TP + FP}$$
Precision measures the proportion of positive predictions that are truly positive (i.e., when the model predicts a customer will churn, how often is it right?). Precision is critical when False Positives incur high financial or brand costs—for example, if every retention offer costs \$200 in free hardware, high precision prevents squandering budget on customers who were never going to leave.

---

### Q16: What is Recall and why is it particularly vital for churn prediction?
**Answer:**  
$$\text{Recall} = \frac{TP}{TP + FN}$$
Recall (sensitivity) measures the proportion of actual positive cases that the model successfully identifies. In customer churn, **Recall is typically the most critical metric** because the business cost of a False Negative (missing a customer who terminates their service, forfeiting hundreds of dollars in annual recurring revenue) is substantially higher than the cost of a False Positive (sending a polite discount email to an already loyal customer).

---

### Q17: What is the F1-Score?
**Answer:**  
$$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
The F1-score is the harmonic mean of Precision and Recall. Unlike the arithmetic mean, the harmonic mean penalizes extreme imbalances between the two metrics, providing a single balanced metric that reflects performance when both false alarms and missed detections carry consequence.

---

### Q18: What is ROC-AUC?
**Answer:**  
ROC-AUC measures the **Area Under the Receiver Operating Characteristic Curve**. The ROC curve plots the True Positive Rate (Recall) against the False Positive Rate ($1 - \text{Specificity}$) across all classification probability thresholds (from 0.0 to 1.0). An AUC of 1.0 indicates perfect discrimination, 0.5 indicates random guessing, and our model's AUC of ~**0.843** indicates an 84.3% probability that a randomly chosen churner will be assigned a higher churn risk score than a randomly chosen retained customer.

---

### Q19: What is a Confusion Matrix?
**Answer:**  
A confusion matrix is a $2 \times 2$ contingency table summarizing the counts of:
- **True Negatives (TN):** Actual non-churners predicted as non-churners.
- **False Positives (FP) / Type I Error:** Actual non-churners predicted as churners.
- **False Negatives (FN) / Type II Error:** Actual churners predicted as non-churners.
- **True Positives (TP):** Actual churners correctly identified.

---

### Q20: Why can Accuracy be misleading in churn prediction?
**Answer:**  
Because of **class imbalance**. In our dataset, 73.5% of customers are retained and 26.5% churn. A naive "dummy" classifier that predicts `No Churn` for 100% of customers achieves an apparent **73.5% Accuracy**, yet its Recall is **0.0%** and it fails to identify a single at-risk customer. Therefore, relying solely on accuracy gives a false illusion of model efficacy.

---

### Q21: How did you address class imbalance?
**Answer:**  
We implemented **cost-sensitive learning** using `class_weight='balanced'`. This dynamically scales the penalty weights inversely proportional to class frequencies:
$$w_j = \frac{N}{2 \times N_j}$$
For the minority churn class ($N_1 = 1,869$), errors are penalized roughly 2.77 times more heavily than errors on the majority class ($N_0 = 5,174$). This shifted the decision threshold, driving our test Recall up to **78.88%** without needing synthetic data oversampling (SMOTE) which can introduce artificial noise into categorical distributions.

---

### Q22: What is data leakage and how did you prevent it?
**Answer:**  
Data leakage occurs when information from outside the training dataset (such as test set statistics or future information) is inadvertently used to train the model. For example, fitting a `StandardScaler` on the entire dataset before splitting leaks the mean and standard deviation of the test set into training.  
We prevented data leakage by:
1. Splitting data into train/test partitions **before** any fitting.
2. Encapsulating all scaling and one-hot encoding inside a Scikit-Learn `ColumnTransformer` within an end-to-end `Pipeline`.
3. Fitting the pipeline strictly on training folds during cross-validation.

---

### Q23: Why use StandardScaler on numerical features?
**Answer:**  
`StandardScaler` standardizes features by subtracting the mean and dividing by the standard deviation:
$$z = \frac{x - \mu}{\sigma}$$
This is crucial for distance- and gradient-based algorithms like Logistic Regression, Support Vector Machines, and Neural Networks, ensuring features with large numerical magnitudes (e.g., `TotalCharges` up to \$8,600) do not dominate features with small ranges (e.g., `tenure` 0–72). While decision trees are invariant to monotonic feature scaling, keeping the preprocessor uniform ensures pipeline reusability across all algorithm types.

---

### Q24: Why use OneHotEncoder with `drop='first'`?
**Answer:**  
`OneHotEncoder` converts categorical variables into binary dummy columns. Setting `drop='first'` removes the first category level for each categorical variable (e.g., for `gender`, creating only `gender_Male` instead of both `gender_Male` and `gender_Female`). This eliminates multicollinearity (the "dummy variable trap"), which is critical for linear models like Logistic Regression to prevent singular covariance matrices and inflated coefficient variance.

---

### Q25: What is hyperparameter tuning?
**Answer:**  
Hyperparameters are configuration variables set before training that control the learning process and model architecture (e.g., `max_depth`, `n_estimators`, `learning_rate`), in contrast to model parameters (such as tree split thresholds or logistic weights) which are learned automatically during training. Tuning searches for the hyperparameter combination that maximizes out-of-sample generalization.

---

### Q26: Why use GridSearchCV over manual tuning?
**Answer:**  
`GridSearchCV` automates an exhaustive, reproducible search across a specified hyperparameter grid, evaluating every candidate combination using $K$-fold stratified cross-validation on the training set. It avoids human bias, eliminates manual trial-and-error, and directly optimizes a target metric (`roc_auc`).

---

### Q27: How did you select the final model?
**Answer:**  
The model was selected based on multi-criteria benchmarking rather than accuracy alone:
1. **ROC-AUC:** Random Forest achieved **0.8432** vs. 0.8417 for Logistic Regression.
2. **Recall & Precision Trade-off:** Random Forest achieved **78.88% Recall** while maintaining **53.54% Precision** (Logistic Regression achieved 78.34% Recall at only 50.43% Precision).
3. **Robustness:** 5-fold cross-validation demonstrated low standard deviation ($\pm 1.06\%$), indicating high stability across validation folds.

---

### Q28: How does Random Forest calculate feature importance?
**Answer:**  
Random Forest computes **Mean Decrease in Impurity (MDI)**, also known as Gini importance. For each feature, it sums the total reduction in Gini impurity brought by all splits on that feature across all trees in the forest, normalized by the total number of trees. Features that frequently split nodes near the top of trees and achieve clean separation receive higher importance scores.

---

### Q29: How would you deploy this model in a production environment?
**Answer:**  
1. **Model Serialization:** We packaged the preprocessing `ColumnTransformer` and the tuned model together into a single pipeline serialized with `joblib` (`best_model.pkl`).
2. **API Layer:** Expose a REST API using **FastAPI** or **Flask** with a `/predict` endpoint that accepts JSON customer payloads.
3. **Interactive UI:** Deploy the created **Streamlit** dashboard for retention team self-service.
4. **Batch Inference:** Schedule a daily SQL/Airflow batch job to score all existing subscribers and flag high-risk accounts in Salesforce or Zendesk.
5. **Monitoring:** Implement drift detection (Evidently AI / MLflow) to monitor data drift and performance degradation over time.

---

### Q30: If you had more time and data, how would you improve this project?
**Answer:**  
1. **Dynamic Time-Series & Event Streams:** Incorporate temporal usage trajectories (e.g., data consumption decline over the last 3 months, payment delay trends).
2. **Survival Analysis:** Implement Cox Proportional Hazards or Random Survival Forests to predict *when* a customer will churn (time-to-event) rather than just a binary outcome.
3. **Advanced Explainability:** Integrate TreeSHAP (SHapley Additive exPlanations) for local, customer-level waterfall explanations in the Streamlit UI.
4. **Threshold Optimization:** Tune the classification threshold based on a formal business utility function:
$$\text{Expected Value} = TP \times (\text{Saved Revenue} - \text{Incentive Cost}) - FP \times (\text{Incentive Cost}) - FN \times (\text{Lost Revenue})$$

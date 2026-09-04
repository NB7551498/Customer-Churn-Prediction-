"""
Standalone Prediction Script for Customer Churn.
Loads the serialized best model pipeline (models/best_model.pkl) and performs
inference on new customer records, outputting predicted churn status, probability,
and risk tier.
"""

import os
import sys
import joblib
import pandas as pd


MODEL_PATH = os.path.join("models", "best_model.pkl")


def load_model(model_path: str = MODEL_PATH):
    """Load the trained machine learning pipeline."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please run src/train.py first.")
    return joblib.load(model_path)


def predict_churn(customer_data: dict, model=None) -> dict:
    """
    Predict churn status and probability for a single customer.

    Args:
        customer_data (dict): Dictionary of customer features matching the dataset schema.
        model: Preloaded model pipeline (optional).

    Returns:
        dict: Prediction results including probability, status, and risk tier.
    """
    if model is None:
        model = load_model()

    input_df = pd.DataFrame([customer_data])

    # Model pipeline automatically handles scaling and one-hot encoding
    churn_prob = model.predict_proba(input_df)[0, 1]
    churn_pred = model.predict(input_df)[0]

    if churn_prob < 0.30:
        risk_tier = "Low Risk"
        recommendation = "Customer is satisfied and stable. Continue regular engagement."
    elif churn_prob < 0.60:
        risk_tier = "Medium Risk"
        recommendation = "Customer is at moderate risk. Offer customer loyalty discount or check service satisfaction."
    else:
        risk_tier = "High Risk"
        recommendation = "Immediate retention action required! Provide dedicated account manager, discounted annual contract, and tech support bundle."

    return {
        'churn_prediction': "Likely to Churn" if churn_pred == 1 else "Retained",
        'churn_probability_pct': round(churn_prob * 100, 2),
        'risk_tier': risk_tier,
        'action_recommendation': recommendation
    }


def format_report(customer_name: str, result: dict):
    print("\n" + "=" * 50)
    print(f"Customer Churn Prediction: {customer_name}")
    print("=" * 50)
    print(f"Prediction        : {result['churn_prediction']}")
    print(f"Churn Probability : {result['churn_probability_pct']}%")
    print(f"Risk Tier         : {result['risk_tier']}")
    print(f"Recommendation    : {result['action_recommendation']}")
    print("=" * 50)


if __name__ == "__main__":
    model = load_model()

    # Sample Customer 1: High Risk Profile (Month-to-month contract, fiber optic, high charges, low tenure)
    high_risk_customer = {
        'gender': 'Female',
        'SeniorCitizen': '0',
        'Partner': 'No',
        'Dependents': 'No',
        'tenure': 2,
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'InternetService': 'Fiber optic',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'No',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'Yes',
        'StreamingMovies': 'Yes',
        'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 89.50,
        'TotalCharges': 179.00
    }

    # Sample Customer 2: Low Risk Profile (Two-year contract, long tenure, security features)
    low_risk_customer = {
        'gender': 'Male',
        'SeniorCitizen': '0',
        'Partner': 'Yes',
        'Dependents': 'Yes',
        'tenure': 60,
        'PhoneService': 'Yes',
        'MultipleLines': 'Yes',
        'InternetService': 'DSL',
        'OnlineSecurity': 'Yes',
        'OnlineBackup': 'Yes',
        'DeviceProtection': 'Yes',
        'TechSupport': 'Yes',
        'StreamingTV': 'No',
        'StreamingMovies': 'No',
        'Contract': 'Two year',
        'PaperlessBilling': 'No',
        'PaymentMethod': 'Bank transfer (automatic)',
        'MonthlyCharges': 55.20,
        'TotalCharges': 3312.00
    }

    res1 = predict_churn(high_risk_customer, model)
    format_report("Customer #101 (New Month-to-Month Fiber Subscriber)", res1)

    res2 = predict_churn(low_risk_customer, model)
    format_report("Customer #202 (Long-term Bundled 2-Year Contract)", res2)

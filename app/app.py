"""
Streamlit Web Application for Customer Churn Prediction and Retention Intelligence.
Allows business retention managers and analysts to interactively assess customer churn risk,
view predicted probability, risk tiers, and get actionable retention recommendations.
"""

import os
import joblib
import pandas as pd
import streamlit as st


# Set Page Configuration
st.set_page_config(
    page_title="Customer Churn Prediction & Retention Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card-low {
        background-color: #ECFDF5;
        border-left: 5px solid #10B981;
        padding: 1.2rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .metric-card-medium {
        background-color: #FFFBEB;
        border-left: 5px solid #F59E0B;
        padding: 1.2rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .metric-card-high {
        background-color: #FEF2F2;
        border-left: 5px solid #EF4444;
        padding: 1.2rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_churn_model():
    """Load pre-trained machine learning pipeline."""
    possible_paths = [
        os.path.join("models", "best_model.pkl"),
        os.path.join("..", "models", "best_model.pkl"),
        os.path.join("customer-churn-prediction", "models", "best_model.pkl")
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return joblib.load(path)
    return None


model = load_churn_model()

# Header
st.markdown('<div class="main-header">🎯 Customer Churn Prediction & Retention Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Machine Learning Classification System to identify churn risk and proactively protect recurring revenue.</div>', unsafe_allow_html=True)

if model is None:
    st.error("⚠️ Model artifact not found at `models/best_model.pkl`. Please execute `src/train.py` first to train and serialize the model.")
    st.stop()

# Layout: Two main tabs
tab_pred, tab_insights = st.tabs(["🔮 Single Customer Risk Assessment", "📈 Model Benchmarks & Metrics"])

with tab_pred:
    st.markdown("### Customer Profile Input")
    st.info("Enter customer subscription details below to evaluate their churn probability.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("👤 Demographics")
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen", ["0", "1"], format_func=lambda x: "Yes" if x == "1" else "No")
        partner = st.selectbox("Has Partner?", ["No", "Yes"])
        dependents = st.selectbox("Has Dependents?", ["No", "Yes"])
        tenure = st.slider("Tenure (Months with Company)", min_value=0, max_value=72, value=12, step=1)

    with col2:
        st.subheader("📞 Services Subscribed")
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

    with col3:
        st.subheader("💳 Billing & Contract")
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
        )
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=65.0, step=0.5)
        # Suggest realistic TotalCharges
        suggested_total = round(tenure * monthly_charges, 2)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=float(suggested_total), step=10.0)

    # Prediction Action
    st.markdown("---")
    predict_btn = st.button("🚀 Predict Churn Probability", use_container_width=True, type="primary")

    if predict_btn:
        customer_dict = {
            'gender': gender,
            'SeniorCitizen': senior_citizen,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phone_service,
            'MultipleLines': multiple_lines,
            'InternetService': internet_service,
            'OnlineSecurity': online_security,
            'OnlineBackup': online_backup,
            'DeviceProtection': device_protection,
            'TechSupport': tech_support,
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies,
            'Contract': contract,
            'PaperlessBilling': paperless_billing,
            'PaymentMethod': payment_method,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges
        }

        input_df = pd.DataFrame([customer_dict])

        # Inference
        churn_prob = model.predict_proba(input_df)[0, 1]
        churn_pct = churn_prob * 100

        res_col1, res_col2 = st.columns([1, 1.5])

        with res_col1:
            if churn_prob < 0.30:
                card_class = "metric-card-low"
                status_text = "🟢 Retained (Low Churn Risk)"
                tier_badge = "LOW RISK (0% - 30%)"
            elif churn_prob < 0.60:
                card_class = "metric-card-medium"
                status_text = "🟡 Moderate Churn Risk"
                tier_badge = "MEDIUM RISK (30% - 60%)"
            else:
                card_class = "metric-card-high"
                status_text = "🔴 High Churn Risk"
                tier_badge = "HIGH RISK (60% - 100%)"

            st.markdown(f"""
            <div class="{card_class}">
                <h3 style="margin-top:0;">{status_text}</h3>
                <h1 style="font-size: 3rem; margin: 0.5rem 0;">{churn_pct:.1f}%</h1>
                <p><strong>Risk Tier:</strong> {tier_badge}</p>
            </div>
            """, unsafe_allow_html=True)
            st.progress(float(churn_prob))

        with res_col2:
            st.subheader("💡 Tailored Retention Strategy")
            recommendations = []

            if contract == "Month-to-month":
                recommendations.append("🔹 **Incentivize Annual Commitment:** Customer is on a flexible Month-to-month plan. Offer a 15% discount on a 1-year or 2-year contract.")
            if payment_method == "Electronic check":
                recommendations.append("🔹 **Payment Optimization:** Electronic check has the highest churn rate. Offer a one-time $10 credit to switch to automatic credit card or bank transfer.")
            if internet_service == "Fiber optic" and tech_support == "No":
                recommendations.append("🔹 **Support Engagement:** High monthly fiber charges without dedicated Tech Support creates frustration. Offer 3 months of free priority Tech Support.")
            if tenure <= 6:
                recommendations.append("🔹 **Onboarding Support:** Customer is in the critical first-half-year retention window. Trigger proactive check-in call from customer success team.")
            if monthly_charges > 80.0 and online_security == "No":
                recommendations.append("🔹 **Value Add-on:** Bundle free Online Security & Cloud Backup to increase perceived plan value.")

            if not recommendations:
                recommendations.append("✅ Customer has strong long-term indicators (long tenure, multi-year contract, automated payment). Maintain standard loyalty rewards.")

            for rec in recommendations:
                st.markdown(rec)

with tab_insights:
    st.subheader("📊 Cross-Validation & Model Comparison")
    st.write("Performance evaluated across 5-Fold Stratified Cross-Validation on training data and benchmarked on the 20% test set:")

    if os.path.exists("reports/test_metrics_summary.csv"):
        test_summary = pd.read_csv("reports/test_metrics_summary.csv")
        st.dataframe(test_summary.style.format({
            'Accuracy': '{:.2%}',
            'Precision': '{:.2%}',
            'Recall': '{:.2%}',
            'F1-Score': '{:.2%}',
            'ROC-AUC': '{:.4f}'
        }), use_container_width=True)
    else:
        st.write("Run evaluation scripts to display benchmark metrics.")

    col_img1, col_img2 = st.columns(2)
    with col_img1:
        if os.path.exists("reports/figures/roc_curves.png"):
            st.image("reports/figures/roc_curves.png", caption="Comparative ROC Curves", use_container_width=True)
    with col_img2:
        if os.path.exists("reports/figures/feature_importance.png"):
            st.image("reports/figures/feature_importance.png", caption="Top Churn Drivers (Random Forest)", use_container_width=True)

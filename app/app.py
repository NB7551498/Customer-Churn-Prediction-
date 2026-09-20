"""
Streamlit Web Application — Responsible ML Customer Churn Dashboard.

5-tab layout:
  1. 🔮 Prediction      — submit profile, get churn risk + retention strategy
  2. 🔍 Explainability  — SHAP feature attributions from /explain
  3. ⚖️  Fairness        — group metrics + mitigation experiment results
  4. 📊 Performance     — model benchmarks, ROC, confusion matrix
  5. 📖 API Docs        — live link to Swagger UI
"""

import os

import pandas as pd
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ChurnGuard AI — Enterprise Retention Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = os.getenv("API_URL", "http://localhost:8000")

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .main-header {font-size:2rem;font-weight:700;color:#1E3A8A;margin-bottom:.2rem}
    .sub-header  {font-size:1rem;color:#4B5563;margin-bottom:1.2rem}
    .card-low    {background:#ECFDF5;border-left:5px solid #10B981;padding:1.1rem;border-radius:8px;margin-bottom:1rem}
    .card-medium {background:#FFFBEB;border-left:5px solid #F59E0B;padding:1.1rem;border-radius:8px;margin-bottom:1rem}
    .card-high   {background:#FEF2F2;border-left:5px solid #EF4444;padding:1.1rem;border-radius:8px;margin-bottom:1rem}
    .persona-badge {background:#F3F4F6;border:1px solid #E5E7EB;border-radius:8px;padding:0.75rem;margin-bottom:1rem}
    .shap-pos    {color:#EF4444;font-weight:600}
    .shap-neg    {color:#10B981;font-weight:600}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🛡️ ChurnGuard AI — Customer Retention Intelligence Platform</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Calibrated Inference · SHAP Attributions · Behavioral Personas · Automated Retention Playbooks · Bulk Scoring</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar — API status & Executive Hub
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚡ Platform Hub")
    st.markdown(f"📊 [**Open Executive Web Dashboard**]({API_URL}/dashboard)")
    st.markdown(f"📖 [**API Interactive Docs (Swagger)**]({API_URL}/docs)")
    st.divider()

    st.header("🔌 Service Diagnostics")
    try:
        resp = requests.get(f"{API_URL}/health", timeout=3)
        health = resp.json()
        st.success(f"✅ System: {health['status'].upper()}")
        st.info(f"ML Pipeline: {'✅ Ready' if health['model_loaded'] else '❌ Offline'}")
        st.info(f"K-Means Personas: {'✅ Active' if health.get('segmenter_loaded') else '❌ Offline'}")
        st.info(f"SHAP Engine: {'✅ Calibrated' if health.get('explainer_loaded') else '❌ Offline'}")
        st.caption(f"Engine Version: v{health.get('version', '2.0.0')}")
    except Exception:
        st.error("❌ API service unreachable")
        st.caption(f"Connecting to: {API_URL}")

    st.divider()
    st.caption("Dataset: Telco Customer Churn (Kaggle)\n7,043 subscribers · 19 features")

# ---------------------------------------------------------------------------
# Shared customer profile form builder
# ---------------------------------------------------------------------------

def build_customer_form(key_prefix: str = "") -> dict:
    """Renders the customer profile form and returns a payload dict."""
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("👤 Demographics")
        gender = st.selectbox("Gender", ["Female", "Male"], key=f"{key_prefix}gender")
        senior = st.selectbox("Senior Citizen", ["0", "1"], format_func=lambda x: "Yes" if x == "1" else "No", key=f"{key_prefix}senior")
        partner = st.selectbox("Has Partner?", ["No", "Yes"], key=f"{key_prefix}partner")
        dependents = st.selectbox("Has Dependents?", ["No", "Yes"], key=f"{key_prefix}dependents")
        tenure = st.slider("Tenure (months)", 0, 72, 12, key=f"{key_prefix}tenure")

    with col2:
        st.subheader("📞 Services")
        phone_service = st.selectbox("Phone Service", ["Yes", "No"], key=f"{key_prefix}phone")
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"], key=f"{key_prefix}lines")
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], key=f"{key_prefix}internet")
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"], key=f"{key_prefix}security")
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"], key=f"{key_prefix}backup")
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"], key=f"{key_prefix}device")
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"], key=f"{key_prefix}tech")
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"], key=f"{key_prefix}stv")
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"], key=f"{key_prefix}smov")

    with col3:
        st.subheader("💳 Billing & Contract")
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], key=f"{key_prefix}contract")
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"], key=f"{key_prefix}paperless")
        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            key=f"{key_prefix}payment",
        )
        monthly_charges = st.number_input("Monthly Charges ($)", 18.0, 120.0, 65.0, step=0.5, key=f"{key_prefix}monthly")
        suggested_total = round(tenure * monthly_charges, 2)
        total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, float(suggested_total), step=10.0, key=f"{key_prefix}total")

    return {
        "gender": gender, "SeniorCitizen": senior,
        "Partner": partner, "Dependents": dependents, "tenure": tenure,
        "PhoneService": phone_service, "MultipleLines": multiple_lines,
        "InternetService": internet_service, "OnlineSecurity": online_security,
        "OnlineBackup": online_backup, "DeviceProtection": device_protection,
        "TechSupport": tech_support, "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies, "Contract": contract,
        "PaperlessBilling": paperless_billing, "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges, "TotalCharges": total_charges,
    }


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_predict, tab_batch, tab_explain, tab_fairness, tab_perf, tab_docs = st.tabs([
    "🔮 Real-Time Assessment",
    "📁 Batch CSV Scoring",
    "🔍 Explainability (SHAP)",
    "⚖️ Fairness Analysis",
    "📊 Model Benchmarks",
    "📖 API Docs",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — REAL-TIME ASSESSMENT
# ══════════════════════════════════════════════════════════════════════════════
with tab_predict:
    st.markdown("### Customer Profile — Retention Intelligence & Scoring")
    st.info("Configure customer attributes below and click **Evaluate Churn Risk** to run the complete ML pipeline.")

    payload = build_customer_form(key_prefix="pred_")
    st.divider()

    if st.button("🚀 Evaluate Churn Risk & Generate Playbook", use_container_width=True, type="primary", key="predict_btn"):
        try:
            resp = requests.post(f"{API_URL}/predict", json=payload, timeout=12)
            if resp.status_code == 200:
                data = resp.json()
                prob = data["churn_probability"]
                pct = prob * 100
                risk = data["risk_tier"].capitalize()

                card_class = {"Low": "card-low", "Medium": "card-medium", "High": "card-high"}.get(risk, "card-medium")
                icon = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}.get(risk, "🟡")

                res_col1, res_col2 = st.columns([1, 1.3])
                with res_col1:
                    st.markdown(f"""
                    <div class="{card_class}">
                        <h3 style="margin-top:0">{icon} {risk} Churn Risk</h3>
                        <h1 style="font-size:3.2rem;margin:.2rem 0">{pct:.1f}%</h1>
                        <p><strong>Decision Cutoff:</strong> {data['decision_threshold']:.2f}</p>
                        <p><strong>Predicted Outcome:</strong> {'CHURN ⚠️' if data['is_churn'] else 'RETAIN ✅'}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.progress(float(prob))

                    # Customer Persona
                    seg = data.get("segmentation", {})
                    st.markdown(f"""
                    <div class="persona-badge">
                        <h4 style="margin:0 0 0.2rem 0">🏷️ Segment: {seg.get('segment_name', 'General Subscriber')}</h4>
                        <p style="margin:0 0 0.4rem 0;color:#4B5563;font-size:0.9rem">{seg.get('description', '')}</p>
                        <span style="font-weight:600;font-size:0.85rem">Risk Classification: {seg.get('risk_profile', 'Standard')}</span>
                    </div>
                    """, unsafe_allow_html=True)

                with res_col2:
                    st.subheader("💡 Automated Retention Playbook")
                    pb = data.get("retention_playbook", {})
                    st.markdown(f"**Primary Driver:** {pb.get('primary_issue', 'General retention vulnerability.')}")

                    st.markdown("**Recommended Interventions:**")
                    for act in pb.get("recommended_actions", []):
                        st.markdown(f"- ✅ {act}")

                    st.divider()
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Risk Reduction", pb.get("projected_risk_reduction", "N/A"))
                    m2.metric("ARR at Risk", pb.get("estimated_arr_at_risk", "N/A"))
                    m3.metric("Projected Saved CLV", pb.get("projected_clv_preserved", "N/A"))

                # Attribution factors
                st.divider()
                st.subheader("🔍 Top Predictive Factors")
                f_col1, f_col2 = st.columns(2)
                with f_col1:
                    st.markdown("**🔴 Churn Vulnerability Factors (+ Impact)**")
                    for d in data.get("top_churn_drivers", []):
                        st.markdown(f"• **{d['factor']}** ({d['relative_pct']})")
                with f_col2:
                    st.markdown("**🟢 Retention Anchors (- Impact)**")
                    for d in data.get("top_retention_factors", []):
                        st.markdown(f"• **{d['factor']}** ({d['relative_pct']})")

                # AI Outreach Draft
                draft = pb.get("llm_outreach_draft", {})
                if draft:
                    with st.expander("✉️ View AI-Generated Outreach Communication Script", expanded=False):
                        st.markdown(f"**Channel:** {draft.get('channel', 'Email / Direct Call')}")
                        st.markdown(f"**Subject:** `{draft.get('subject', '')}`")
                        st.text_area("Outreach Template", draft.get("body", ""), height=130)
            else:
                st.error(f"API error {resp.status_code}: {resp.text}")
        except requests.exceptions.ConnectionError:
            st.error("Cannot reach the API. Is it running? Check the sidebar.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — BATCH CSV SCORING
# ══════════════════════════════════════════════════════════════════════════════
with tab_batch:
    st.markdown("### 📁 Batch Customer Evaluation")
    st.info("Upload customer CSV records to calculate churn probabilities and prioritize retention outreach.")

    uploaded_file = st.file_uploader("Upload Customer CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            df_in = pd.read_csv(uploaded_file)
            st.write(f"Loaded **{len(df_in):,}** customer records.")
            st.dataframe(df_in.head(5), use_container_width=True)

            if st.button("⚡ Score Entire Portfolio", type="primary", key="batch_score_btn"):
                with st.spinner("Evaluating subscribers through ML pipeline..."):
                    uploaded_file.seek(0)
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
                    resp = requests.post(f"{API_URL}/predict/batch", files=files, timeout=60)

                    if resp.status_code == 200:
                        import io
                        df_scored = pd.read_csv(io.StringIO(resp.text))
                        st.success(f"Successfully scored {len(df_scored):,} accounts!")

                        # Metrics
                        b1, b2, b3 = st.columns(3)
                        high_cnt = (df_scored["RiskTier"] == "HIGH").sum()
                        med_cnt = (df_scored["RiskTier"] == "MEDIUM").sum()
                        low_cnt = (df_scored["RiskTier"] == "LOW").sum()
                        b1.metric("🔴 High Risk Accounts", f"{high_cnt:,}")
                        b2.metric("🟡 Medium Risk Accounts", f"{med_cnt:,}")
                        b3.metric("🟢 Low Risk Accounts", f"{low_cnt:,}")

                        st.dataframe(df_scored, use_container_width=True)

                        st.download_button(
                            label="📥 Download Prioritized Outreach CSV",
                            data=resp.text,
                            file_name="scored_retention_prioritized.csv",
                            mime="text/csv",
                        )
                    else:
                        st.error(f"Batch evaluation error: {resp.text}")
        except Exception as err:
            st.error(f"Could not parse uploaded CSV: {err}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════
with tab_explain:
    st.markdown("### SHAP Feature Attributions — Why Did the Model Predict This?")
    st.info(
        "**Positive SHAP values** push the prediction toward churn (red). "
        "**Negative SHAP values** push it toward retention (green). "
        "Fill in the same profile as Prediction tab and click Explain."
    )

    payload_exp = build_customer_form(key_prefix="exp_")
    st.divider()

    if st.button("🔍 Explain Prediction", use_container_width=True, type="primary", key="explain_btn"):
        try:
            resp = requests.post(f"{API_URL}/explain", json=payload_exp, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                prob = data["probability"]
                pred = data["prediction"]

                col_left, col_right = st.columns([1, 2])

                with col_left:
                    risk = "High" if prob >= 0.60 else ("Medium" if prob >= 0.30 else "Low")
                    card_class = {"Low": "card-low", "Medium": "card-medium", "High": "card-high"}[risk]
                    icon = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}[risk]
                    st.markdown(f"""
                    <div class="{card_class}">
                        <h4 style="margin-top:0">{icon} {risk} Risk</h4>
                        <h2 style="margin:.3rem 0">{prob*100:.1f}% churn</h2>
                        <p>Prediction: {'CHURN ⚠️' if pred == 1 else 'RETAIN ✅'}</p>
                    </div>
                    """, unsafe_allow_html=True)

                with col_right:
                    st.subheader("Feature Impact (SHAP Values)")
                    explanation = data["explanation"]

                    features = [e["feature"] for e in explanation]
                    impacts = [e["impact"] for e in explanation]
                    colors = ["#EF4444" if v > 0 else "#10B981" for v in impacts]

                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt

                    fig, ax = plt.subplots(figsize=(8, max(4, len(features) * 0.45)))
                    y_pos = range(len(features))
                    bars = ax.barh(y_pos, impacts, color=colors, edgecolor="white", height=0.65)
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(features, fontsize=10)
                    ax.axvline(0, color="black", linewidth=0.8, linestyle="-")
                    ax.set_xlabel("SHAP Value (impact on churn probability)", fontsize=10)
                    ax.set_title("Why did the model make this prediction?", fontsize=11, fontweight="bold")
                    ax.invert_yaxis()
                    for bar, val in zip(bars, impacts):
                        ax.text(
                            val + (0.003 if val >= 0 else -0.003),
                            bar.get_y() + bar.get_height() / 2,
                            f"{val:+.3f}",
                            va="center",
                            ha="left" if val >= 0 else "right",
                            fontsize=9,
                        )
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

                # Explanation table
                with st.expander("📋 Raw SHAP values table"):
                    df_exp = pd.DataFrame(explanation)
                    df_exp["direction"] = df_exp["impact"].apply(lambda x: "→ Churn ⚠️" if x > 0 else "→ Retain ✅")
                    st.dataframe(df_exp, use_container_width=True)

            elif resp.status_code == 503:
                st.warning("SHAP explainer is not ready. Check if the model loaded correctly.")
            else:
                st.error(f"API error {resp.status_code}: {resp.text}")
        except requests.exceptions.ConnectionError:
            st.error("Cannot reach the API. Is it running?")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — FAIRNESS
# ══════════════════════════════════════════════════════════════════════════════
with tab_fairness:
    st.markdown("### ⚖️ Fairness Analysis & Bias Mitigation")
    st.info(
        "Group performance metrics across **SeniorCitizen** and **gender** — "
        "the only sensitive attributes available in this dataset. "
        "Run `python -m src.fairness` to generate these reports."
    )

    # Fairness report
    fairness_csv = "reports/fairness_report.csv"
    if os.path.exists(fairness_csv):
        df_fair = pd.read_csv(fairness_csv)
        st.subheader("Group Performance Metrics")

        for attr, gdf in df_fair.groupby("attribute"):
            st.markdown(f"**{attr}**")
            display_cols = ["group", "n_samples", "accuracy", "precision", "recall", "f1_score", "fpr", "fnr", "selection_rate"]
            st.dataframe(
                gdf[display_cols].style
                  .format({c: "{:.3f}" for c in ["accuracy", "precision", "recall", "f1_score", "fpr", "fnr", "selection_rate"]})
                  .background_gradient(subset=["fpr"], cmap="Reds")
                  .background_gradient(subset=["accuracy"], cmap="Greens"),
                use_container_width=True,
            )
            # Compute disparity
            fpr_gap = gdf["fpr"].max() - gdf["fpr"].min()
            acc_gap = gdf["accuracy"].max() - gdf["accuracy"].min()
            col1, col2 = st.columns(2)
            col1.metric("FPR Disparity", f"{fpr_gap:.3f}", help="Max - Min false positive rate across groups")
            col2.metric("Accuracy Disparity", f"{acc_gap:.3f}", help="Max - Min accuracy across groups")
            st.divider()

        if os.path.exists("reports/figures/fairness_comparison.png"):
            st.image("reports/figures/fairness_comparison.png", caption="Fairness Metrics by Group", use_container_width=True)
    else:
        st.warning("Fairness report not found. Run: `python -m src.fairness`")

    # Mitigation experiment
    mit_csv = "reports/mitigation_experiment.csv"
    if os.path.exists(mit_csv):
        st.subheader("🔬 Bias Mitigation Experiment — Threshold Adjustment")
        st.markdown(
            "**Method:** Post-processing threshold adjustment per group (SeniorCitizen). "
            "Raises the classification threshold for the non-senior group to equalise FPR."
        )
        df_mit = pd.read_csv(mit_csv)
        before = df_mit[df_mit["phase"] == "before"][["group", "accuracy", "recall", "fpr", "fnr"]].set_index("group")
        after  = df_mit[df_mit["phase"] == "after"][["group", "accuracy", "recall", "fpr", "fnr"]].set_index("group")
        comparison = before.join(after, lsuffix="_before", rsuffix="_after")
        st.dataframe(comparison.style.format("{:.3f}"), use_container_width=True)

        st.markdown("""
        > **Trade-off discussion:** Threshold adjustment reduces the FPR gap between senior and non-senior customers
        > at the cost of a small accuracy and recall reduction. This is the inherent trade-off in fairness interventions:
        > improving equity for one group often slightly reduces aggregate performance. The optimal operating point
        > depends on business and ethical priorities — there is no universally "fair" solution.
        """)

        if os.path.exists("reports/figures/mitigation_comparison.png"):
            st.image("reports/figures/mitigation_comparison.png", caption="Before vs After Mitigation", use_container_width=True)

    st.markdown("---")
    st.markdown("""
    **Documented Limitations:**
    - Only **SeniorCitizen** and **gender** are available as sensitive attributes. Race, ethnicity, and disability status are absent.
    - Senior citizen subgroup represents ~16% of the dataset — metrics carry higher variance.
    - Mitigation uses post-processing only; in-processing (adversarial debiasing) would require full retraining.
    """)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
with tab_perf:
    st.markdown("### 📊 Model Performance Benchmarks")

    # Live metrics from API
    try:
        resp = requests.get(f"{API_URL}/metrics", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            st.caption(f"Best model: **{data['best_model']}**")
            df_metrics = pd.DataFrame(data["models"])
            st.dataframe(
                df_metrics.style.format({
                    "accuracy": "{:.2%}", "precision": "{:.2%}",
                    "recall": "{:.2%}", "f1_score": "{:.2%}", "roc_auc": "{:.4f}",
                }).highlight_max(subset=["accuracy", "roc_auc", "f1_score"], color="#d1fae5"),
                use_container_width=True,
            )
    except Exception:
        # Fall back to CSV
        if os.path.exists("reports/test_metrics_summary.csv"):
            df = pd.read_csv("reports/test_metrics_summary.csv")
            st.dataframe(df.style.format({
                "Accuracy": "{:.2%}", "Precision": "{:.2%}",
                "Recall": "{:.2%}", "F1-Score": "{:.2%}", "ROC-AUC": "{:.4f}",
            }), use_container_width=True)

    # Model info
    try:
        resp_info = requests.get(f"{API_URL}/model-info", timeout=5)
        if resp_info.status_code == 200:
            meta = resp_info.json()
            st.subheader("🏷️ Model Metadata")
            cols = st.columns(4)
            cols[0].metric("Algorithm", meta["algorithm"])
            cols[1].metric("Version", meta["version"])
            cols[2].metric("Features", meta["features"])
            cols[3].metric("Training Date", meta["training_date"])
            st.caption(meta.get("description", ""))
    except Exception:
        pass

    # Plots
    st.subheader("📈 Visualisations")
    fig_paths = {
        "ROC Curves": "reports/figures/roc_curves.png",
        "Feature Importance": "reports/figures/feature_importance.png",
        "Financial Threshold Curve": "reports/figures/financial_threshold_curve.png",
        "Confusion Matrix": "reports/figures/confusion_matrix.png",
    }
    available = {k: v for k, v in fig_paths.items() if os.path.exists(v)}
    if available:
        cols = st.columns(min(2, len(available)))
        for i, (caption, path) in enumerate(available.items()):
            cols[i % 2].image(path, caption=caption, use_container_width=True)
    else:
        st.info("Run `python src/evaluate.py` to generate performance plots.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — API DOCS
# ══════════════════════════════════════════════════════════════════════════════
with tab_docs:
    st.markdown("### 📖 Interactive API Documentation")
    st.markdown(f"The full Swagger UI is available at **[{API_URL}/docs]({API_URL}/docs)**")
    st.markdown(f"ReDoc is available at **[{API_URL}/redoc]({API_URL}/redoc)**")

    st.subheader("Available Endpoints")
    endpoints = [
        ("GET",  "/dashboard",     "Executive",   "Full-screen responsive Tailwind CSS + Chart.js analytics dashboard"),
        ("POST", "/predict",       "Inference",   "Real-time scoring, persona segmentation, SHAP drivers & retention playbook"),
        ("POST", "/predict/batch", "Inference",   "Bulk CSV scoring, churn prioritization, and downloadable enriched CSV"),
        ("POST", "/explain",       "Inference",   "Direct SHAP feature attributions and impact percentages"),
        ("GET",  "/analytics",     "Analytics",   "Portfolio MRR at risk, segment distributions, and risk breakdowns"),
        ("GET",  "/model/metrics", "MLOps",       "5-model stratified cross-validation benchmark comparison table"),
        ("GET",  "/monitoring",    "MLOps",       "Latency, throughput, and data drift telemetry indicators"),
        ("POST", "/auth/login",    "Security",    "JWT authentication token generator with bcrypt credentials"),
        ("GET",  "/health",        "Diagnostics", "Liveness probe for pipeline, segmenter, and explainer models"),
        ("GET",  "/model-info",    "Diagnostics", "Production model metadata and versioning specification"),
    ]
    df_ep = pd.DataFrame(endpoints, columns=["Method", "Endpoint", "Category", "Description"])
    st.dataframe(df_ep, use_container_width=True, hide_index=True)

    st.subheader("Example: POST /predict")
    st.code("""
curl -X POST http://localhost:8000/predict \\
  -H "Content-Type: application/json" \\
  -d '{
    "gender": "Female", "SeniorCitizen": "0",
    "Partner": "No", "Dependents": "No", "tenure": 3,
    "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "Yes", "StreamingMovies": "Yes",
    "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 89.50, "TotalCharges": 268.50
  }'
    """, language="bash")

    st.subheader("Example: POST /explain")
    st.code("""
# Same payload as /predict — returns SHAP attributions
curl -X POST http://localhost:8000/explain \\
  -H "Content-Type: application/json" \\
  -d '{ ... same payload ... }'

# Response:
{
  "prediction": 1,
  "probability": 0.87,
  "explanation": [
    {"feature": "Contract_Two year",            "impact": -0.42},
    {"feature": "tenure",                       "impact": -0.31},
    {"feature": "InternetService_Fiber optic",  "impact":  0.28},
    {"feature": "PaymentMethod_Electronic check","impact":  0.21}
  ]
}
    """, language="json")

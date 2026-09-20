"""
Unit tests for ChurnGuard AI Recommendation Engine.
"""

from src.recommendation import generate_retention_playbook


def test_generate_retention_playbook_structure():
    """Verify retention playbook structure and financial calculations."""
    profile = {
        "customerID": "CUST-9999",
        "tenure": 2,
        "Contract": "Month-to-month",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 95.0,
        "TotalCharges": 190.0,
        "TechSupport": "No",
    }
    segment_info = {
        "segment_id": 1,
        "segment_name": "High-Value At-Risk",
    }
    shap_drivers = [
        {"factor": "Contract Commitment", "impact_score": 0.35, "relative_pct": "+35%"},
        {"factor": "Payment Method", "impact_score": 0.22, "relative_pct": "+22%"},
    ]

    playbook = generate_retention_playbook(
        customer_profile=profile,
        churn_probability=0.88,
        risk_tier="HIGH",
        segment_info=segment_info,
        shap_drivers=shap_drivers,
    )

    assert "primary_issue" in playbook
    assert "recommended_actions" in playbook
    assert len(playbook["recommended_actions"]) > 0
    assert "projected_risk_reduction" in playbook
    assert "estimated_arr_at_risk" in playbook
    assert "projected_clv_preserved" in playbook
    assert "llm_outreach_draft" in playbook
    assert "subject" in playbook["llm_outreach_draft"]
    assert "body" in playbook["llm_outreach_draft"]
    assert "channel" in playbook["llm_outreach_draft"]

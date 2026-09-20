"""
ChurnGuard AI — AI Retention Recommendation Engine.
Synthesizes customer churn probability, K-Means behavioral segment, and individual
SHAP feature drivers into high-impact, actionable customer retention playbooks.
"""

from typing import Any, Dict, List


def generate_retention_playbook(
    customer_profile: Dict[str, Any],
    churn_probability: float,
    risk_tier: str,
    segment_info: Dict[str, Any],
    shap_drivers: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Generate tailored business intervention strategy, estimated ROI, and
    LLM customer success outreach prompt.
    """
    monthly_charge = float(customer_profile.get("MonthlyCharges", 0.0))
    tenure = int(customer_profile.get("tenure", 0))
    contract = str(customer_profile.get("Contract", ""))
    payment_method = str(customer_profile.get("PaymentMethod", ""))
    tech_support = str(customer_profile.get("TechSupport", ""))
    segment_name = segment_info.get("segment_name", "General Subscriber")

    annual_revenue = round(monthly_charge * 12, 2)
    actions: List[str] = []
    primary_issue = "General subscription risk."

    # Identify primary vulnerability from profile & SHAP
    top_driver_names = [d["factor"].lower() for d in shap_drivers[:3]]

    if "month-to-month" in contract.lower() or any("contract" in d for d in top_driver_names):
        primary_issue = "High volatility due to flexible month-to-month commitment."
        actions.append("Present 15% discount incentive on a 1-year loyalty agreement.")

    if "electronic check" in payment_method.lower() or any("payment" in d for d in top_driver_names):
        actions.append("Offer a one-time $10 bill credit to enroll in automated bank transfer or card billing.")

    if tech_support.lower() == "no" or any("support" in d for d in top_driver_names):
        actions.append("Bundle complimentary priority 24/7 TechSupport for 6 months.")

    if tenure <= 6 or any("tenure" in d for d in top_driver_names):
        actions.append("Assign a dedicated Customer Success Specialist for an onboarding health-check call.")

    if not actions:
        if risk_tier == "High":
            actions.append("Dispatch urgent account manager escalation with tailored renewal terms.")
        elif risk_tier == "Medium":
            actions.append("Send mid-contract satisfaction check-in with optional complimentary add-on.")
        else:
            actions.append("Maintain standard service engagement and enroll in quarterly loyalty perks.")

    # Financial ROI calculation
    if risk_tier == "High":
        projected_reduction = "-35% churn probability"
        expected_roi = f"${round(annual_revenue * 0.70, 2):,.2f} preserved recurring ARR"
    elif risk_tier == "Medium":
        projected_reduction = "-20% churn probability"
        expected_roi = f"${round(annual_revenue * 0.40, 2):,.2f} preserved recurring ARR"
    else:
        projected_reduction = "Stable retention baseline"
        expected_roi = "Sustained account health"

    # Draft LLM Outreach Script
    outreach_subject = f"Special appreciation offer for your {customer_profile.get('InternetService', 'Telecom')} account"
    outreach_body = (
        f"Hi there,\n\n"
        f"We noticed you've been with us for {tenure} months. To show our appreciation, "
        f"we'd love to offer you an exclusive renewal benefit: "
        f"{actions[0].replace('Present ', '').replace('Offer ', '')}\n\n"
        f"Let us know if we can assist you with your subscription today!\n\n"
        f"Best regards,\nYour Customer Retention Team"
    )

    return {
        "primary_issue": primary_issue,
        "segment": segment_name,
        "recommended_actions": actions,
        "projected_risk_reduction": projected_reduction,
        "estimated_arr_at_risk": f"${annual_revenue:,.2f}",
        "projected_clv_preserved": expected_roi,
        "llm_outreach_draft": {
            "channel": "Email / Relationship Manager",
            "subject": outreach_subject,
            "body": outreach_body,
        },
    }

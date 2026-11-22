"""Local demo for a complaints handler.

    streamlit run src/fos_triage/streamlit_app.py
"""

from __future__ import annotations

import streamlit as st

from fos_triage.service import DISCLAIMER, TriageService

st.set_page_config(page_title="Complaint triage", layout="wide")
st.title("Complaint triage")
st.caption("For a UK financial-services complaints team. Scores compare a narrative with published ombudsman decisions.")
st.info(DISCLAIMER)

@st.cache_resource
def load_service() -> TriageService:
    return TriageService.load()

try:
    service = load_service()
except FileNotFoundError:
    st.error("Train the models first: python -m fos_triage.train")
    st.stop()

with st.form("complaint"):
    business = st.text_input("Business name (optional)", placeholder="Used only to label the firm type")
    text = st.text_area(
        "Complaint narrative",
        height=240,
        placeholder="Paste the customer's complaint and the factual background. Do not paste the ombudsman's decision.",
    )
    submitted = st.form_submit_button("Triage", type="primary")

if submitted:
    if len(text.strip()) < 40:
        st.warning("Paste a bit more of the complaint. A sentence or two is not enough to code the case.")
    else:
        result = service.triage(text, business or None)
        if result["intake_text_used"]:
            st.warning("The paste looked like a full ombudsman decision. Scoring used the complaint and background only.")
        product = result["product"]
        uphold = result["uphold"]
        left, right = st.columns(2)
        with left:
            st.subheader("Product")
            st.metric("Most likely code", product["label"], delta=f"{product['confidence']:.0%} confidence")
            st.dataframe(product["top3"], hide_index=True, use_container_width=True)
        with right:
            st.subheader("Uphold likelihood")
            st.metric("Probability", f"{uphold['probability']:.0%}", delta=f"{uphold['priority']} priority")
            st.caption(f"Review threshold on the later-period validation slice: {uphold['threshold']:.0%}.")
            if result["firm_type"]:
                st.write(f"Firm type from the name: **{result['firm_type']}**")
        st.subheader("Case note")
        st.write(result["summary"])
        if result["key_issues"]:
            st.write("Key phrases: " + ", ".join(result["key_issues"]))

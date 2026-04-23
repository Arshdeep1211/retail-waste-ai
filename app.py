import streamlit as st

st.set_page_config(
    page_title="Retail AI",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Retail AI – Waste Reduction System")
st.subheader("Welcome to your AI-powered retail copilot")

st.markdown("### Choose a module")

col1, col2, col3 = st.columns(3)

with col1:
    st.info("📊 Dashboard")
    st.caption("View forecasts, risk, and recommendations")

with col2:
    st.info("📂 Upload Data")
    st.caption("Upload sales, inventory, and product files")

with col3:
    st.info("🤖 AI Copilot")
    st.caption("Ask questions about stock, orders, and risk")

st.markdown("---")
st.write("Use the sidebar on the left to open each module.")

import streamlit as st
import pandas as pd

from src.forecast import forecast_next_day
from src.risk import compute_risk
from src.recommend import recommend_actions
from src.customer_ai import suggest_campaign

st.set_page_config(page_title="Retail AI", layout="wide")

st.title("Retail AI – Waste Reduction System")

# Load data
sales = pd.read_csv("data/sales_daily.csv")
inventory = pd.read_csv("data/inventory_snapshot.csv")
products = pd.read_csv("data/products.csv")
customers = pd.read_csv("data/customers.csv")
purchases = pd.read_csv("data/customer_purchases.csv")

store_id = st.number_input("Store ID", value=1)

# ======================
# FORECAST
# ======================
forecast = forecast_next_day(sales, store_id)

st.subheader("Forecast")

if forecast.empty:
    st.warning("⚠️ No forecast data for this store")
    st.stop()

st.dataframe(forecast)

# ======================
# INVENTORY FILTER (IMPORTANT FIX)
# ======================
inventory_store = inventory[inventory["store_id"] == store_id]

if inventory_store.empty:
    st.warning("⚠️ No inventory data for this store")
    st.stop()

# ======================
# RISK
# ======================
try:
    risk = compute_risk(forecast, inventory_store, products)

    st.subheader("Risk Analysis")

    if risk.empty:
        st.info("No risk data available")
    else:
        st.dataframe(risk)

except Exception as e:
    st.error(f"Risk calculation failed: {e}")
    st.stop()

# ======================
# RECOMMENDATIONS
# ======================
try:
    recommend = recommend_actions(risk)

    st.subheader("Recommended Actions")

    if recommend.empty:
        st.info("No recommendations available")
    else:
        st.dataframe(recommend)

except Exception as e:
    st.error(f"Recommendation failed: {e}")
    st.stop()

# ======================
# CUSTOMER AI
# ======================
try:
    campaigns = suggest_campaign(recommend, customers, purchases)

    st.subheader("Customer Campaigns")

    if campaigns.empty:
        st.info("No campaigns generated")
    else:
        st.dataframe(campaigns)

except Exception as e:
    st.error(f"Campaign generation failed: {e}")

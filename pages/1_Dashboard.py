import streamlit as st
import pandas as pd

from src.forecast import forecast_next_day
from src.risk import compute_risk
from src.recommend import recommend_actions

st.title("📊 Dashboard")

# Load data
sales = pd.read_csv("data/sales_daily.csv")
inventory = pd.read_csv("data/inventory_snapshot.csv")
products = pd.read_csv("data/products.csv")

store_id = st.number_input("Store ID", value=1)

forecast = forecast_next_day(sales, store_id)
inventory_store = inventory[inventory["store_id"] == store_id]

if forecast.empty or inventory_store.empty:
    st.warning("No data for this store")
    st.stop()

risk = compute_risk(forecast, inventory_store, products)
recommend = recommend_actions(risk)

# KPIs
col1, col2, col3 = st.columns(3)
col1.metric("Total SKUs", len(forecast))
col2.metric("High Risk Items", (risk["risk_level"] == "HIGH").sum())
col3.metric("Total Stock", int(inventory_store["stock_on_hand"].sum()))

# Alerts
st.subheader("🚨 Alerts")
high_risk = risk[risk["risk_level"] == "HIGH"]

if not high_risk.empty:
    for _, row in high_risk.iterrows():
        st.error(f"{row['sku']} is HIGH risk")
else:
    st.success("No critical risks")

# Tables
st.subheader("Forecast")
st.dataframe(forecast)

st.subheader("Risk Analysis")
st.dataframe(risk)

st.subheader("Recommended Actions")
st.dataframe(recommend)

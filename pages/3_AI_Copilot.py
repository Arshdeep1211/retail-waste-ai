import streamlit as st
import pandas as pd

from src.forecast import forecast_next_day
from src.risk import compute_risk

st.title("🤖 AI Copilot")

sales = pd.read_csv("data/sales_daily.csv")
inventory = pd.read_csv("data/inventory_snapshot.csv")
products = pd.read_csv("data/products.csv")

store_id = st.number_input("Store ID", value=1)

forecast = forecast_next_day(sales, store_id)
inventory_store = inventory[inventory["store_id"] == store_id]

if forecast.empty or inventory_store.empty:
    st.warning("No data available")
    st.stop()

risk = compute_risk(forecast, inventory_store, products)

query = st.text_input("Ask your AI Copilot...")

def answer(query):
    q = query.lower()

    if "run out" in q:
        low = risk.sort_values("stock_on_hand").head(3)
        return "\n".join([f"{r['sku']} may run out soon" for _, r in low.iterrows()])

    elif "order" in q:
        reorder = risk[risk["unsold_risk"] < 0]
        return "\n".join([f"Order {r['sku']}" for _, r in reorder.iterrows()]) or "No urgent orders"

    elif "risk" in q:
        high = risk[risk["risk_level"] == "HIGH"]
        return ", ".join(high["sku"].tolist()) or "No high risks"

    return "Ask about stock, risk, or orders."

if query:
    st.chat_message("user").write(query)
    st.chat_message("assistant").write(answer(query))

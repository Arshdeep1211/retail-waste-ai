import streamlit as st
import pandas as pd

from src.forecast import forecast_next_day
from src.recommend import recommend_order
from src.risk import expiry_risk

st.set_page_config(page_title="Retail Waste AI", layout="wide")


@st.cache_data
def load_data():
    sales = pd.read_csv("data/sales_daily.csv", parse_dates=["date"])
    inv = pd.read_csv("data/inventory_snapshot.csv", parse_dates=["timestamp"])
    products = pd.read_csv("data/products.csv")
    stores = pd.read_csv("data/stores.csv")
    return sales, inv, products, stores


def get_latest_inventory(inv: pd.DataFrame, store_id: int) -> pd.DataFrame:
    store_inv = inv[inv["store_id"] == store_id].copy()
    latest = store_inv.sort_values("timestamp").groupby("sku").tail(1)
    return latest


st.title("Closed-Loop Retail AI — MVP Dashboard")
st.write("Forecast demand, recommend orders, and flag overstock risk.")

sales, inv, products, stores = load_data()

store_options = stores.set_index("store_id")["name"].to_dict()
selected_store_id = st.selectbox(
    "Select Store",
    options=list(store_options.keys()),
    format_func=lambda x: store_options[x]
)

lookback_days = st.slider("Forecast Lookback Days", min_value=2, max_value=5, value=3)
safety_factor = st.slider("Safety Factor", min_value=0.0, max_value=0.5, value=0.2, step=0.05)

pred = forecast_next_day(sales, selected_store_id, lookback_days=lookback_days)
inv_latest = get_latest_inventory(inv, selected_store_id)
orders = recommend_order(pred, inv_latest, products, safety_factor=safety_factor)
risks = expiry_risk(inv_latest, pred, products)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("SKUs tracked", len(orders))

with col2:
    total_recommended = int(orders["order_units"].sum())
    st.metric("Total recommended order units", total_recommended)

with col3:
    high_risk = (risks["risk_score"] >= 80).sum()
    st.metric("High risk SKUs", int(high_risk))

st.subheader("Today’s Recommended Order")
st.dataframe(orders, use_container_width=True)

st.subheader("Expiry / Overstock Risk Alerts")
st.dataframe(risks, use_container_width=True)

st.subheader("Summary")
st.write(
    """
    This MVP uses a simple moving-average forecast and basic overstock logic.
    Later versions can add:
    - batch expiry dates
    - markdown recommendations
    - customer discount notifications
    - POS/API integrations
    """
)

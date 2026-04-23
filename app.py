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

# Forecast
forecast = forecast_next_day(sales, store_id)
st.subheader("Forecast")
st.dataframe(forecast)

# Risk
risk = compute_risk(forecast, inventory, products)
st.subheader("Risk Analysis")
st.dataframe(risk)

# Recommendations
recommend = recommend_actions(risk)
st.subheader("Recommended Actions")
st.dataframe(recommend)

# Customer AI
campaigns = suggest_campaign(recommend, customers, purchases)
st.subheader("Customer Campaigns")
st.dataframe(campaigns)

import streamlit as st
import pandas as pd

st.title("📂 Upload Data")

sales_file = st.file_uploader("Upload Sales CSV", type=["csv"])
inventory_file = st.file_uploader("Upload Inventory CSV", type=["csv"])
products_file = st.file_uploader("Upload Products CSV", type=["csv"])

if sales_file:
    df = pd.read_csv(sales_file)
    st.success("Sales uploaded")
    st.dataframe(df.head())

if inventory_file:
    df = pd.read_csv(inventory_file)
    st.success("Inventory uploaded")
    st.dataframe(df.head())

if products_file:
    df = pd.read_csv(products_file)
    st.success("Products uploaded")
    st.dataframe(df.head())

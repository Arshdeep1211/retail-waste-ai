import streamlit as st
import pandas as pd
from src.forecast import forecast_next_day
from src.risk import compute_risk
from src.recommend import recommend_actions
from src.customer_ai import suggest_campaign
st.set_page_config(
    page_title="Retail AI Copilot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)
# ---------- STYLE ----------
st.markdown("""
<style>
:root {
    --bg: #0b1020;
    --panel: #11172b;
    --panel2: #161d36;
    --text: #f5f7ff;
    --muted: #9aa4c7;
    --accent: #6ea8fe;
    --green: #24c38e;
    --red: #ff6b6b;
    --orange: #ffb454;
    --border: rgba(255,255,255,0.08);
}
html, body, [class*="css"]  {
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
}
.stApp {
    background: linear-gradient(180deg, #0a0f1f 0%, #10182f 100%);
    color: var(--text);
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1350px;
}
h1, h2, h3 {
    color: white !important;
}
.hero {
    background: linear-gradient(135deg, rgba(110,168,254,0.18), rgba(36,195,142,0.12));
    border: 1px solid var(--border);
    border-radius: 24px;
    padding: 28px 28px 18px 28px;
    margin-bottom: 18px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.25);
}
.hero-title {
    font-size: 42px;
    font-weight: 800;
    line-height: 1.1;
    color: white;
    margin-bottom: 8px;
}
.hero-sub {
    color: #c8d2f0;
    font-size: 18px;
    margin-bottom: 0;
}
.card {
    background: rgba(17, 23, 43, 0.88);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 18px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.18);
}
.kpi {
    background: rgba(17, 23, 43, 0.92);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 18px;
    min-height: 120px;
}
.kpi-label {
    font-size: 13px;
    color: #9fb0d9;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 8px;
}
.kpi-value {
    font-size: 34px;
    font-weight: 800;
    color: white;
    line-height: 1;
}
.kpi-note {
    font-size: 13px;
    color: #96a2c8;
    margin-top: 8px;
}
.alert-high, .alert-medium, .alert-good {
    border-radius: 16px;
    padding: 14px 16px;
    margin-bottom: 10px;
    border: 1px solid var(--border);
    font-weight: 600;
}
.alert-high {
    background: rgba(255, 107, 107, 0.12);
    color: #ffd2d2;
}
.alert-medium {
    background: rgba(255, 180, 84, 0.12);
    color: #ffe2b8;
}
.alert-good {
    background: rgba(36, 195, 142, 0.10);
    color: #d2ffef;
}
.small-muted {
    color: #96a2c8;
    font-size: 13px;
}
[data-testid="stDataFrame"] {
    border-radius: 16px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.06);
}
section[data-testid="stSidebar"] {
    background: #0b1020;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
}
.stTabs [data-baseweb="tab"] {
    background-color: rgba(255,255,255,0.04);
    border-radius: 12px;
    padding: 10px 16px;
    color: white;
}
.stTabs [aria-selected="true"] {
    background-color: rgba(110,168,254,0.18) !important;
    border: 1px solid rgba(110,168,254,0.35);
}
div[data-testid="stMetric"] {
    background: rgba(17, 23, 43, 0.92);
    border: 1px solid var(--border);
    padding: 12px;
    border-radius: 18px;
}
.stTextInput > div > div > input,
.stNumberInput input,
.stSelectbox div[data-baseweb="select"] > div,
.stFileUploader {
    background: rgba(255,255,255,0.03) !important;
    color: white !important;
    border-radius: 12px !important;
}
hr {
    border-color: rgba(255,255,255,0.08);
}
</style>
""", unsafe_allow_html=True)
# ---------- HELPERS ----------
def safe_load_csv(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def compute_runout_days(row):
    pred = row.get("pred_units", 0)
    stock = row.get("stock_on_hand", 0)
    if pred and pred > 0:
        return round(stock / pred, 1)
    return None

def render_alerts(risk_df):
    if risk_df.empty:
        st.markdown('<div class="alert-medium">No data available to generate alerts.</div>', unsafe_allow_html=True)
        return

    high = risk_df[risk_df["risk_level"] == "HIGH"]
    medium = risk_df[risk_df["risk_level"] == "MEDIUM"]

    if len(high) == 0 and len(medium) == 0:
        st.markdown('<div class="alert-good">All monitored items are currently low risk.</div>', unsafe_allow_html=True)
        return

    for _, row in high.iterrows():
        st.markdown(
            f'<div class="alert-high">⚠️ High risk: <b>{row["name"]}</b> may create avoidable waste. '
            f'Unsold risk: <b>{round(row["unsold_risk"], 1)}</b> units.</div>',
            unsafe_allow_html=True
        )

    for _, row in medium.iterrows():
        st.markdown(
            f'<div class="alert-medium">🟠 Medium risk: <b>{row["name"]}</b> should be monitored. '
            f'Unsold risk: <b>{round(row["unsold_risk"], 1)}</b> units.</div>',
            unsafe_allow_html=True
        )

def copilot_answer(query, forecast_df, risk_df, recommend_df):
    q = query.lower().strip()

    if forecast_df.empty or risk_df.empty:
        return "I don’t have enough data for this store yet. Upload sales, inventory, and product data first."

    if "run out" in q or "stock over" in q or "finish" in q:
        temp = risk_df.copy()
        temp["runout_days"] = temp.apply(compute_runout_days, axis=1)
        temp = temp.sort_values(by="runout_days", na_position="last")
        rows = []
        for _, r in temp.head(3).iterrows():
            if pd.notna(r["runout_days"]):
                rows.append(f"- {r['name']} may run out in about {r['runout_days']} days.")
        return "\n".join(rows) if rows else "No imminent stockout risk detected."

    if "what should i order" in q or "order today" in q or "reorder" in q:
        need = risk_df[risk_df["unsold_risk"] < 0].copy()
        if need.empty:
            return "No urgent reorder recommendation right now."
        rows = []
        for _, r in need.iterrows():
            qty = abs(int(round(r["unsold_risk"])))
            rows.append(f"- Reorder {r['name']} soon. Estimated shortage risk: {qty} units.")
        return "\n".join(rows)

    if "risk" in q or "waste" in q:
        risky = risk_df[risk_df["risk_level"].isin(["HIGH", "MEDIUM"])]
        if risky.empty:
            return "No high or medium waste risks at the moment."
        rows = []
        for _, r in risky.iterrows():
            rows.append(f"- {r['name']}: {r['risk_level']} risk, unsold risk {round(r['unsold_risk'], 1)} units.")
        return "\n".join(rows)

    if "bread" in q or "milk" in q or "meat" in q or "yogurt" in q or "cheese" in q:
        found = None
        for _, r in risk_df.iterrows():
            if str(r["name"]).lower() in q or str(r["sku"]).lower() in q:
                found = r
                break
        if found is not None:
            rd = compute_runout_days(found)
            if rd:
                return (
                    f"{found['name']} currently has {found['stock_on_hand']} units on hand, "
                    f"predicted demand of {found['pred_units']} units/day, "
                    f"and an estimated runout in {rd} days. "
                    f"Risk level: {found['risk_level']}."
                )
            return f"{found['name']} is currently at {found['risk_level']} risk."

    return (
        "I can help with questions like:\n"
        "- Which items are risky today?\n"
        "- What should I reorder?\n"
        "- When will bread stock run out?\n"
        "- Which products may create waste?"
    )

# ---------- LOAD DEFAULT DATA ----------
sales_default = safe_load_csv("data/sales_daily.csv")
inventory_default = safe_load_csv("data/inventory_snapshot.csv")
products_default = safe_load_csv("data/products.csv")
customers_default = safe_load_csv("data/customers.csv")
purchases_default = safe_load_csv("data/customer_purchases.csv")

# ---------- HERO ----------
st.markdown("""
<div class="hero">
    <div class="hero-title">🧠 Retail AI Copilot</div>
    <p class="hero-sub">
        Inventory intelligence, expiry risk detection, decision support, and demand activation —
        all in one premium control layer for retail operations.
    </p>
</div>
""", unsafe_allow_html=True)

# ---------- TOP CONTROLS ----------
with st.container():
    left, right = st.columns([2, 1])
    with left:
        st.markdown("### Control Center")
        st.markdown('<div class="small-muted">Choose a store and optionally upload fresh datasets to simulate a live software workflow.</div>', unsafe_allow_html=True)
    with right:
        st.markdown("### Live Status")
        st.markdown('<div class="alert-good">System online • AI engine active</div>', unsafe_allow_html=True)

tab_dashboard, tab_upload, tab_copilot = st.tabs(["📊 Dashboard", "📂 Data Hub", "🤖 AI Copilot"])

# ---------- UPLOAD TAB ----------
with tab_upload:
    st.markdown("### Data Hub")
    st.markdown('<div class="small-muted">Upload files to simulate a live retail software workflow. Uploaded files are used only for the current session.</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        sales_file = st.file_uploader("Upload Sales Data", type=["csv"])
    with c2:
        inventory_file = st.file_uploader("Upload Inventory Data", type=["csv"])
    with c3:
        products_file = st.file_uploader("Upload Product Master", type=["csv"])

    sales = pd.read_csv(sales_file) if sales_file is not None else sales_default.copy()
    inventory = pd.read_csv(inventory_file) if inventory_file is not None else inventory_default.copy()
    products = pd.read_csv(products_file) if products_file is not None else products_default.copy()

    st.markdown("#### Current Dataset Preview")
    p1, p2, p3 = st.columns(3)
    with p1:
        st.write("Sales")
        st.dataframe(sales.head(), use_container_width=True)
    with p2:
        st.write("Inventory")
        st.dataframe(inventory.head(), use_container_width=True)
    with p3:
        st.write("Products")
        st.dataframe(products.head(), use_container_width=True)

# Use uploaded data if present, else defaults
sales = pd.read_csv(sales_file) if 'sales_file' in locals() and sales_file is not None else sales_default.copy()
inventory = pd.read_csv(inventory_file) if 'inventory_file' in locals() and inventory_file is not None else inventory_default.copy()
products = pd.read_csv(products_file) if 'products_file' in locals() and products_file is not None else products_default.copy()
customers = customers_default.copy()
purchases = purchases_default.copy()

# ---------- NORMALIZE COLUMNS ----------
for df in [sales, inventory, products]:
    if "sku" in df.columns:
        df["sku"] = df["sku"].astype(str)

if "store_id" in sales.columns:
    available_store_ids = sorted(sales["store_id"].dropna().unique().tolist())
else:
    available_store_ids = [1]

store_col, refresh_col = st.columns([3, 1])
with store_col:
    selected_store = st.selectbox("Store", available_store_ids, index=0)
with refresh_col:
    st.write("")
    st.write("")
    st.caption("Data-driven selection")

forecast = forecast_next_day(sales, selected_store) if not sales.empty else pd.DataFrame()
inventory_store = inventory[inventory["store_id"] == selected_store].copy() if "store_id" in inventory.columns else pd.DataFrame()

if not forecast.empty and not inventory_store.empty and not products.empty:
    risk = compute_risk(forecast, inventory_store, products)
    recommend = recommend_actions(risk)
else:
    risk = pd.DataFrame()
    recommend = pd.DataFrame()

# ---------- DASHBOARD TAB ----------
with tab_dashboard:
    st.markdown("### Executive Dashboard")

    total_skus = len(forecast) if not forecast.empty else 0
    high_risk_count = int((risk["risk_level"] == "HIGH").sum()) if not risk.empty else 0
    med_risk_count = int((risk["risk_level"] == "MEDIUM").sum()) if not risk.empty else 0
    total_stock = int(inventory_store["stock_on_hand"].sum()) if not inventory_store.empty and "stock_on_hand" in inventory_store.columns else 0

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""
        <div class="kpi">
            <div class="kpi-label">Tracked SKUs</div>
            <div class="kpi-value">{total_skus}</div>
            <div class="kpi-note">Active assortment monitored</div>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="kpi">
            <div class="kpi-label">High Risk Items</div>
            <div class="kpi-value">{high_risk_count}</div>
            <div class="kpi-note">Immediate attention required</div>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="kpi">
            <div class="kpi-label">Medium Risk Items</div>
            <div class="kpi-value">{med_risk_count}</div>
            <div class="kpi-note">Monitor and plan response</div>
        </div>
        """, unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
        <div class="kpi">
            <div class="kpi-label">Total Stock</div>
            <div class="kpi-value">{total_stock}</div>
            <div class="kpi-note">Units currently on hand</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Smart Alerts")
    render_alerts(risk)

    c1, c2 = st.columns([1.15, 1])
    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Demand Forecast")
        if forecast.empty:
            st.info("No forecast available for this store.")
        else:
            chart_df = forecast.set_index("sku")[["pred_units"]]
            st.bar_chart(chart_df, use_container_width=True)
            st.dataframe(forecast, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Action Queue")
        if recommend.empty:
            st.info("No recommendations available.")
        else:
            show_cols = [c for c in ["name", "risk_level", "unsold_risk", "action"] if c in recommend.columns]
            st.dataframe(recommend[show_cols], use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### Detailed Risk Matrix")
    if risk.empty:
        st.info("No risk matrix available.")
    else:
        st.dataframe(risk, use_container_width=True)

# ---------- COPILOT TAB ----------
with tab_copilot:
    st.markdown("### Retail AI Copilot")
    st.markdown('<div class="small-muted">Ask operational questions in natural language and get inventory-aware guidance.</div>', unsafe_allow_html=True)

    example_col1, example_col2, example_col3 = st.columns(3)
    with example_col1:
        st.markdown('<div class="card"><b>Example</b><br>When will bread stock run out?</div>', unsafe_allow_html=True)
    with example_col2:
        st.markdown('<div class="card"><b>Example</b><br>What should I reorder today?</div>', unsafe_allow_html=True)
    with example_col3:
        st.markdown('<div class="card"><b>Example</b><br>Which products are risky?</div>', unsafe_allow_html=True)

    query = st.text_input("Talk to your copilot", placeholder="Ask about stock, risk, orders, or waste...")

    if query:
        st.chat_message("user").write(query)
        answer = copilot_answer(query, forecast, risk, recommend)
        st.chat_message("assistant").write(answer)

    st.markdown("### Suggested Campaigns")
    try:
        campaigns = suggest_campaign(recommend, customers, purchases)
        if isinstance(campaigns, pd.DataFrame) and not campaigns.empty:
            st.dataframe(campaigns, use_container_width=True)
        else:
            st.info("No customer campaigns generated right now.")
    except Exception:
        st.info("Customer campaign engine is available, but no campaigns were generated for the selected store.")

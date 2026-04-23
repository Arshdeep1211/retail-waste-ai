import streamlit as st

import pandas as pd

import plotly.express as px

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

st.markdown("""

<style>

:root {

    --bg: #07111f;

    --panel: #0d1728;

    --panel2: #101c30;

    --line: rgba(255,255,255,0.08);

    --text: #f5f7ff;

    --muted: #9aa7c7;

    --green: #20c997;

    --blue: #4f7cff;

    --purple: #7c5cff;

    --orange: #f5a623;

    --red: #ff6b6b;

}

html, body, [class*="css"] {

    font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif;

}

.stApp {

    background:

      radial-gradient(circle at top right, rgba(28,160,255,0.10), transparent 30%),

      linear-gradient(180deg, #06101d 0%, #09162a 100%);

    color: var(--text);

}

.block-container {

    max-width: 1450px;

    padding-top: 1.6rem;

    padding-bottom: 2rem;

}

h1, h2, h3, h4 { color: white !important; }

.hero {

    background: linear-gradient(135deg, rgba(76,124,255,0.18), rgba(32,201,151,0.10));

    border: 1px solid var(--line);

    border-radius: 24px;

    padding: 26px 28px;

    margin-bottom: 22px;

    box-shadow: 0 18px 40px rgba(0,0,0,0.22);

}

.hero-title {

    font-size: 34px;

    font-weight: 800;

    color: white;

    margin-bottom: 8px;

}

.hero-sub {

    color: #ccd6f4;

    font-size: 17px;

    margin: 0;

}

.section-title {

    font-size: 18px;

    font-weight: 700;

    color: white;

    margin-bottom: 10px;

}

.small-muted {

    color: var(--muted);

    font-size: 13px;

}

.panel {

    background: rgba(13,23,40,0.92);

    border: 1px solid var(--line);

    border-radius: 22px;

    padding: 18px;

    box-shadow: 0 10px 30px rgba(0,0,0,0.16);

    height: 100%;

}

.kpi {

    background: rgba(13,23,40,0.95);

    border: 1px solid var(--line);

    border-radius: 20px;

    padding: 18px;

    min-height: 122px;

}

.kpi-label {

    font-size: 12px;

    letter-spacing: 0.08em;

    text-transform: uppercase;

    color: #9eb1dd;

    margin-bottom: 10px;

}

.kpi-value {

    font-size: 34px;

    font-weight: 800;

    color: white;

    line-height: 1;

}

.kpi-note {

    font-size: 13px;

    color: var(--muted);

    margin-top: 10px;

}

.status-card {

    background: linear-gradient(135deg, rgba(32,201,151,0.14), rgba(79,124,255,0.10));

    border: 1px solid var(--line);

    border-radius: 18px;

    padding: 14px 16px;

    color: #dffcf3;

    font-weight: 600;

}

.alert-high, .alert-medium, .alert-good {

    border-radius: 16px;

    padding: 14px 16px;

    border: 1px solid var(--line);

    margin-bottom: 10px;

    font-weight: 600;

}

.alert-high {

    background: rgba(255,107,107,0.12);

    color: #ffd6d6;

}

.alert-medium {

    background: rgba(245,166,35,0.12);

    color: #ffe6bc;

}

.alert-good {

    background: rgba(32,201,151,0.10);

    color: #d8fff2;

}

.dataframe-wrap {

    background: rgba(13,23,40,0.92);

    border: 1px solid var(--line);

    border-radius: 18px;

    padding: 10px;

}

[data-testid="stDataFrame"] {

    border-radius: 14px;

    overflow: hidden;

    border: 1px solid rgba(255,255,255,0.05);

}

.stTabs [data-baseweb="tab-list"] {

    gap: 10px;

}

.stTabs [data-baseweb="tab"] {

    background: rgba(255,255,255,0.04);

    color: white;

    border-radius: 12px;

    padding: 10px 16px;

}

.stTabs [aria-selected="true"] {

    background: rgba(79,124,255,0.16) !important;

    border: 1px solid rgba(79,124,255,0.35);

}

div[data-testid="stMetric"] {

    background: rgba(13,23,40,0.95);

    border: 1px solid var(--line);

    border-radius: 18px;

    padding: 10px;

}

.stSelectbox div[data-baseweb="select"] > div,

.stTextInput input,

.stNumberInput input,

.stFileUploader {

    background: rgba(255,255,255,0.03) !important;

    color: white !important;

    border-radius: 12px !important;

}

div[data-testid="stChatMessage"] {

    background: rgba(13,23,40,0.92);

    border: 1px solid var(--line);

    border-radius: 18px;

    padding: 8px;

}

hr {

    border-color: rgba(255,255,255,0.08);

}

</style>

""", unsafe_allow_html=True)

def safe_read(path):

    try:

        return pd.read_csv(path)

    except Exception:

        return pd.DataFrame()

def render_alerts(risk_df):

    if risk_df.empty:

        st.markdown('<div class="alert-medium">No data available to generate alerts.</div>', unsafe_allow_html=True)

        return

    high = risk_df[risk_df["risk_level"] == "HIGH"]

    med = risk_df[risk_df["risk_level"] == "MEDIUM"]

    if len(high) == 0 and len(med) == 0:

        st.markdown('<div class="alert-good">All monitored items are currently low risk.</div>', unsafe_allow_html=True)

        return

    for _, row in high.iterrows():

        st.markdown(

            f'<div class="alert-high">⚠️ High risk: <b>{row["name"]}</b> — unsold risk <b>{round(row["unsold_risk"],1)}</b> units.</div>',

            unsafe_allow_html=True

        )

    for _, row in med.iterrows():

        st.markdown(

            f'<div class="alert-medium">🟠 Medium risk: <b>{row["name"]}</b> should be monitored closely.</div>',

            unsafe_allow_html=True

        )

def copilot_answer(query, forecast_df, risk_df, rec_df):

    q = query.lower().strip()

    if forecast_df.empty or risk_df.empty:

        return "I don’t have enough store data yet. Upload sales, inventory, and product files first."

    if "run out" in q or "stock over" in q or "finish" in q:

        tmp = risk_df.copy()

        tmp["runout_days"] = tmp.apply(

            lambda r: round(r["stock_on_hand"] / r["pred_units"], 1) if r["pred_units"] > 0 else None,

            axis=1

        )

        tmp = tmp.sort_values("runout_days", na_position="last")

        rows = []

        for _, r in tmp.head(3).iterrows():

            if pd.notna(r["runout_days"]):

                rows.append(f"- {r['name']} may run out in about {r['runout_days']} days.")

        return "\n".join(rows) if rows else "No imminent stockout detected."

    if "order" in q or "reorder" in q:

        shortage = risk_df[risk_df["unsold_risk"] < 0]

        if shortage.empty:

            return "No urgent reorder recommendation right now."

        rows = []

        for _, r in shortage.iterrows():

            rows.append(f"- Reorder {r['name']} soon. Potential shortage: {abs(int(round(r['unsold_risk'])))} units.")

        return "\n".join(rows)

    if "risk" in q or "waste" in q:

        risky = risk_df[risk_df["risk_level"].isin(["HIGH", "MEDIUM"])]

        if risky.empty:

            return "No medium or high waste risks at the moment."

        rows = []

        for _, r in risky.iterrows():

            rows.append(f"- {r['name']}: {r['risk_level']} risk, unsold risk {round(r['unsold_risk'],1)} units.")

        return "\n".join(rows)

    top = forecast_df.sort_values("pred_units", ascending=False).head(3)

    top_names = ", ".join(top["sku"].tolist())

    return f"Top predicted demand items right now are: {top_names}. Ask me about stock, reorder, or risk."

# ---------------- DATA ----------------

sales_default = safe_read("data/sales_daily.csv")

inventory_default = safe_read("data/inventory_snapshot.csv")

products_default = safe_read("data/products.csv")

customers_default = safe_read("data/customers.csv")

purchases_default = safe_read("data/customer_purchases.csv")

# ---------------- HERO ----------------

st.markdown("""

<div class="hero">

    <div class="hero-title">🧠 Retail AI Copilot</div>

    <p class="hero-sub">

        Inventory intelligence, expiry risk detection, decision support, and demand activation —

        all in one premium control layer for retail operations.

    </p>

</div>

""", unsafe_allow_html=True)

left, right = st.columns([2.2, 1])

with left:

    st.markdown('<div class="section-title">Control Center</div>', unsafe_allow_html=True)

    st.markdown('<div class="small-muted">Choose a store and optionally upload fresh datasets to simulate a live software workflow.</div>', unsafe_allow_html=True)

with right:

    st.markdown('<div class="section-title">Live Status</div>', unsafe_allow_html=True)

    st.markdown('<div class="status-card">System online • AI engine active</div>', unsafe_allow_html=True)

tab_dashboard, tab_data, tab_copilot = st.tabs(["📊 Dashboard", "📂 Data Hub", "🤖 AI Copilot"])

# ---------------- DATA HUB ----------------

with tab_data:

    st.markdown("### Data Hub")

    st.markdown('<div class="small-muted">Upload your operational files to simulate a live retail software workflow.</div>', unsafe_allow_html=True)

    u1, u2, u3 = st.columns(3)

    with u1:

        sales_file = st.file_uploader("Upload Sales Data", type=["csv"])

    with u2:

        inventory_file = st.file_uploader("Upload Inventory Data", type=["csv"])

    with u3:

        products_file = st.file_uploader("Upload Products Data", type=["csv"])

    sales = pd.read_csv(sales_file) if sales_file else sales_default.copy()

    inventory = pd.read_csv(inventory_file) if inventory_file else inventory_default.copy()

    products = pd.read_csv(products_file) if products_file else products_default.copy()

    p1, p2, p3 = st.columns(3)

    with p1:

        st.markdown("#### Sales Preview")

        st.dataframe(sales.head(), use_container_width=True)

    with p2:

        st.markdown("#### Inventory Preview")

        st.dataframe(inventory.head(), use_container_width=True)

    with p3:

        st.markdown("#### Product Preview")

        st.dataframe(products.head(), use_container_width=True)

# use uploaded if present

sales = pd.read_csv(sales_file) if 'sales_file' in locals() and sales_file else sales_default.copy()

inventory = pd.read_csv(inventory_file) if 'inventory_file' in locals() and inventory_file else inventory_default.copy()

products = pd.read_csv(products_file) if 'products_file' in locals() and products_file else products_default.copy()

customers = customers_default.copy()

purchases = purchases_default.copy()

for df in [sales, inventory, products]:

    if "sku" in df.columns:

        df["sku"] = df["sku"].astype(str)

available_stores = sorted(sales["store_id"].dropna().unique().tolist()) if "store_id" in sales.columns else [1]

selected_store = st.selectbox("Store", available_stores, index=0)

forecast = forecast_next_day(sales, selected_store) if not sales.empty else pd.DataFrame()

inventory_store = inventory[inventory["store_id"] == selected_store].copy() if "store_id" in inventory.columns else pd.DataFrame()

if not forecast.empty and not inventory_store.empty and not products.empty:

    risk = compute_risk(forecast, inventory_store, products)

    recommend = recommend_actions(risk)

else:

    risk = pd.DataFrame()

    recommend = pd.DataFrame()

# ---------------- DASHBOARD ----------------

with tab_dashboard:

    st.markdown("### Executive Dashboard")

    total_skus = len(forecast) if not forecast.empty else 0

    high_risk = int((risk["risk_level"] == "HIGH").sum()) if not risk.empty else 0

    med_risk = int((risk["risk_level"] == "MEDIUM").sum()) if not risk.empty else 0

    total_stock = int(inventory_store["stock_on_hand"].sum()) if not inventory_store.empty and "stock_on_hand" in inventory_store.columns else 0

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.markdown(f'<div class="kpi"><div class="kpi-label">Tracked SKUs</div><div class="kpi-value">{total_skus}</div><div class="kpi-note">Active assortment monitored</div></div>', unsafe_allow_html=True)

    with c2:

        st.markdown(f'<div class="kpi"><div class="kpi-label">High Risk Items</div><div class="kpi-value">{high_risk}</div><div class="kpi-note">Immediate attention required</div></div>', unsafe_allow_html=True)

    with c3:

        st.markdown(f'<div class="kpi"><div class="kpi-label">Medium Risk Items</div><div class="kpi-value">{med_risk}</div><div class="kpi-note">Monitor and plan response</div></div>', unsafe_allow_html=True)

    with c4:

        st.markdown(f'<div class="kpi"><div class="kpi-label">Total Stock</div><div class="kpi-value">{total_stock}</div><div class="kpi-note">Units currently on hand</div></div>', unsafe_allow_html=True)

    st.markdown("### Smart Alerts")

    render_alerts(risk)

    if not forecast.empty:

        fig_bar = px.bar(

            forecast,

            x="sku",

            y="pred_units",

            title="Demand Forecast",

            color="pred_units",

            color_continuous_scale=["#4f7cff", "#7c5cff"]

        )

        fig_bar.update_layout(

            plot_bgcolor="rgba(0,0,0,0)",

            paper_bgcolor="rgba(0,0,0,0)",

            font_color="white",

            margin=dict(l=10, r=10, t=40, b=10)

        )

    else:

        fig_bar = None

    if not risk.empty:

        pie_df = risk["risk_level"].value_counts().reset_index()

        pie_df.columns = ["risk_level", "count"]

        fig_pie = px.pie(

            pie_df,

            names="risk_level",

            values="count",

            title="Risk Distribution",

            color="risk_level",

            color_discrete_map={

                "LOW": "#20c997",

                "MEDIUM": "#f5a623",

                "HIGH": "#ff6b6b"

            }

        )

        fig_pie.update_layout(

            plot_bgcolor="rgba(0,0,0,0)",

            paper_bgcolor="rgba(0,0,0,0)",

            font_color="white",

            margin=dict(l=10, r=10, t=40, b=10)

        )

    else:

        fig_pie = None

    g1, g2 = st.columns([1.2, 1])

    with g1:

        st.markdown('<div class="panel">', unsafe_allow_html=True)

        st.markdown("#### Demand Forecast")

        if fig_bar:

            st.plotly_chart(fig_bar, use_container_width=True)

        else:

            st.info("No forecast chart available.")

        st.markdown('</div>', unsafe_allow_html=True)

    with g2:

        st.markdown('<div class="panel">', unsafe_allow_html=True)

        st.markdown("#### Risk Distribution")

        if fig_pie:

            st.plotly_chart(fig_pie, use_container_width=True)

        else:

            st.info("No risk distribution available.")

        st.markdown('</div>', unsafe_allow_html=True)

    a1, a2 = st.columns([1.1, 1])

    with a1:

        st.markdown('<div class="panel">', unsafe_allow_html=True)

        st.markdown("#### Action Queue")

        if recommend.empty:

            st.info("No action queue available.")

        else:

            show_cols = [c for c in ["name", "risk_level", "unsold_risk", "action"] if c in recommend.columns]

            st.dataframe(recommend[show_cols], use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with a2:

        st.markdown('<div class="panel">', unsafe_allow_html=True)

        st.markdown("#### Risk Matrix")

        if risk.empty:

            st.info("No risk matrix available.")

        else:

            show_cols = [c for c in ["name", "stock_on_hand", "pred_units", "days_left", "risk_level"] if c in risk.columns]

            st.dataframe(risk[show_cols], use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

# ---------------- COPILOT ----------------

with tab_copilot:

    st.markdown("### AI Copilot")

    st.markdown('<div class="small-muted">Ask operational questions in natural language and get inventory-aware guidance.</div>', unsafe_allow_html=True)

    e1, e2, e3 = st.columns(3)

    with e1:

        st.markdown('<div class="panel"><b>Example</b><br>When will bread stock run out?</div>', unsafe_allow_html=True)

    with e2:

        st.markdown('<div class="panel"><b>Example</b><br>What should I reorder today?</div>', unsafe_allow_html=True)

    with e3:

        st.markdown('<div class="panel"><b>Example</b><br>Which products are risky?</div>', unsafe_allow_html=True)

    question = st.chat_input("Ask your copilot about stock, ordering, or waste...")

    if question:

        st.chat_message("user").write(question)

        st.chat_message("assistant").write(copilot_answer(question, forecast, risk, recommend))

    st.markdown("### Campaign Suggestions")

    try:

        campaigns = suggest_campaign(recommend, customers, purchases)

        if isinstance(campaigns, pd.DataFrame) and not campaigns.empty:

            st.dataframe(campaigns, use_container_width=True)

        else:

            st.info("No campaign suggestions generated right now.")

    except Exception:

        st.info("Campaign engine available, but no campaign output for this store yet.")

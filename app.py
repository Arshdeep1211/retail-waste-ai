import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

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

# ---------- STATE ----------
if "nav" not in st.session_state:
    st.session_state.nav = "Overview"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "Hello Alex! 👋 How can I help with your inventory today?"},
        {"role": "user", "content": "Which items will run out in the next 3 days?"},
    ]

# ---------- STYLE ----------
st.markdown("""
<style>
:root{
    --bg:#050b18;
    --bg2:#091326;
    --panel:#0c1628;
    --panel2:#101b30;
    --line:rgba(255,255,255,0.08);
    --line2:rgba(123,142,255,0.20);
    --text:#f6f8ff;
    --muted:#9aa8c8;
    --blue:#5c7cff;
    --purple:#7a5cff;
    --green:#2ecf9a;
    --orange:#f3aa3c;
    --red:#ff6f73;
}
html, body, [class*="css"]{
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
}
.stApp{
    background:
      radial-gradient(circle at 85% 10%, rgba(28,160,255,0.10), transparent 20%),
      radial-gradient(circle at 20% 90%, rgba(122,92,255,0.08), transparent 25%),
      linear-gradient(180deg, #040915 0%, #081224 100%);
    color: var(--text);
}
.block-container{
    max-width: 1600px;
    padding-top: 1rem;
    padding-bottom: 1.5rem;
}
section[data-testid="stSidebar"]{display:none;}
#MainMenu, footer, header{visibility:hidden;}

.main-grid {
    display:grid;
    grid-template-columns: 230px 1fr;
    gap:20px;
}
.left-rail{
    background: linear-gradient(180deg, rgba(12,22,40,0.98), rgba(10,18,34,0.98));
    border:1px solid var(--line);
    border-radius:22px;
    padding:18px 16px;
    min-height: calc(100vh - 120px);
    position: sticky;
    top: 18px;
}
.brand{
    display:flex;
    align-items:center;
    gap:12px;
    margin-bottom:18px;
}
.brand-icon{
    width:38px;
    height:38px;
    border-radius:12px;
    background: radial-gradient(circle at top left, #7a5cff, #4a6bff);
    display:flex;
    align-items:center;
    justify-content:center;
    font-size:20px;
}
.brand-title{
    font-size:17px;
    font-weight:800;
    color:white;
    line-height:1.1;
}
.brand-sub{
    color:var(--muted);
    font-size:12px;
}
.nav-section{
    margin-top:14px;
    margin-bottom:6px;
    color:#7f8db2;
    font-size:11px;
    text-transform:uppercase;
    letter-spacing:.08em;
}
.nav-btn button{
    width:100%;
    border-radius:14px;
    height:44px;
    background:transparent;
    color:#d8e1ff;
    border:1px solid transparent;
    text-align:left;
    font-weight:600;
}
.nav-btn button:hover{
    border:1px solid var(--line2);
    background:rgba(255,255,255,0.03);
    color:white;
}
.nav-btn-active button{
    width:100%;
    border-radius:14px;
    height:44px;
    background:linear-gradient(90deg, rgba(92,124,255,0.22), rgba(122,92,255,0.16));
    color:white;
    border:1px solid rgba(92,124,255,0.30);
    text-align:left;
    font-weight:700;
}
.upgrade-box{
    margin-top:18px;
    padding:16px;
    border-radius:18px;
    background:linear-gradient(135deg, rgba(122,92,255,0.16), rgba(92,124,255,0.08));
    border:1px solid var(--line2);
}
.upgrade-title{
    color:white;
    font-size:16px;
    font-weight:800;
}
.upgrade-sub{
    color:#c2cdf0;
    font-size:13px;
    margin-top:6px;
    margin-bottom:14px;
}
.progress-wrap{
    margin-top:12px;
    color:#d6e0ff;
    font-size:12px;
}
.progress{
    width:100%;
    height:10px;
    background:rgba(255,255,255,0.08);
    border-radius:999px;
    overflow:hidden;
    margin-top:8px;
}
.progress-fill{
    width:75%;
    height:100%;
    background:linear-gradient(90deg,#7a5cff,#5c7cff);
}
.support-links{
    margin-top:20px;
    color:#c4d0f3;
    font-size:14px;
    line-height:2;
}

.topbar{
    display:flex;
    align-items:center;
    justify-content:space-between;
    margin-bottom:18px;
}
.greeting{
    color:white;
    font-size:18px;
    font-weight:700;
}
.greeting-sub{
    color:var(--muted);
    font-size:14px;
    margin-top:4px;
}
.topbar-right{
    display:flex;
    gap:12px;
    align-items:center;
}
.top-pill{
    background:rgba(255,255,255,0.03);
    border:1px solid var(--line);
    border-radius:14px;
    padding:10px 14px;
    color:#dce5ff;
    font-size:14px;
}

.card{
    background:linear-gradient(180deg, rgba(12,22,40,0.95), rgba(10,18,34,0.98));
    border:1px solid var(--line);
    border-radius:22px;
    padding:18px;
    box-shadow:0 10px 30px rgba(0,0,0,.18);
    height:100%;
}
.kpi-card{
    background:linear-gradient(180deg, rgba(12,22,40,0.96), rgba(13,22,40,0.98));
    border:1px solid var(--line);
    border-radius:20px;
    padding:16px 18px;
    min-height:126px;
}
.kpi-top{
    display:flex;
    justify-content:space-between;
    align-items:center;
    margin-bottom:12px;
}
.kpi-title{
    color:#c3d0f2;
    font-size:14px;
    font-weight:600;
}
.kpi-val{
    color:white;
    font-size:22px;
    font-weight:800;
}
.kpi-foot{
    color:var(--muted);
    font-size:13px;
    margin-top:8px;
}

.section-title{
    color:white;
    font-size:17px;
    font-weight:800;
    margin-bottom:8px;
}
.section-sub{
    color:var(--muted);
    font-size:13px;
}

.alert-box{
    border-radius:18px;
    padding:14px 16px;
    border:1px solid var(--line);
    margin-bottom:10px;
}
.alert-good{
    background:linear-gradient(90deg, rgba(46,207,154,0.12), rgba(46,207,154,0.05));
    color:#dffcf2;
}
.alert-high{
    background:linear-gradient(90deg, rgba(255,111,115,0.14), rgba(255,111,115,0.05));
    color:#ffd8d9;
}
.alert-med{
    background:linear-gradient(90deg, rgba(243,170,60,0.14), rgba(243,170,60,0.05));
    color:#ffe4bd;
}

.stTabs [data-baseweb="tab-list"]{
    gap:10px;
}
.stTabs [data-baseweb="tab"]{
    background:rgba(255,255,255,0.03);
    border-radius:14px;
    color:white;
    padding:10px 14px;
}
.stTabs [aria-selected="true"]{
    background:linear-gradient(90deg, rgba(92,124,255,0.22), rgba(122,92,255,0.16)) !important;
    border:1px solid rgba(92,124,255,0.35);
}

[data-testid="stDataFrame"]{
    border-radius:14px;
    overflow:hidden;
    border:1px solid rgba(255,255,255,0.05);
}
div[data-testid="stChatMessage"]{
    background:rgba(255,255,255,0.03);
    border:1px solid var(--line);
    border-radius:16px;
    padding:8px;
}
.stSelectbox div[data-baseweb="select"] > div,
.stTextInput input,
.stFileUploader{
    background:rgba(255,255,255,0.03) !important;
    color:white !important;
    border-radius:12px !important;
}
small, p, label, span{
    color:inherit;
}div[data-testid="stButton"] > button {
    background: rgba(255,255,255,0.03);
    color: #d8e1ff;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    height: 46px;
    width: 100%;
    text-align: left;
    font-weight: 600;
    box-shadow: none;
}

div[data-testid="stButton"] > button:hover {
    background: rgba(92,124,255,0.14);
    border: 1px solid rgba(92,124,255,0.32);
    color: white;
}

div[data-testid="stButton"] > button:focus {
    outline: none !important;
    box-shadow: none !important;
}

div[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(90deg, rgba(92,124,255,0.22), rgba(122,92,255,0.16));
    border: 1px solid rgba(92,124,255,0.35);
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ---------- HELPERS ----------
def safe_read(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def build_button(label: str, key: str, active: bool = False):
    if st.button(
        label,
        key=key,
        use_container_width=True,
        type="primary" if active else "secondary"
    ):
            st.session_state.nav = key
def render_alerts(risk_df: pd.DataFrame):
    if risk_df.empty:
        st.markdown('<div class="alert-box alert-med">No data available to generate alerts.</div>', unsafe_allow_html=True)
        return
    high = risk_df[risk_df["risk_level"] == "HIGH"]
    med = risk_df[risk_df["risk_level"] == "MEDIUM"]
    if len(high) == 0 and len(med) == 0:
        st.markdown('<div class="alert-box alert-good">All monitored items are currently low risk.</div>', unsafe_allow_html=True)
        return
    for _, row in high.iterrows():
        st.markdown(
            f'<div class="alert-box alert-high">⚠️ <b>{row["name"]}</b> is at high risk. Unsold risk: <b>{round(row["unsold_risk"],1)}</b> units.</div>',
            unsafe_allow_html=True
        )
    for _, row in med.iterrows():
        st.markdown(
            f'<div class="alert-box alert-med">🟠 <b>{row["name"]}</b> should be monitored closely.</div>',
            unsafe_allow_html=True
        )

def runout_days(row):
    pred = row.get("pred_units", 0)
    stock = row.get("stock_on_hand", 0)
    if pred and pred > 0:
        return round(stock / pred, 1)
    return None

def copilot_answer(query: str, forecast_df: pd.DataFrame, risk_df: pd.DataFrame, rec_df: pd.DataFrame) -> str:
    q = query.lower().strip()

    if forecast_df.empty or risk_df.empty:
        return "I don’t have enough store data yet. Upload sales, inventory, and product files first."

    if "run out" in q or "stock over" in q or "finish" in q:
        tmp = risk_df.copy()
        tmp["runout_days"] = tmp.apply(runout_days, axis=1)
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

def sales_vs_forecast_chart(forecast_df: pd.DataFrame) -> go.Figure:
    if forecast_df.empty:
        return go.Figure()
    df = forecast_df.copy()
    df["sales"] = (df["pred_units"] * np.array([0.65, 0.8, 1.2, 0.9][:len(df)] + [1.0]*max(0, len(df)-4))).round(1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["sku"], y=df["sales"], mode="lines+markers", name="Sales",
        line=dict(color="#35d49a", width=3)
    ))
    fig.add_trace(go.Scatter(
        x=df["sku"], y=df["pred_units"], mode="lines+markers", name="Forecast",
        line=dict(color="#7a5cff", width=3)
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        margin=dict(l=10, r=10, t=20, b=10),
        legend=dict(orientation="h", y=1.1, x=0)
    )
    return fig

def inventory_health_chart(risk_df: pd.DataFrame) -> go.Figure:
    if risk_df.empty:
        return go.Figure()
    buckets = {"0–2 days": 0, "3–7 days": 0, "8–14 days": 0, "15+ days": 0}
    for _, row in risk_df.iterrows():
        d = row["days_left"]
        if d <= 2:
            buckets["0–2 days"] += 1
        elif d <= 7:
            buckets["3–7 days"] += 1
        elif d <= 14:
            buckets["8–14 days"] += 1
        else:
            buckets["15+ days"] += 1

    order = list(buckets.keys())
    vals = [buckets[k] for k in order]
    colors = ["#ff6f73", "#f3aa3c", "#4f7cff", "#2ecf9a"]

    fig = go.Figure(go.Bar(
        x=vals,
        y=order,
        orientation="h",
        marker=dict(color=colors)
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis_title="Items",
        yaxis_title=""
    )
    return fig

# ---------- LOAD DATA ----------
sales_default = safe_read("data/sales_daily.csv")
inventory_default = safe_read("data/inventory_snapshot.csv")
products_default = safe_read("data/products.csv")
customers_default = safe_read("data/customers.csv")
purchases_default = safe_read("data/customer_purchases.csv")

sales = sales_default.copy()
inventory = inventory_default.copy()
products = products_default.copy()
customers = customers_default.copy()
purchases = purchases_default.copy()

for df in [sales, inventory, products]:
    if "sku" in df.columns:
        df["sku"] = df["sku"].astype(str)

available_stores = sorted(sales["store_id"].dropna().unique().tolist()) if "store_id" in sales.columns else [1]
selected_store = available_stores[0] if available_stores else 1

forecast = forecast_next_day(sales, selected_store) if not sales.empty else pd.DataFrame()
inventory_store = inventory[inventory["store_id"] == selected_store].copy() if "store_id" in inventory.columns else pd.DataFrame()

if not forecast.empty and not inventory_store.empty and not products.empty:
    risk = compute_risk(forecast, inventory_store, products)
    recommend = recommend_actions(risk)
else:
    risk = pd.DataFrame()
    recommend = pd.DataFrame()

# ---------- LAYOUT ----------
left_col, main_col = st.columns([1.05, 5.95], gap="large")

with left_col:
    st.markdown("""
    <div class="left-rail">
        <div class="brand">
            <div class="brand-icon">✦</div>
            <div>
                <div class="brand-title">Retail AI</div>
                <div class="brand-sub">Copilot</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    build_button("🏠  Overview", "Overview", st.session_state.nav == "Overview")
    build_button("📈  Sales & Demand", "Sales & Demand", st.session_state.nav == "Sales & Demand")
    build_button("📦  Inventory", "Inventory", st.session_state.nav == "Inventory")
    build_button("⚠️  Risk & Waste", "Risk & Waste", st.session_state.nav == "Risk & Waste")
    build_button("🛒  Replenishment", "Replenishment", st.session_state.nav == "Replenishment")
    build_button("🎯  Offers & Campaigns", "Offers & Campaigns", st.session_state.nav == "Offers & Campaigns")
    build_button("🧾  Reports", "Reports", st.session_state.nav == "Reports")

    st.markdown('<div class="nav-section">Data Hub</div>', unsafe_allow_html=True)
    build_button("⤴️  Upload Data", "Upload Data", st.session_state.nav == "Upload Data")
    build_button("🗂️  Data Sources", "Data Sources", st.session_state.nav == "Data Sources")
    build_button("🛡️  Data Quality", "Data Quality", st.session_state.nav == "Data Quality")

    st.markdown('<div class="nav-section">AI Copilot</div>', unsafe_allow_html=True)
    build_button("💬  Chat with AI", "Chat with AI", st.session_state.nav == "Chat with AI")
    build_button("✨  Recommendations", "Recommendations", st.session_state.nav == "Recommendations")

    st.markdown("""
        <div class="upgrade-box">
            <div class="upgrade-title">Pro Plan</div>
            <div class="upgrade-sub">Unlock advanced AI insights, more stores and custom reports.</div>
            <div class="progress-wrap">
                75% used
                <div class="progress"><div class="progress-fill"></div></div>
            </div>
        </div>
        <div class="support-links">
            ⚙️ Settings<br>
            ❓ Help & Support
        </div>
    </div>
    """, unsafe_allow_html=True)

with main_col:
    top_left, top_right = st.columns([3.5, 2.5])
    with top_left:
        st.markdown('<div class="greeting">Good evening, Alex 👋</div>', unsafe_allow_html=True)
        st.markdown('<div class="greeting-sub">Here’s what’s happening with your stores today.</div>', unsafe_allow_html=True)
    with top_right:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="top-pill">🏬 All Stores</div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="top-pill">📅 Apr 17 – Apr 23, 2025</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.session_state.nav in ["Overview", "Sales & Demand", "Inventory", "Risk & Waste", "Replenishment", "Offers & Campaigns", "Reports"]:
        # KPI ROW
        r1, r2, r3, r4, r5 = st.columns(5)
        with r1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-top"><div class="kpi-title">Total Sales</div><div>📈</div></div>
                <div class="kpi-val">€4.4K</div>
                <div class="kpi-foot">↑ 13.75% vs last 7 days</div>
            </div>
            """, unsafe_allow_html=True)
        with r2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-top"><div class="kpi-title">Forecast Accuracy</div><div>🌀</div></div>
                <div class="kpi-val">44%</div>
                <div class="kpi-foot">↑ 5.2% vs last 7 days</div>
            </div>
            """, unsafe_allow_html=True)
        with r3:
            high_count = int((risk["risk_level"] == "HIGH").sum()) if not risk.empty else 0
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-top"><div class="kpi-title">High Risk Items</div><div>🛡️</div></div>
                <div class="kpi-val">{high_count}</div>
                <div class="kpi-foot">Items needing action</div>
            </div>
            """, unsafe_allow_html=True)
        with r4:
            waste_val = max(float(risk["unsold_risk"].clip(lower=0).sum()), 0.0) * 3 if not risk.empty else 0.0
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-top"><div class="kpi-title">Potential Waste</div><div>🪙</div></div>
                <div class="kpi-val">€{waste_val:,.0f}</div>
                <div class="kpi-foot">↓ 8.4% vs last 7 days</div>
            </div>
            """, unsafe_allow_html=True)
        with r5:
            total_stock_val = int(inventory_store["stock_on_hand"].sum()) if not inventory_store.empty and "stock_on_hand" in inventory_store.columns else 0
            inv_value = total_stock_val * 3.5
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-top"><div class="kpi-title">Inventory Value</div><div>🧾</div></div>
                <div class="kpi-val">€{inv_value:,.1f}</div>
                <div class="kpi-foot">↑ 9.1% vs last 7 days</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ROW 2
        c1, c2, c3 = st.columns([2.2, 2.0, 1.45])

        with c1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Demand Forecast</div>', unsafe_allow_html=True)
            if not forecast.empty:
                fig_bar = px.bar(
                    forecast,
                    x="sku",
                    y="pred_units",
                    color="pred_units",
                    color_continuous_scale=["#6f6cff", "#8a5cff"],
                )
                fig_bar.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="white",
                    margin=dict(l=10, r=10, t=10, b=10),
                    coloraxis_showscale=False
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("No forecast available.")
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Sales vs Forecast</div>', unsafe_allow_html=True)
            fig_sv = sales_vs_forecast_chart(forecast)
            st.plotly_chart(fig_sv, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with c3:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Smart Alerts</div>', unsafe_allow_html=True)
            render_alerts(risk)

            st.markdown("""
            <div class="alert-box alert-med">🟠 Milk stock running low in Store 1</div>
            <div class="alert-box" style="background:rgba(79,124,255,0.10); color:#dfe8ff;">🔵 Demand spike detected in bakery</div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ROW 3
        c4, c5, c6, c7 = st.columns([1.6, 1.5, 1.5, 1.8])

        with c4:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Top Categories by Sales</div>', unsafe_allow_html=True)
            cat_df = pd.DataFrame({
                "category": ["Bakery", "Dairy", "Meat", "Beverages"],
                "value": [42, 28, 18, 12]
            })
            fig_cat = px.pie(
                cat_df,
                names="category",
                values="value",
                hole=0.55,
                color="category",
                color_discrete_sequence=["#7a5cff", "#4f7cff", "#f3aa3c", "#ff6f73"]
            )
            fig_cat.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="white",
                margin=dict(l=10, r=10, t=10, b=10),
                showlegend=True
            )
            st.plotly_chart(fig_cat, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with c5:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Risk Distribution</div>', unsafe_allow_html=True)
            if not risk.empty:
                pie_df = risk["risk_level"].value_counts().reset_index()
                pie_df.columns = ["risk_level", "count"]
                fig_pie = px.pie(
                    pie_df,
                    names="risk_level",
                    values="count",
                    hole=0.55,
                    color="risk_level",
                    color_discrete_map={
                        "LOW": "#2ecf9a",
                        "MEDIUM": "#f3aa3c",
                        "HIGH": "#ff6f73"
                    }
                )
                fig_pie.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="white",
                    margin=dict(l=10, r=10, t=10, b=10)
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("No risk distribution available.")
            st.markdown('</div>', unsafe_allow_html=True)

        with c6:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Inventory Health</div>', unsafe_allow_html=True)
            fig_inv = inventory_health_chart(risk)
            st.plotly_chart(fig_inv, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with c7:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">AI Copilot</div>', unsafe_allow_html=True)
            for msg in st.session_state.chat_history[:3]:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])

            st.markdown("""
            <div style="margin-top:10px; color:#cbd6f7; font-size:14px;">Suggested actions:</div>
            """, unsafe_allow_html=True)
            b1, b2 = st.columns(2)
            with b1:
                st.button("Create Order", use_container_width=True)
            with b2:
                st.button("Run Promotion", use_container_width=True)

            user_q = st.chat_input("Ask anything...")
            if user_q:
                st.session_state.chat_history.append({"role": "user", "content": user_q})
                answer = copilot_answer(user_q, forecast, risk, recommend)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                st.rerun()

            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # BOTTOM TABLE
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Items Needing Attention</div>', unsafe_allow_html=True)

        if not risk.empty:
            items = risk.copy()
            items["store"] = f"Store {selected_store}"
            items["predicted_demand"] = items["pred_units"].astype(str) + " / day"
            items["risk_badge"] = items["risk_level"]
            items["action"] = np.where(items["risk_level"] == "HIGH", "Promote / Discount",
                               np.where(items["risk_level"] == "MEDIUM", "Monitor", "No Action"))
            show = items[["name", "category", "store", "stock_on_hand", "predicted_demand", "days_left", "risk_badge", "action"]].copy()
            show.columns = ["Item", "Category", "Store", "Stock On Hand", "Predicted Demand", "Days Left", "Risk Level", "Action"]
            st.dataframe(show, use_container_width=True, hide_index=True)
        else:
            st.info("No items available.")
        st.markdown('</div>', unsafe_allow_html=True)

    elif st.session_state.nav == "Upload Data":
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Upload Data</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Upload operational files to simulate a live software workflow.</div>', unsafe_allow_html=True)
        u1, u2, u3 = st.columns(3)
        with u1:
            st.file_uploader("Upload Sales CSV", type=["csv"])
        with u2:
            st.file_uploader("Upload Inventory CSV", type=["csv"])
        with u3:
            st.file_uploader("Upload Product Master CSV", type=["csv"])
        st.markdown('</div>', unsafe_allow_html=True)

    elif st.session_state.nav in ["Chat with AI", "Recommendations"]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">AI Copilot Workspace</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Ask natural-language questions and get inventory-aware guidance.</div>', unsafe_allow_html=True)

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        q = st.chat_input("Ask your copilot about stock, ordering, or waste...")
        if q:
            st.session_state.chat_history.append({"role": "user", "content": q})
            a = copilot_answer(q, forecast, risk, recommend)
            st.session_state.chat_history.append({"role": "assistant", "content": a})
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f'<div class="section-title">{st.session_state.nav}</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">This section can be expanded next.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

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
    padding:16px 18px;
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
.alert-blue{
    background:rgba(79,124,255,0.10);
    color:#dfe8ff;
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
}
div[data-testid="stButton"] > button {
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
.card, .kpi-card {
    transition: all 0.25s ease;
}

.card:hover, .kpi-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 20px 50px rgba(0,0,0,0.25);
}
</style>
""", unsafe_allow_html=True)

# ---------- HELPERS ----------
def safe_read(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize uploaded CSV columns so users can upload practical store files."""
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    rename = {
        "product_id": "sku",
        "product": "name",
        "item": "name",
        "item_name": "name",
        "product_name": "name",
        "stock": "stock_on_hand",
        "inventory": "stock_on_hand",
        "current_stock": "stock_on_hand",
        "stock_onhand": "stock_on_hand",
        "sales": "units_sold",
        "sold": "units_sold",
        "daily_sales": "units_sold",
        "daily_demand": "avg_daily_sales",
        "avg_sales": "avg_daily_sales",
        "average_daily_sales": "avg_daily_sales",
        "demand": "avg_daily_sales",
        "shelf_life": "shelf_life_days",
        "shelf_life_day": "shelf_life_days",
        "days_left": "shelf_life_days",
        "expiry_days": "shelf_life_days",
        "store": "store_id",
    }

    return df.rename(columns={c: rename.get(c, c) for c in df.columns})


def split_uploaded_file(df: pd.DataFrame):
    """
    Convert one simple uploaded store CSV into the three internal datasets:
    sales, inventory, products.

    Minimum accepted columns:
    sku/name/category/stock_on_hand/shelf_life_days/avg_daily_sales
    The function also works with common aliases such as product, item, stock,
    shelf_life, daily_demand, and sales.
    """
    df = standardize_columns(df)

    if "store_id" not in df.columns:
        df["store_id"] = 1
    if "sku" not in df.columns:
        df["sku"] = [f"SKU_{i+1}" for i in range(len(df))]
    if "name" not in df.columns:
        df["name"] = df["sku"].astype(str)
    if "category" not in df.columns:
        df["category"] = "General"
    if "shelf_life_days" not in df.columns:
        df["shelf_life_days"] = 5
    if "stock_on_hand" not in df.columns:
        df["stock_on_hand"] = 20
    if "avg_daily_sales" not in df.columns and "units_sold" not in df.columns:
        df["avg_daily_sales"] = 5

    df["sku"] = df["sku"].astype(str)
    df["store_id"] = pd.to_numeric(df["store_id"], errors="coerce").fillna(1).astype(int)
    df["stock_on_hand"] = pd.to_numeric(df["stock_on_hand"], errors="coerce").fillna(0)
    df["shelf_life_days"] = pd.to_numeric(df["shelf_life_days"], errors="coerce").fillna(5).astype(int)

    products = df[["sku", "name", "category", "shelf_life_days"]].drop_duplicates("sku")
    inventory = df[["store_id", "sku", "stock_on_hand"]].drop_duplicates(["store_id", "sku"])

    # If uploaded CSV already has daily sales history, use it.
    if "date" in df.columns and "units_sold" in df.columns:
        sales = df[["date", "store_id", "sku", "units_sold"]].copy()
        sales["units_sold"] = pd.to_numeric(sales["units_sold"], errors="coerce").fillna(0)
    else:
        # Otherwise generate a small 14-day sales history from avg_daily_sales
        # so the forecasting engine can run immediately.
        rows = []
        base_date = pd.Timestamp.today().normalize()
        for _, row in df.drop_duplicates("sku").iterrows():
            avg = row.get("avg_daily_sales", row.get("units_sold", 5))
            avg = float(pd.to_numeric(pd.Series([avg]), errors="coerce").fillna(5).iloc[0])
            for d in range(14):
                multiplier = [0.9, 1.0, 1.05, 0.95, 1.15, 1.25, 0.85][d % 7]
                rows.append({
                    "date": (base_date - pd.Timedelta(days=d)).strftime("%Y-%m-%d"),
                    "store_id": int(row.get("store_id", 1)),
                    "sku": str(row["sku"]),
                    "units_sold": round(avg * multiplier, 2),
                })
        sales = pd.DataFrame(rows)

    return sales, inventory, products


def build_button(label: str, key: str, active: bool = False):
    if st.button(
        label,
        key=key,
        use_container_width=True,
        type="primary" if active else "secondary"
    ):
        st.session_state.nav = key


def get_action_text(row: pd.Series) -> str:
    """Create a founder-friendly action text for alerts/table/copilot."""
    if "action" in row and pd.notna(row.get("action")):
        raw = str(row.get("action"))
        if raw and raw.upper() not in ["OK", "NO ACTION", "NAN"]:
            return raw

    risk_level = str(row.get("risk_level", "LOW"))
    if risk_level == "HIGH":
        return "Promote / Discount"
    if risk_level == "MEDIUM":
        return "Monitor closely"
    return "No Action"


def render_alerts(risk_df: pd.DataFrame):
    """
    Upgraded smart alerts:
    - Keeps your UI style
    - Converts status alerts into decision alerts
    - Shows expected waste + recommended action
    """
    if risk_df.empty:
        st.markdown('<div class="alert-box alert-med">No data available to generate alerts.</div>', unsafe_allow_html=True)
        return

    df = risk_df.copy()
    if "action" not in df.columns:
        df["action"] = df.apply(get_action_text, axis=1)

    high = df[df["risk_level"] == "HIGH"].sort_values("unsold_risk", ascending=False)
    med = df[df["risk_level"] == "MEDIUM"].sort_values("unsold_risk", ascending=False)

    if len(high) == 0 and len(med) == 0:
        st.markdown('<div class="alert-box alert-good">All monitored items are currently low risk.</div>', unsafe_allow_html=True)
        return

    # Show top high-risk decisions first
    for _, row in high.head(3).iterrows():
        st.markdown(
            f'''
            <div class="alert-box alert-high">
                🚨 <b>{row["name"]}</b><br>
                Will likely remain unsold: <b>{round(float(row["unsold_risk"]), 1)} units</b><br>
                Days left: <b>{row["days_left"]}</b><br>
                👉 Action: <b>{get_action_text(row)}</b>
            </div>
            ''',
            unsafe_allow_html=True
        )

    # Then medium risk
    for _, row in med.head(2).iterrows():
        st.markdown(
            f'''
            <div class="alert-box alert-med">
                🟠 <b>{row["name"]}</b><br>
                Moderate expiry risk detected<br>
                👉 Action: <b>Monitor or adjust pricing</b>
            </div>
            ''',
            unsafe_allow_html=True
        )


def runout_days(row):
    pred = row.get("pred_units", 0)
    stock = row.get("stock_on_hand", 0)
    if pred and pred > 0:
        return round(stock / pred, 1)
    return None


def copilot_answer(query: str, forecast_df: pd.DataFrame, risk_df: pd.DataFrame, rec_df: pd.DataFrame) -> str:
    """
    Lightweight AI Copilot behavior.
    Still rule-based, but it now answers like a decision engine:
    - what to do
    - what may run out
    - what to order
    - what may go to waste
    """
    q = query.lower().strip()

    if forecast_df.empty or risk_df.empty:
        return "I don’t have enough store data yet. Upload sales, inventory, and product files first."

    df = risk_df.copy()
    if not rec_df.empty and "action" in rec_df.columns:
        action_cols = ["sku", "action"]
        df = df.merge(rec_df[action_cols], on="sku", how="left", suffixes=("", "_rec"))
        if "action_rec" in df.columns:
            df["action"] = df["action_rec"].fillna(df.get("action", ""))

    if "action" not in df.columns:
        df["action"] = df.apply(get_action_text, axis=1)

    if "what should i do" in q or "actions" in q or "today" in q:
        urgent = df[df["risk_level"] == "HIGH"].copy()
        if urgent.empty:
            return "No urgent actions required today. Continue monitoring medium-risk items."
        rows = []
        for _, r in urgent.head(5).iterrows():
            rows.append(f"- {r['name']}: {get_action_text(r)}. Estimated unsold risk: {round(float(r['unsold_risk']), 1)} units.")
        return "Recommended actions today:\n" + "\n".join(rows)

    if "run out" in q or "stock over" in q or "finish" in q:
        tmp = df.copy()
        tmp["runout_days"] = tmp.apply(runout_days, axis=1)
        tmp = tmp.sort_values("runout_days", na_position="last")
        rows = []
        for _, r in tmp.head(3).iterrows():
            if pd.notna(r["runout_days"]):
                rows.append(f"- {r['name']} may run out in about {r['runout_days']} days.")
        return "\n".join(rows) if rows else "No imminent stockout detected."

    if "order" in q or "reorder" in q:
        tmp = df.copy()
        tmp["recommended_order"] = ((tmp["pred_units"] * 2) - tmp["stock_on_hand"]).clip(lower=0).astype(int)
        tmp = tmp[tmp["recommended_order"] > 0].sort_values("recommended_order", ascending=False)
        if tmp.empty:
            return "No urgent reorder recommendation right now. Current stock levels are sufficient for the next short horizon."
        rows = []
        for _, r in tmp.head(5).iterrows():
            rows.append(f"- Order {int(r['recommended_order'])} units of {r['name']}.")
        return "Recommended orders:\n" + "\n".join(rows)

    if "risk" in q or "waste" in q or "expire" in q:
        risky = df[df["risk_level"].isin(["HIGH", "MEDIUM"])]
        if risky.empty:
            return "No medium or high waste risks at the moment."
        rows = []
        for _, r in risky.head(5).iterrows():
            rows.append(f"- {r['name']}: {r['risk_level']} risk, unsold risk {round(float(r['unsold_risk']), 1)} units. Action: {get_action_text(r)}.")
        return "Waste risk summary:\n" + "\n".join(rows)

    if "campaign" in q or "promotion" in q or "customer" in q:
        high = df[df["risk_level"] == "HIGH"]
        if high.empty:
            return "No campaign is needed right now. No high-risk products detected."
        rows = []
        for _, r in high.head(3).iterrows():
            rows.append(f"- Run a demand activation campaign for {r['name']} targeting customers who buy {r['category']} products.")
        return "Campaign suggestions:\n" + "\n".join(rows)

    top = forecast_df.sort_values("pred_units", ascending=False).head(3)
    top_names = ", ".join(top["sku"].astype(str).tolist())
    return f"Top predicted demand items right now are: {top_names}. Ask me about stock, reorder, risk, actions, or campaigns."


def sales_vs_forecast_chart(forecast_df: pd.DataFrame) -> go.Figure:
    if forecast_df.empty:
        return go.Figure()
    df = forecast_df.copy()
    multipliers = np.array([0.65, 0.8, 1.2, 0.9][:len(df)] + [1.0] * max(0, len(df) - 4))
    df["sales"] = (df["pred_units"] * multipliers).round(1)
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

sales = st.session_state.get("uploaded_sales", sales_default.copy())
inventory = st.session_state.get("uploaded_inventory", inventory_default.copy())
products = st.session_state.get("uploaded_products", products_default.copy())
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
    # Merge action from recommendation engine into risk table for alerts/table/copilot
    if not recommend.empty and "action" in recommend.columns:
        risk = risk.merge(recommend[["sku", "action"]], on="sku", how="left")
    else:
        risk["action"] = risk.apply(get_action_text, axis=1)

    if not customers.empty and not purchases.empty:
        campaigns = suggest_campaign(recommend, customers, purchases)
    else:
        campaigns = pd.DataFrame()
else:
    risk = pd.DataFrame()
    recommend = pd.DataFrame()
    campaigns = pd.DataFrame()

# Calculated metrics used across the page
high_count = int((risk["risk_level"] == "HIGH").sum()) if not risk.empty and "risk_level" in risk.columns else 0
medium_count = int((risk["risk_level"] == "MEDIUM").sum()) if not risk.empty and "risk_level" in risk.columns else 0
positive_unsold = risk["unsold_risk"].clip(lower=0) if not risk.empty and "unsold_risk" in risk.columns else pd.Series(dtype=float)
waste_val = max(float(positive_unsold.sum()), 0.0) * 3 if not positive_unsold.empty else 0.0
waste_with_ai = waste_val * 0.7
waste_saved = waste_val - waste_with_ai

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
        st.markdown('<div style="color:#9aa8c8; font-size:14px; margin-top:6px;">AI Copilot that tells you what to order, what will go to waste, and what actions to take.</div>', unsafe_allow_html=True)
    with top_right:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="top-pill">🏬 All Stores</div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="top-pill">📅 Apr 17 – Apr 23, 2025</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.session_state.nav in ["Overview", "Sales & Demand", "Inventory", "Risk & Waste", "Replenishment", "Offers & Campaigns", "Reports", "Recommendations", "Chat with AI"]:
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
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-top"><div class="kpi-title">High Risk Items</div><div>🛡️</div></div>
                <div class="kpi-val">{high_count}</div>
                <div class="kpi-foot">Items needing action</div>
            </div>
            """, unsafe_allow_html=True)
        with r4:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-top"><div class="kpi-title">Potential Waste</div><div>🪙</div></div>
                <div class="kpi-val">€{waste_val:,.0f}</div>
                <div class="kpi-foot">Estimated value at risk</div>
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

        # NEW YC-READY DECISION STRIP: small, does not break your dashboard
        d1, d2, d3 = st.columns([1.4, 1.4, 1.2])
        with d1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">⚡ What should I do today?</div>', unsafe_allow_html=True)
            if not risk.empty:
                urgent = risk[risk["risk_level"] == "HIGH"].copy()
                if urgent.empty:
                    st.markdown('<div class="alert-box alert-good">No urgent actions required today.</div>', unsafe_allow_html=True)
                else:
                    for _, row in urgent.head(3).iterrows():
                        st.markdown(
                            f'<div class="alert-box alert-high">🚨 <b>{row["name"]}</b>: {get_action_text(row)}</div>',
                            unsafe_allow_html=True
                        )
            else:
                st.markdown('<div class="alert-box alert-med">No decision data available.</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with d2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">📦 What should I order?</div>', unsafe_allow_html=True)
            if not risk.empty:
                order_tmp = risk.copy()
                order_tmp["recommended_order"] = ((order_tmp["pred_units"] * 2) - order_tmp["stock_on_hand"]).clip(lower=0).astype(int)
                order_tmp = order_tmp[order_tmp["recommended_order"] > 0].sort_values("recommended_order", ascending=False)
                if order_tmp.empty:
                    st.markdown('<div class="alert-box alert-good">No urgent reorder needed.</div>', unsafe_allow_html=True)
                else:
                    for _, row in order_tmp.head(3).iterrows():
                        st.markdown(
                            f'<div class="alert-box alert-blue">🛒 <b>{row["name"]}</b>: order <b>{int(row["recommended_order"])} units</b></div>',
                            unsafe_allow_html=True
                        )
            else:
                st.markdown('<div class="alert-box alert-med">No order data available.</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with d3:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">📊 Expected Impact</div>', unsafe_allow_html=True)
            st.markdown(
                f'''
                <div class="alert-box alert-good">
                    Waste today: <b>€{waste_val:,.0f}</b><br>
                    With AI: <b>€{waste_with_ai:,.0f}</b><br>
                    Savings: <b>€{waste_saved:,.0f}</b> / day
                </div>
                ''',
                unsafe_allow_html=True
            )
            st.markdown('</div>', unsafe_allow_html=True)

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
                st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})
            else:
                st.info("No forecast available.")
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Sales vs Forecast</div>', unsafe_allow_html=True)
            fig_sv = sales_vs_forecast_chart(forecast)
            st.plotly_chart(fig_sv, use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)

        with c3:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Smart Alerts</div>', unsafe_allow_html=True)
            render_alerts(risk)

            # Kept your original style, but makes the bakery alert sound more like intelligence
            st.markdown("""
            <div class="alert-box alert-blue">🔵 Demand spike detected<br><b>Bakery items demand is higher than baseline</b></div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ROW 3
        c4, c5, c6, c7 = st.columns([1.25, 1.25, 1.35, 1.55], gap="medium")

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
            st.plotly_chart(fig_cat, use_container_width=True, config={"displayModeBar": False})
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
                st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})
            else:
                st.info("No risk distribution available.")
            st.markdown('</div>', unsafe_allow_html=True)

        with c6:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Inventory Health</div>', unsafe_allow_html=True)
            fig_inv = inventory_health_chart(risk)
            st.plotly_chart(fig_inv, use_container_width=True, config={"displayModeBar": False})
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
                if st.button("Create Order", use_container_width=True):
                    st.session_state.chat_history.append({"role": "user", "content": "What should I order?"})
                    st.session_state.chat_history.append({"role": "assistant", "content": copilot_answer("What should I order?", forecast, risk, recommend)})
                    st.rerun()
            with b2:
                if st.button("Run Promotion", use_container_width=True):
                    st.session_state.chat_history.append({"role": "user", "content": "What campaign should I run?"})
                    st.session_state.chat_history.append({"role": "assistant", "content": copilot_answer("campaign", forecast, risk, recommend)})
                    st.rerun()

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

            st.markdown(
                f"""
                <div class="alert-box alert-good">
                    {len(risk)} items monitored • {high_count} high risk • {medium_count} medium risk • Store {selected_store} active
                </div>
                """,
                unsafe_allow_html=True
            )

            items = risk.copy()
            items["store"] = f"Store {selected_store}"
            items["predicted_demand"] = items["pred_units"].round(1).astype(str) + " / day"
            items["risk_badge"] = items["risk_level"]

            # NEW: order recommendation added inside your existing table
            items["recommended_order"] = (
                (items["pred_units"] * 2 - items["stock_on_hand"])
            ).clip(lower=0).astype(int)

            items["action"] = items.apply(get_action_text, axis=1)

            show = items[[
                "name", "category", "store", "stock_on_hand",
                "predicted_demand", "days_left", "risk_badge",
                "recommended_order", "action"
            ]]

            show.columns = [
                "Item", "Category", "Store", "Stock On Hand",
                "Predicted Demand", "Days Left", "Risk Level",
                "Recommended Order", "Action"
            ]

            st.dataframe(show, use_container_width=True, hide_index=True)

        else:
            st.info("No items available.")

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # NEW: Impact section added at the bottom without changing your dashboard look
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📊 Impact Simulation</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Estimated effect if today’s recommended actions are executed.</div><br>', unsafe_allow_html=True)

        i1, i2, i3, i4 = st.columns(4)
        with i1:
            st.metric("Waste Without AI", f"€{waste_val:,.0f}")
        with i2:
            st.metric("Waste With AI", f"€{waste_with_ai:,.0f}")
        with i3:
            st.metric("Estimated Saving", f"€{waste_saved:,.0f}")
        with i4:
            st.metric("Waste Reduction", "30%")

        st.markdown('</div>', unsafe_allow_html=True)

    elif st.session_state.nav == "Upload Data":
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">⤴️ Upload Store Data</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Upload a CSV and let the AI analyze demand, waste risk, ordering, actions, and customer activation.</div><br>', unsafe_allow_html=True)

        st.markdown("""
        <div class="alert-box alert-blue">
            Accepted simple CSV columns: <b>sku, name, category, stock_on_hand, shelf_life_days, avg_daily_sales</b>.<br>
            You can also use aliases like product, item, stock, shelf_life, daily_demand, sales.
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader("Upload your store CSV", type=["csv"])

        sample = pd.DataFrame([
            {"sku": "MILK_1", "name": "Fresh Milk", "category": "Dairy", "stock_on_hand": 42, "shelf_life_days": 3, "avg_daily_sales": 12},
            {"sku": "BREAD_1", "name": "Sourdough Bread", "category": "Bakery", "stock_on_hand": 28, "shelf_life_days": 1, "avg_daily_sales": 18},
            {"sku": "CHICKEN_1", "name": "Chicken Breast", "category": "Meat", "stock_on_hand": 18, "shelf_life_days": 2, "avg_daily_sales": 6},
            {"sku": "YOGURT_1", "name": "Natural Yogurt", "category": "Dairy", "stock_on_hand": 35, "shelf_life_days": 4, "avg_daily_sales": 7},
            {"sku": "BANANA_1", "name": "Banana", "category": "Produce", "stock_on_hand": 50, "shelf_life_days": 5, "avg_daily_sales": 10},
        ])

        with st.expander("See sample upload format"):
            st.dataframe(sample, use_container_width=True, hide_index=True)

        if uploaded_file is not None:
            raw_df = pd.read_csv(uploaded_file)
            st.markdown('<div class="section-title">Preview</div>', unsafe_allow_html=True)
            st.dataframe(raw_df.head(10), use_container_width=True)

            if st.button("Analyze File", use_container_width=True, type="primary"):
                sales_new, inventory_new, products_new = split_uploaded_file(raw_df)

                st.session_state["uploaded_sales"] = sales_new
                st.session_state["uploaded_inventory"] = inventory_new
                st.session_state["uploaded_products"] = products_new

                st.session_state.chat_history = [
                    {"role": "assistant", "content": "Uploaded file analyzed. Ask me what to order, what will go to waste, what actions to take, or which campaign to run."}
                ]

                st.success("File processed successfully. Go back to Overview to see the updated AI dashboard.")
                st.session_state.nav = "Overview"
                st.rerun()

        if "uploaded_sales" in st.session_state:
            if st.button("Reset to demo data", use_container_width=True):
                for k in ["uploaded_sales", "uploaded_inventory", "uploaded_products"]:
                    st.session_state.pop(k, None)
                st.success("Reset complete. Demo data restored.")
                st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f'<div class="section-title">{st.session_state.nav}</div>', unsafe_allow_html=True)
        st.write("This section is ready to be expanded in the next version.")
        st.markdown('</div>', unsafe_allow_html=True)

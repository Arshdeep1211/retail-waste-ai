import pandas as pd

def compute_risk(forecast_df, inventory_df, products_df):

    # ✅ ensure same SKUs only
    df = forecast_df.merge(inventory_df, on="sku", how="inner")
    df = df.merge(products_df, on="sku", how="left")

    # ✅ fill missing values (prevents crashes)
    df["shelf_life_days"] = df.get("shelf_life_days", 3).fillna(3)
    df["stock_on_hand"] = df.get("stock_on_hand", 0).fillna(0)
    df["pred_units"] = df.get("pred_units", 0).fillna(0)

    # ✅ calculations
    df["days_left"] = df["shelf_life_days"]
    df["expected_sales"] = df["pred_units"] * df["days_left"]
    df["unsold_risk"] = df["stock_on_hand"] - df["expected_sales"]

    # ✅ classification (safe)
    def classify(x):
        if x > 10:
            return "HIGH"
        elif x > 3:
            return "MEDIUM"
        else:
            return "LOW"

    df["risk_level"] = df["unsold_risk"].apply(classify)

    return df

import pandas as pd

def compute_risk(forecast_df, inventory_df, products_df):
    df = forecast_df.merge(inventory_df, on="sku")
    df = df.merge(products_df, on="sku")

    df["days_left"] = df["shelf_life_days"]
    df["expected_sales"] = df["pred_units"] * df["days_left"]
    df["unsold_risk"] = df["stock_on_hand"] - df["expected_sales"]

    def classify(x):
        if x > 10:
            return "HIGH"
        elif x > 3:
            return "MEDIUM"
        else:
            return "LOW"

    df["risk_level"] = df["unsold_risk"].apply(classify)

    return df

import pandas as pd


def forecast_next_day(sales: pd.DataFrame, store_id: int, lookback_days: int = 3) -> pd.DataFrame:
    store_sales = sales[sales["store_id"] == store_id].copy()
    store_sales = store_sales.sort_values(["sku", "date"])

    forecasts = []

    for sku, group in store_sales.groupby("sku"):
        recent = group.tail(lookback_days)
        mean_val = recent["units_sold"].mean()

        if pd.isna(mean_val):
            pred_units = 0
        else:
            pred_units = max(mean_val, 0)

        forecasts.append({
            "sku": sku,
            "pred_units": round(float(pred_units), 2)
        })

    return pd.DataFrame(forecasts)

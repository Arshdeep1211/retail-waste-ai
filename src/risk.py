import pandas as pd


def expiry_risk(inv_latest: pd.DataFrame, pred: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    merged = inv_latest.merge(pred, on="sku", how="left")
    merged = merged.merge(products[["sku", "name", "category", "shelf_life_days_default"]], on="sku", how="left")
    merged["pred_units"] = merged["pred_units"].fillna(0)

    def score_row(row):
        on_hand = row["on_hand_units"]
        pred_units = row["pred_units"]

        if pred_units <= 0 and on_hand > 0:
            return 95, "Do not reorder / review stock"
        if on_hand >= pred_units * 3 and on_hand > 0:
            return 85, "High overstock risk"
        if on_hand >= pred_units * 2 and on_hand > 0:
            return 65, "Moderate overstock risk"
        if on_hand < pred_units:
            return 25, "Low expiry risk"
        return 40, "Monitor"

    scored = merged.apply(lambda row: score_row(row), axis=1, result_type="expand")
    merged["risk_score"] = scored[0]
    merged["recommended_action"] = scored[1]

    return merged[[
        "sku",
        "name",
        "category",
        "on_hand_units",
        "pred_units",
        "risk_score",
        "recommended_action"
    ]].sort_values("risk_score", ascending=False)

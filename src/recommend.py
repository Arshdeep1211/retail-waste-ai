import math
import pandas as pd


def recommend_order(pred: pd.DataFrame, inv_latest: pd.DataFrame, products: pd.DataFrame, safety_factor: float = 0.2) -> pd.DataFrame:
    merged = pred.merge(inv_latest[["sku", "on_hand_units"]], on="sku", how="left")
    merged = merged.merge(products[["sku", "name", "category", "case_pack"]], on="sku", how="left")

    merged["on_hand_units"] = merged["on_hand_units"].fillna(0)
    merged["target_stock"] = merged["pred_units"] * (1 + safety_factor)
    merged["raw_order_units"] = (merged["target_stock"] - merged["on_hand_units"]).clip(lower=0)

    def round_to_case_pack(row):
        case_pack = row["case_pack"] if pd.notna(row["case_pack"]) and row["case_pack"] > 0 else 1
        raw_order = row["raw_order_units"]
        return int(math.ceil(raw_order / case_pack) * case_pack) if raw_order > 0 else 0

    merged["order_units"] = merged.apply(round_to_case_pack, axis=1)

    def confidence_label(pred_units):
        if pred_units >= 20:
            return "High"
        if pred_units >= 10:
            return "Medium"
        return "Low"

    merged["confidence"] = merged["pred_units"].apply(confidence_label)

    return merged[[
        "sku",
        "name",
        "category",
        "pred_units",
        "on_hand_units",
        "order_units",
        "confidence"
    ]]

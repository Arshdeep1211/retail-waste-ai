import pandas as pd

def score_customers(customers, purchases, sku_category):
    scores = []

    for _, c in customers.iterrows():
        customer_id = c["customer_id"]

        past = purchases[purchases["customer_id"] == customer_id]

        affinity = (past["category"] == sku_category).mean()
        frequency = len(past)
        discount = c["discount_buyer_flag"]

        score = 0.4*affinity + 0.3*frequency + 0.3*discount

        scores.append({
            "customer_id": customer_id,
            "score": score
        })

    return pd.DataFrame(scores).sort_values("score", ascending=False)


def suggest_campaign(risk_df, customers, purchases):
    campaigns = []

    for _, row in risk_df.iterrows():
        if "PUSH DEMAND" in row["action"]:
            top_customers = score_customers(
                customers,
                purchases,
                row["category"]
            ).head(5)

            campaigns.append({
                "sku": row["sku"],
                "target_customers": list(top_customers["customer_id"]),
                "message": f"{row['sku']} on discount today!"
            })

    return pd.DataFrame(campaigns)

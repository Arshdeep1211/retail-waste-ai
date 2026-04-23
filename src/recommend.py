def recommend_actions(risk_df):
    actions = []

    for _, row in risk_df.iterrows():
        if row["risk_level"] == "HIGH":
            action = "MARKDOWN 15% + PUSH DEMAND"
        elif row["risk_level"] == "MEDIUM":
            action = "MARKDOWN 10%"
        else:
            action = "NO ACTION"

        actions.append(action)

    risk_df["action"] = actions
    return risk_df

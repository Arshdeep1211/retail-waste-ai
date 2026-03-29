# Retail Waste AI

AI-driven retail operations MVP for:
- demand forecasting
- order recommendation
- overstock / expiry risk alerts

## MVP Scope
This first version focuses on:
- reading CSV sales and inventory data
- forecasting next-day demand
- recommending order quantities
- flagging overstock risk

## Run
```bash
pip install -r requirements.txt
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
---

## Step 6: Add demo data

### `data/stores.csv`

```csv
store_id,name,postcode
1,Store Munich Center,80331
2,Store Munich East,81667
3,Store Munich North,80939

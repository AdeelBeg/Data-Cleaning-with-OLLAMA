import pandas as pd
from src.common.schema import RetailSchema
from src.traditional_cleaning.pipeline import clean_traditional

def test_traditional_pipeline_smoke():
    s = RetailSchema()
    df = pd.DataFrame({
        s.transaction_id: ["TXN_1"],
        s.customer_id: ["CUST_1"],
        s.category: [" food "],
        s.item: ["Item_1_FOOD"],
        s.price_per_unit: ["10"],
        s.quantity: ["2.0"],
        s.total_spent: ["20.0"],
        s.payment_method: ["credit card"],
        s.location: ["in-store"],
        s.transaction_date: ["2024-01-01"],
        s.discount_applied: ["TRUE"],
    })
    out = clean_traditional(df, s)
    assert out[s.category].iloc[0] == "Food"

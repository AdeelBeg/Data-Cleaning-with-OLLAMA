import pandas as pd
from src.common.schema import RetailSchema
from src.common.metrics import quality_report

def test_quality_report_has_score():
    s = RetailSchema()
    df = pd.DataFrame({s.transaction_id:["TXN_1"], s.customer_id:["CUST_1"], s.price_per_unit:[1], s.quantity:[1], s.total_spent:[1], s.transaction_date:["2024-01-01"]})
    qr = quality_report(df, s)
    assert "quality_score" in qr

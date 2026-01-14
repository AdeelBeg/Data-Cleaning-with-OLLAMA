# This test does not call the OpenAI API.
# It only checks that the module imports and that guardrails helper runs.
import pandas as pd
from src.common.schema import RetailSchema
from src.genai_cleaning.guardrails import accept_if_quality_not_worse

def test_guardrails_accepts_equal_quality():
    s = RetailSchema()
    df = pd.DataFrame({s.transaction_id:["TXN_1"], s.customer_id:["CUST_1"], s.price_per_unit:[1], s.quantity:[1], s.total_spent:[1], s.transaction_date:["2024-01-01"]})
    assert accept_if_quality_not_worse(df, df.copy(), s) is True

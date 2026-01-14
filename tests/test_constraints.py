import pandas as pd
from src.common.schema import RetailSchema
from src.common.constraints import run_all_constraints

def test_constraints_run():
    s = RetailSchema()
    df = pd.DataFrame({
        s.transaction_id: ["TXN_1", "TXN_2"],
        s.customer_id: ["CUST_1", "CUST_2"],
        s.category: ["Food", "Food"],
        s.item: ["Item_1_FOOD", "Item_2_FOOD"],
        s.price_per_unit: [10, 5],
        s.quantity: [2, 3],
        s.total_spent: [20, 15],
        s.payment_method: ["Credit Card", "Cash"],
        s.location: ["Online", "In-store"],
        s.transaction_date: ["2024-01-01", "2024-01-02"],
        s.discount_applied: ["TRUE", "FALSE"],
    })
    res = run_all_constraints(df, s)
    assert len(res) >= 1

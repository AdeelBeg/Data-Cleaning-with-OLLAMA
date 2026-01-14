from __future__ import annotations
from dataclasses import dataclass
from typing import List

@dataclass(frozen=True)
class RetailSchema:
    transaction_id: str = "Transaction ID"
    customer_id: str = "Customer ID"
    category: str = "Category"
    item: str = "Item"
    price_per_unit: str = "Price Per Unit"
    quantity: str = "Quantity"
    total_spent: str = "Total Spent"
    payment_method: str = "Payment Method"
    location: str = "Location"
    transaction_date: str = "Transaction Date"
    discount_applied: str = "Discount Applied"

    def all_columns(self) -> List[str]:
        return [
            self.transaction_id, self.customer_id, self.category, self.item,
            self.price_per_unit, self.quantity, self.total_spent,
            self.payment_method, self.location, self.transaction_date,
            self.discount_applied,
        ]

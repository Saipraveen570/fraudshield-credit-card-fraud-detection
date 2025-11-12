from pydantic import BaseModel
from typing import Optional

class Transaction(BaseModel):
    transaction_id: Optional[int] = None
    customer_id: int
    merchant_id: int
    amount: float
    country: str
    mcc: str
    channel: str
    device_id: str
    hour: Optional[int] = None
    dayofweek: Optional[int] = None
    tx_count_1h: Optional[float] = None
    tx_count_24h: Optional[float] = None
    avg_amount_7d: Optional[float] = None
    std_amount_7d: Optional[float] = None
    device_new: Optional[int] = None
    is_cross_border: Optional[int] = None
    merchant_risk: Optional[float] = None

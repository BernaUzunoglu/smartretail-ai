from pydantic import BaseModel

class OrderHabitRequest(BaseModel):
    total_spend: float
    order_count: int
    avg_order_size: float

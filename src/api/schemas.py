from pydantic import BaseModel
from typing import Dict
from typing import Optional

class OrderHabitRequest(BaseModel):
    total_spend: float
    order_count: int
    avg_order_size: float
class OrderRequest(BaseModel):
    customer_id: str
    order_id: int

class SHAPExplanation(BaseModel):
    base_value: float
    feature_contributions: Dict[str, float]

class OrderPredictionResponse(BaseModel):
    high_return_risk: bool
    probability: float
    explanation: Optional[SHAPExplanation] = None # Optional olarak ekledik

class PurchasePredictionRequest(BaseModel):
    customer_id: str
    category_name: str

class TopCategoriesRequest(BaseModel):
    customer_id: str
    top_n: int = 3

class CategoryPrediction(BaseModel):
    category_name: str
    probability: float




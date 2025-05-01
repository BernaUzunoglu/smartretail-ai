from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Tuple
from ...services.product_purchase_service import ProductPurchaseService

router = APIRouter()
service = ProductPurchaseService()

class PurchasePredictionRequest(BaseModel):
    customer_id: str
    category_name: str

class TopCategoriesRequest(BaseModel):
    customer_id: str
    top_n: int = 3

class CategoryPrediction(BaseModel):
    category_name: str
    probability: float

@router.post("/predict", response_model=float)
async def predict_purchase_potential(request: PurchasePredictionRequest):
    """
    Belirli bir müşterinin belirli bir kategorideki ürünleri satın alma potansiyelini tahmin eder.
    """
    prediction = service.predict_purchase_potential(
        request.customer_id,
        request.category_name
    )
    
    if prediction is None:
        raise HTTPException(status_code=400, detail="Tahmin yapılamadı")
    
    return prediction

@router.post("/top-categories", response_model=List[CategoryPrediction])
async def get_top_categories(request: TopCategoriesRequest):
    """
    Bir müşteri için en yüksek satın alma potansiyeline sahip kategorileri döndürür.
    """
    predictions = service.get_top_categories(
        request.customer_id,
        request.top_n
    )
    
    if not predictions:
        raise HTTPException(status_code=400, detail="Kategori tahminleri yapılamadı")
    
    return [
        CategoryPrediction(
            category_name=category,
            probability=prob
        )
        for category, prob in predictions
    ] 
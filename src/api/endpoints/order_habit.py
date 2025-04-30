# api/endpoints/order_habit.py
from fastapi import APIRouter
from src.api.schemas import OrderHabitRequest
from src.services.order_habit_service import OrderHabitService

router = APIRouter()

@router.post("/predict-order-habit")
def predict_order_habit(data: OrderHabitRequest):
    service = OrderHabitService()
    result = service.predict(data.dict())
    return {
        "prediction_probability": result,
        "will_reorder": result >= 0.5
    }

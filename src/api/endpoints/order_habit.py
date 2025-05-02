# api/endpoints/order_habit.py
from fastapi import APIRouter
from src.api.schemas import OrderHabitRequest
from src.services.order_habit_service import OrderHabitService

router = APIRouter()

@router.get("/predict-order-habit-by-id/{customer_id}")
def predict_order_habit_by_id(customer_id: str):
    service = OrderHabitService()
    try:
        result = service.predict_by_customer_id(customer_id)
        return result
    except ValueError as ve:
        return {"error": str(ve)}
    except Exception as e:
        import traceback
        traceback.print_exc()  # terminalde detaylı hata gösterimi
        return {"error": "Beklenmeyen bir hata oluştu."}


from fastapi import FastAPI, HTTPException
from api.schemas import OrderRequest, OrderPredictionResponse
from src.services.return_risk_service import ReturnRiskService
from fastapi import APIRouter

app = FastAPI()
service = ReturnRiskService()
router = APIRouter()


@router.post("/predict-return-risk", response_model=OrderPredictionResponse)
def predict_return_risk(order: OrderRequest):
    try:
        result = service.predict(order.customer_id, order.order_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")


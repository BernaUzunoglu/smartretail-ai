# api/app.py
from fastapi import FastAPI
from src.api.endpoints import order_habit, return_risk, product_purchase


app = FastAPI(title="Order Prediction API")

app.include_router(order_habit.router, prefix="/order-habit", tags=["Order Habit"])
app.include_router(return_risk.router, prefix="/predict-return-risk", tags=["Return Risk"])
app.include_router(product_purchase.router)

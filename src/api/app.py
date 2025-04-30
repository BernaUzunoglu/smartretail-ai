# api/app.py
from fastapi import FastAPI
from src.api.endpoints import order_habit

app = FastAPI(title="Order Prediction API")

app.include_router(order_habit.router, prefix="/order-habit", tags=["Order Habit"])

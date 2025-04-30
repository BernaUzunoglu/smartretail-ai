import joblib
import pandas as pd
from tensorflow.keras.models import load_model
from core.config import Config

class OrderHabitService:
    def __init__(self):
        self.model = load_model(Config.PROJECT_ROOT / "models/order_habit/model.h5")
        self.preprocessor = joblib.load(Config.PROJECT_ROOT / "models/order_habit/preprocessor.pkl")

    def predict(self, features: dict) -> float:
        df = pd.DataFrame([features])
        X = self.preprocessor.transform(df)
        prediction = self.model.predict(X)
        return float(prediction[0][0])

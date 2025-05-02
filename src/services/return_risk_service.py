import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from data.fetch_orders import get_customer_orders_date
from core.config import Config
import shap

class ReturnRiskService:
    def __init__(self):
        self.model = load_model(Config.PROJECT_ROOT / 'src/models/return_risk/model.h5')
        self.scaler = StandardScaler()
        self.explainer = None
        self.background_initialized = False
        self.feature_names = ['discount', 'quantity', 'unit_price', 'total_spend']

    def _initialize_shap(self, X_sample):
        """SHAP kernel explainer'ı başlat."""
        if not self.background_initialized:
            self.explainer = shap.KernelExplainer(self.model.predict, X_sample[:100])
            self.background_initialized = True

    def prepare_data(self, order_df: pd.DataFrame) -> np.ndarray:
        features = order_df[self.feature_names]
        return self.scaler.fit_transform(features)

    def explain_prediction(self, scaled_input: np.ndarray) -> dict:
        self._initialize_shap(scaled_input)

        shap_values = self.explainer.shap_values(scaled_input[:1])
        base_value = self.explainer.expected_value[0]
        contribution_dict = dict(zip(self.feature_names, shap_values[0][0]))

        return {
            "base_value": float(base_value),  # tam basamaklı
            "feature_contributions": {k: float(v) for k, v in contribution_dict.items()}
        }

    def predict(self, customer_id: str, order_id: int) -> dict:
        order_df = get_customer_orders_date(customer_id, order_id)

        if order_df.empty:
            raise ValueError("Order not found or invalid customer/order ID.")

        input_data = self.prepare_data(order_df)
        probability = self.model.predict(input_data)[0][0]
        label = probability > 0.5

        explanation = self.explain_prediction(input_data)

        return {
            "high_return_risk": bool(label),
            "probability": round(float(probability), 4),
            "explanation": explanation
        }

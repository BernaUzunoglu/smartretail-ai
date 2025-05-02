import joblib
import pandas as pd
from tensorflow.keras.models import load_model
from data.fetch_customer_order_summary import get_customer_order_data
from core.config import Config

class OrderHabitService:
    def __init__(self):
        self.model = load_model(Config.PROJECT_ROOT / "src/models/order_habit/model.h5")
        self.preprocessor = joblib.load(Config.PROJECT_ROOT / "src/models/order_habit/preprocessor.pkl")
        self.feature_names = self.preprocessor.feature_names_in_

    def predict_by_customer_id(self, customer_id: str) -> dict:
        df = get_customer_order_data()
        df['order_date'] = pd.to_datetime(df['order_date'])
        df['order_month'] = df['order_date'].dt.month
        df['order_weekday'] = df['order_date'].dt.weekday

        def get_season(month):
            return ('Winter' if month in [12, 1, 2] else
                    'Spring' if month in [3, 4, 5] else
                    'Summer' if month in [6, 7, 8] else 'Fall')
        df['order_season'] = df['order_month'].apply(get_season)

        customer_df = df[df['customer_id'].str.strip() == customer_id.strip()]
        if customer_df.empty:
            raise ValueError("Customer not found")

        customer_df['total'] = customer_df['unit_price'] * customer_df['quantity']
        total_spend = customer_df['total'].sum()
        order_count = customer_df['order_id'].nunique()
        avg_order_size = total_spend / order_count if order_count else 0
        customer_lifetime = (customer_df['order_date'].max() - customer_df['order_date'].min()).days
        gap_days = customer_df.sort_values('order_date')['order_date'].diff().dt.days
        avg_days_between_orders = gap_days.mean() if not gap_days.empty else 0
        orders_per_month = order_count / (customer_lifetime / 30) if customer_lifetime > 0 else 0
        common_month = customer_df['order_month'].mode()[0]

        seasons = ['Fall', 'Spring', 'Summer']
        season_features = {f'order_season_{s}': 0 for s in seasons}
        most_common_season = customer_df['order_season'].mode()[0]
        if most_common_season in seasons:
            season_features[f'order_season_{most_common_season}'] = 1

        features = {
            'order_count': order_count,
            'avg_order_size': avg_order_size,
            'customer_lifetime': customer_lifetime,
            'avg_days_between_orders': avg_days_between_orders,
            'orders_per_month': orders_per_month,
            'common_order_month': common_month,
            **season_features
        }

        prediction = self.predict(features)

        return {
            "prediction_probability": round(prediction, 4),
            "will_reorder": prediction >= 0.5
        }

    def predict(self, features: dict) -> float:
        df = pd.DataFrame([features])
        df = df.reindex(columns=self.feature_names, fill_value=0)  # eksik kolon varsa 0'la
        X = self.preprocessor.transform(df)
        prediction = self.model.predict(X)
        return float(prediction[0][0])

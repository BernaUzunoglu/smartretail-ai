import os
import joblib
import tensorflow as tf
import numpy as np

class ProductPurchaseService:
    def __init__(self):
        self.model = None
        self.user_encoder = None
        self.category_encoder = None
        self.load_model()
    
    def load_model(self):
        model_dir = os.path.join(os.path.dirname(__file__), '../models/product_purchase_potential')
        
        # Model ve encoder'ları yükleme
        self.model = tf.keras.models.load_model(f'{model_dir}/model.h5')
        self.user_encoder = joblib.load(f'{model_dir}/user_encoder.joblib')
        self.category_encoder = joblib.load(f'{model_dir}/category_encoder.joblib')
    
    def predict_purchase_potential(self, customer_id, category_name):
        """
        Belirli bir müşterinin belirli bir kategorideki ürünleri satın alma potansiyelini tahmin eder.
        
        Args:
            customer_id: Müşteri ID'si
            category_name: Kategori adı
            
        Returns:
            float: Satın alma olasılığı (0-1 arası)
        """
        try:
            # Müşteri ve kategori ID'lerini dönüştürme
            user_id = self.user_encoder.transform([customer_id])[0]
            category_id = self.category_encoder.transform([category_name])[0]
            
            # Tahmin
            prediction = self.model.predict([
                np.array([user_id]),
                np.array([category_id])
            ])
            
            return float(prediction[0][0])
        except Exception as e:
            print(f"Tahmin hatası: {str(e)}")
            return None
    
    def get_top_categories(self, customer_id, top_n=3):
        """
        Bir müşteri için en yüksek satın alma potansiyeline sahip kategorileri döndürür.
        
        Args:
            customer_id: Müşteri ID'si
            top_n: Döndürülecek kategori sayısı
            
        Returns:
            list: (kategori_adı, olasılık) tuple'larının listesi
        """
        try:
            user_id = self.user_encoder.transform([customer_id])[0]
            predictions = []
            
            for category in self.category_encoder.classes_:
                category_id = self.category_encoder.transform([category])[0]
                pred = self.model.predict([
                    np.array([user_id]),
                    np.array([category_id])
                ])
                predictions.append((category, float(pred[0][0])))
            
            # Olasılığa göre sıralama
            predictions.sort(key=lambda x: x[1], reverse=True)
            return predictions[:top_n]
        except Exception as e:
            print(f"Kategori tahmin hatası: {str(e)}")
            return [] 
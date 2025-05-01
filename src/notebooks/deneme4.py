# Gerekli kütüphaneler
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import seaborn as sns
from data.fetch_customer_order_summary import get_customer_order_data
from core.config import Config
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.combine import SMOTETomek
from tensorflow.keras.layers import Dropout
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import random
import tensorflow as tf


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
# 1. Veriyi çek
df = get_customer_order_data()
df.head()

# Feature engineering
df['order_date'] = pd.to_datetime(df['order_date'])
# order_date datetime'e çevrili
df['order_month'] = df['order_date'].dt.month
df['order_weekday'] = df['order_date'].dt.weekday

# Mevsim çıkarımı
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

df['order_season'] = df['order_month'].apply(get_season)

# Müşterinin ilk sipariş tarihi
first_order = df.groupby('customer_id')['order_date'].min().reset_index()
first_order.columns = ['customer_id', 'first_order_date']

# Müşteri ömrü (kaç gün aktif)
df = df.merge(first_order, on='customer_id')
df['customer_lifetime'] = (df['order_date'].max() - df['first_order_date']).dt.days

# Siparişler arası ortalama gün farkı (müşteri bazlı)
df_sorted = df.sort_values(['customer_id', 'order_date'])
df_sorted['gap_days'] = df_sorted.groupby('customer_id')['order_date'].diff().dt.days
avg_order_gap = df_sorted.groupby('customer_id')['gap_days'].mean().reset_index()
avg_order_gap.columns = ['customer_id', 'avg_days_between_orders']

# Ana agg veri seti
df['total'] = df['unit_price'] * df['quantity']

agg = df.groupby('customer_id').agg(
    total_spend=('total', 'sum'),
    order_count=('order_id', 'nunique'),
    last_order_date=('order_date', 'max'),
    customer_lifetime=('customer_lifetime', 'max')
).reset_index()
agg['avg_order_size'] = agg['total_spend'] / agg['order_count']

# Etiketleme: Son 6 ay içinde sipariş verdiyse 1
cutoff_date = agg['last_order_date'].max() - pd.DateOffset(months=6)
agg['label'] = (agg['last_order_date'] > cutoff_date).astype(int)

# Ort. gün farkı ekle
agg = agg.merge(avg_order_gap, on='customer_id', how='left')

# Son olarak date kolonunu at
agg.drop(columns=['last_order_date'], inplace=True)

agg['orders_per_month'] = agg['order_count'] / (agg['customer_lifetime'] / 30)
agg['spend_per_day'] = agg['total_spend'] / agg['customer_lifetime']


X = agg[['total_spend', 'order_count', 'avg_order_size',
         'customer_lifetime', 'avg_days_between_orders',
         'orders_per_month', 'spend_per_day']]

y = agg['label']

#  Verimiz scale edelim
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

joblib.dump(scaler, Config.PROJECT_ROOT / 'src/models/order_habit/preprocessor.pkl')  # preprocessoru kaydet

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, test_size=0.3, random_state=SEED)

# SMOTE parametrelerini özelleştirme
smote = SMOTE(k_neighbors=1, sampling_strategy=0.5, random_state=SEED)

# SMOTETomek'e SMOTE parametrelerini iletme
smt = SMOTETomek(smote=smote, random_state=SEED)
# smote = SMOTE(k_neighbors=1, sampling_strategy=0.5)
X_resampled, y_resampled = smt.fit_resample(X_train, y_train)

# 2. class_weight hesapla (SMOTE sonrası etiketlere göre)
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
weights = dict(zip(np.unique(y_train), class_weights))

#  Model Eğitimi
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), class_weight=weights, epochs=20, batch_size=16)

model.save(Config.PROJECT_ROOT / 'src/models/order_habit/model.h5')

# Eğitim sonucu görselleştirme
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()


# Test veri kümesi üzerinden tahmin yap


y_pred = model.predict(X_test)

fpr, tpr, thresholds = roc_curve(y_test, y_pred)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

y_pred_labels = (y_pred > optimal_threshold).astype(int)

auc_score = roc_auc_score(y_test, y_pred)
print(f"AUC Score: {auc_score:.4f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_labels)

# Görselleştirme
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Detaylı skorlar
print(classification_report(y_test, y_pred_labels))


# Gerekli kütüphaneler
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import seaborn as sns
from data.fetch_customer_order_summary import get_customer_order_data
from core.config import Config
from sklearn.metrics import confusion_matrix, classification_report
import random
import tensorflow as tf

# Sabit tohum
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# 1. Veriyi çek
df = get_customer_order_data()

# Feature engineering
df['order_date'] = pd.to_datetime(df['order_date'])
df['total'] = df['unit_price'] * df['quantity']

agg = df.groupby('customer_id').agg(
    total_spend=('total', 'sum'),
    order_count=('order_id', 'nunique'),
    last_order_date=('order_date', 'max')
).reset_index()

agg['avg_order_size'] = agg['total_spend'] / agg['order_count']
cutoff_date = agg['last_order_date'].max() - pd.DateOffset(months=6)
agg['recency_days'] = (cutoff_date - agg['last_order_date']).dt.days
agg['label'] = (agg['last_order_date'] > cutoff_date).astype(int)
agg.drop(columns=['last_order_date'], inplace=True)

X = agg[['total_spend', 'order_count', 'avg_order_size']]
y = agg['label']

# Ölçekleme
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, Config.PROJECT_ROOT / 'src/models/order_habit/preprocessor.pkl')

# Eğitim/Test ayırma
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, test_size=0.2, random_state=SEED)

# Model tanımı
model = Sequential([
    Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Eğitim (class_weight yok)
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=16)

# Modeli kaydet
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

# Test tahmini
y_pred = model.predict(X_test)
y_pred_labels = (y_pred > 0.5).astype(int)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_labels)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Performans skoru
print(classification_report(y_test, y_pred_labels))

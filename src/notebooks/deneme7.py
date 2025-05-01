# Gerekli kütüphaneler
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns
from data.fetch_customer_order_summary import get_customer_order_data
from core.config import Config
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
from imblearn.combine import SMOTETomek
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
df['order_month'] = df['order_date'].dt.month
df['order_weekday'] = df['order_date'].dt.weekday

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

# Müşteri ömrü hesapla
first_order = df.groupby('customer_id')['order_date'].min().reset_index()
first_order.columns = ['customer_id', 'first_order_date']
df = df.merge(first_order, on='customer_id')
df['customer_lifetime'] = (df['order_date'].max() - df['first_order_date']).dt.days

# Sipariş aralıkları
df_sorted = df.sort_values(['customer_id', 'order_date'])
df_sorted['gap_days'] = df_sorted.groupby('customer_id')['order_date'].diff().dt.days
avg_order_gap = df_sorted.groupby('customer_id')['gap_days'].mean().reset_index()
avg_order_gap.columns = ['customer_id', 'avg_days_between_orders']

# Toplam harcama
df['total'] = df['unit_price'] * df['quantity']

# Müşteri bazlı aggregate
agg = df.groupby('customer_id').agg(
    total_spend=('total', 'sum'),
    order_count=('order_id', 'nunique'),
    last_order_date=('order_date', 'max'),
    customer_lifetime=('customer_lifetime', 'max')
).reset_index()
agg['avg_order_size'] = agg['total_spend'] / agg['order_count']
cutoff_date = agg['last_order_date'].max() - pd.DateOffset(months=6)
agg['label'] = (agg['last_order_date'] > cutoff_date).astype(int)
agg = agg.merge(avg_order_gap, on='customer_id', how='left')
agg.drop(columns=['last_order_date'], inplace=True)
agg['orders_per_month'] = agg['order_count'] / (agg['customer_lifetime'] / 30)
# agg['spend_per_day'] = agg['total_spend'] / agg['customer_lifetime']

# 🔁 Yeni temporal feature'lar
# 1. En sık sipariş verilen ay
most_common_month = df.groupby('customer_id')['order_month'].agg(lambda x: x.mode()[0]).reset_index()
most_common_month.columns = ['customer_id', 'common_order_month']
agg = agg.merge(most_common_month, on='customer_id', how='left')

# 3. Mevsim One-Hot Encoding
season_encoded = pd.get_dummies(df[['customer_id', 'order_season']], columns=['order_season'])
season_agg = season_encoded.groupby('customer_id').sum().reset_index()
agg = agg.merge(season_agg, on='customer_id', how='left')

# Özellikler ve hedef
X = agg.drop(columns=['customer_id', 'label', 'order_season_Winter', 'total_spend', 'order_season_Summer'])
y = agg['label']


# Standardizasyon
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, Config.PROJECT_ROOT / 'src/models/order_habit/preprocessor.pkl')

# Train-test bölme
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, test_size=0.3, random_state=SEED)

# Train/test split – Test setinde daha fazla label=0 olacak şekilde manuel ayır

# 1. Azınlık ve çoğunluk sınıflarını ayır
X_minority = X_scaled[y == 0]
X_majority = X_scaled[y == 1]
y_minority = y[y == 0]
y_majority = y[y == 1]

# 2. Test setine azınlık sınıfından sabit sayıda ayır (örnek: 5)
test_size_min = min(2, len(X_minority))  # Güvenli olması için
X_test_min = X_minority[:test_size_min]
y_test_min = y_minority[:test_size_min]

# 3. Geri kalan azınlık verisini train'e bırak
X_train_min = X_minority[test_size_min:]
y_train_min = y_minority[test_size_min:]

# 4. Çoğunluk sınıfı için klasik split (%70 train, %30 test)
X_train_maj, X_test_maj, y_train_maj, y_test_maj = train_test_split(
    X_majority, y_majority, test_size=0.3, random_state=SEED
)

# 5. Train ve test kümelerini birleştir
X_train = np.vstack([X_train_min, X_train_maj])
y_train = pd.concat([y_train_min, y_train_maj])

X_test = np.vstack([X_test_min, X_test_maj])
y_test = pd.concat([y_test_min, y_test_maj])

# Class weight
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
weights = dict(zip(np.unique(y_train), class_weights))

# SMOTE + Tomek Links
smote = SMOTE(k_neighbors=1, sampling_strategy=0.75, random_state=SEED)
smt = SMOTETomek(smote=smote, random_state=SEED)
X_resampled, y_resampled = smt.fit_resample(X_train, y_train)

# Model tanımı (Dropout'lu)
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Model eğitimi
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    class_weight=weights,
    epochs=50,
    batch_size=16,
    callbacks=[early_stop]
)

model.save(Config.PROJECT_ROOT / 'src/models/order_habit/model.h5')

# Doğruluk grafiği
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Kayıp grafiği
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# Tahmin ve ROC/AUC
y_pred = model.predict(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
y_pred_labels = (y_pred > optimal_threshold).astype(int)

auc_score = roc_auc_score(y_test, y_pred)
print(f"AUC Score: {auc_score:.4f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_labels)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Sınıflandırma metrikleri
print(classification_report(y_test, y_pred_labels))


from sklearn.metrics import roc_auc_score
import copy

baseline_auc = roc_auc_score(y_test, model.predict(X_test))

importances = []
for i in range(X_test.shape[1]):
    X_test_permuted = X_test.copy()
    np.random.shuffle(X_test_permuted[:, i])
    permuted_auc = roc_auc_score(y_test, model.predict(X_test_permuted))
    importances.append(baseline_auc - permuted_auc)

# Sonuçları göster
for feature, imp in zip(X.columns, importances):
    print(f"{feature}: {imp:.4f}")
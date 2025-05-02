import pandas as pd
import numpy as np
from data.fetch_orders import get_orders_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import random
import tensorflow as tf

pd.set_option('display.max_columns', None)  # Tüm kolonları göster
pd.set_option('display.width', 500)        # Satır genişliğini sınırsız yap
pd.set_option('display.max_colwidth', 500)  # Kolon içeriklerini tam göster

# Sabit tohum
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
df = get_orders_data()
df.head()

# df['discount']: indirim oranları
# df['count']: her indirim oranının kaç kez görüldüğü
discount_values_count = df[df['discount'] != 0.00].value_counts().reset_index()
weighted_avg_discount = np.average(discount_values_count['discount'], weights=discount_values_count['count'])

low_spend = df['total_spend'].quantile(0.25)

def label_high_return_risk(row):
    if row['discount'] > weighted_avg_discount and row['total_spend'] < low_spend:
        return 1
    else:
        return 0

df['high_return_risk'] = df.apply(label_high_return_risk, axis=1)


# Giriş ve hedef değişkenleri ayır
X = df[['discount', 'quantity', 'unit_price', 'total_spend']]
y = df['high_return_risk']

# SMOTE uygulayarak dengesiz sınıf problemini çöz
smote = SMOTE(random_state=SEED)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Eğitim-test bölünmesi
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=SEED)

# Normalizasyon
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Model eğitimi
history = model.fit(
    X_train_scaled, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# Tahmin
y_pred = model.predict(X_test_scaled)
y_pred_labels = (y_pred > 0.5).astype(int)

# Rapor
print(confusion_matrix(y_test, y_pred_labels))
print(classification_report(y_test, y_pred_labels))


print("AUC:", roc_auc_score(y_test, y_pred))


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

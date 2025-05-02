import random
import numpy as np
import pandas as pd
import shap
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from core.config import Config
from data.fetch_orders import get_orders_data

# Ayarlar
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.max_colwidth', 500)

# Sabit tohumlar
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Veri Yükleme ve Etiketleme

def calculate_weighted_discount(df):
    discounts = df[df['discount'] != 0.00].value_counts().reset_index()
    return np.average(discounts['discount'], weights=discounts['count'])

def label_return_risk(df, weighted_discount, low_spend_threshold):
    return df.apply(lambda row: int(row['discount'] > weighted_discount and row['total_spend'] < low_spend_threshold), axis=1)

def prepare_data(df):
    weighted_discount = calculate_weighted_discount(df)
    low_spend_threshold = df['total_spend'].quantile(0.25)
    df['high_return_risk'] = label_return_risk(df, weighted_discount, low_spend_threshold)
    X = df[['discount', 'quantity', 'unit_price', 'total_spend']]
    y = df['high_return_risk']
    return X, y

# Model Oluşturma

def build_model(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Grafik Fonksiyonları

def save_accuracy_plot(history, output_dir):
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / 'model_accuracy.png')
    plt.close()

def save_loss_plot(history, output_dir):
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / 'model_loss.png')
    plt.close()

def save_shap_plots(model, X_train_scaled, X_test_scaled, feature_names, output_dir):
    X_background = X_train_scaled[:100]
    explainer = shap.KernelExplainer(model.predict, X_background)
    shap_values = explainer.shap_values(X_test_scaled[:10])

    shap.waterfall_plot(shap.Explanation(
        values=shap_values[0][0],
        base_values=explainer.expected_value[0],
        data=X_test_scaled[0],
        feature_names=feature_names
    ), show=False)
    plt.savefig(output_dir / 'shap_waterfall.png')
    plt.close()

    shap.summary_plot(shap_values[0], X_test_scaled[:10], feature_names=feature_names, show=False)
    plt.savefig(output_dir / 'shap_summary.png')
    plt.close()

def save_confusion_matrix(y_true, y_pred, output_dir):
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=['Actual 0', 'Actual 1'], columns=['Predicted 0', 'Predicted 1'])
    cm_df.to_csv(output_dir / 'confusion_matrix.csv')

def save_classification_report(y_true, y_pred, y_probs, output_dir):
    report = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(output_dir / 'classification_report.csv')

    auc_score = roc_auc_score(y_true, y_probs)
    with open(output_dir / 'auc_score.txt', 'w') as f:
        f.write(f"AUC: {auc_score:.4f}\n")

# Ana Akış
df = get_orders_data()
X, y = prepare_data(df)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
weights = dict(zip(np.unique(y_train), class_weights))

model = build_model(X_train_scaled.shape[1])
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    batch_size=32,
    epochs=100,
    class_weight=weights,
    callbacks=[early_stop],
    verbose=1
)

model_path = Path(Config.PROJECT_ROOT) / 'src/models/return_risk/model.h5'
model.save(model_path)

# Değerlendirme
y_pred_probs = model.predict(X_test_scaled)
y_pred_labels = (y_pred_probs > 0.5).astype(int)

print(confusion_matrix(y_test, y_pred_labels))
print(classification_report(y_test, y_pred_labels))
print("AUC:", roc_auc_score(y_test, y_pred_probs))

# Grafikler
output_dir = Path(Config.PROJECT_ROOT) / 'src/models/return_risk'
save_accuracy_plot(history, output_dir)
save_loss_plot(history, output_dir)
save_shap_plots(model, X_train_scaled, X_test_scaled, X.columns.tolist(), output_dir)
save_confusion_matrix(y_test, y_pred_labels, output_dir)
save_classification_report(y_test, y_pred_labels, y_pred_probs, output_dir)
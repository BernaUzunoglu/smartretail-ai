import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import random
import tensorflow as tf
from data.fetch_customer_order_summary import get_customer_order_data
from core.config import Config
import logging

# Constants
SEED = 42
TEST_SIZE_MINORITY = 3
EPOCHS = 50
BATCH_SIZE = 16
PATIENCE = 5

# Seeding
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

logging.basicConfig(level=logging.INFO)

def preprocess_data():
    df = get_customer_order_data()
    df['order_date'] = pd.to_datetime(df['order_date'])
    df['order_month'] = df['order_date'].dt.month
    df['order_weekday'] = df['order_date'].dt.weekday

    def get_season(month):
        return ('Winter' if month in [12, 1, 2] else
                'Spring' if month in [3, 4, 5] else
                'Summer' if month in [6, 7, 8] else 'Fall')

    df['order_season'] = df['order_month'].apply(get_season)

    first_order = df.groupby('customer_id')['order_date'].min().reset_index()
    first_order.columns = ['customer_id', 'first_order_date']
    df = df.merge(first_order, on='customer_id')
    df['customer_lifetime'] = (df['order_date'].max() - df['first_order_date']).dt.days

    df_sorted = df.sort_values(['customer_id', 'order_date'])
    df_sorted['gap_days'] = df_sorted.groupby('customer_id')['order_date'].diff().dt.days
    avg_order_gap = df_sorted.groupby('customer_id')['gap_days'].mean().reset_index()
    avg_order_gap.columns = ['customer_id', 'avg_days_between_orders']

    df['total'] = df['unit_price'] * df['quantity']

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

    most_common_month = df.groupby('customer_id')['order_month'].agg(lambda x: x.mode()[0]).reset_index()
    most_common_month.columns = ['customer_id', 'common_order_month']
    agg = agg.merge(most_common_month, on='customer_id', how='left')

    season_encoded = pd.get_dummies(df[['customer_id', 'order_season']], columns=['order_season'])
    season_agg = season_encoded.groupby('customer_id').sum().reset_index()
    agg = agg.merge(season_agg, on='customer_id', how='left')

    return agg

def prepare_features(agg):
    X = agg.drop(columns=['customer_id', 'label', 'order_season_Winter', 'total_spend'])
    y = agg['label']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, Config.PROJECT_ROOT / 'src/models/order_habit/preprocessor.pkl')
    return X_scaled, y, X.columns

def manual_train_test_split(X_scaled, y):
    X_minority = X_scaled[y == 0]
    X_majority = X_scaled[y == 1]
    y_minority = y[y == 0]
    y_majority = y[y == 1]

    X_test_min = X_minority[:TEST_SIZE_MINORITY]
    y_test_min = y_minority[:TEST_SIZE_MINORITY]
    X_train_min = X_minority[TEST_SIZE_MINORITY:]
    y_train_min = y_minority[TEST_SIZE_MINORITY:]

    X_train_maj, X_test_maj, y_train_maj, y_test_maj = train_test_split(
        X_majority, y_majority, test_size=0.3, random_state=SEED)

    X_train = np.vstack([X_train_min, X_train_maj])
    y_train = pd.concat([y_train_min, y_train_maj])
    X_test = np.vstack([X_test_min, X_test_maj])
    y_test = pd.concat([y_test_min, y_test_maj])

    return X_train, X_test, y_train, y_test, X_train_min, y_train_min

def balance_with_smote(X_train, y_train):
    smote = SMOTE(k_neighbors=1, sampling_strategy=0.75, random_state=SEED)
    smt = SMOTETomek(smote=smote, random_state=SEED)
    return smt.fit_resample(X_train, y_train)

def build_model(input_shape):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_shape,)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def plot_training(history):
    plt.plot(history.history['accuracy'], label='train_acc')
    plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(Config.PROJECT_ROOT / 'src/models/order_habit/accuracy_plot.png')
    plt.show()

    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(Config.PROJECT_ROOT / 'src/models/order_habit/loss_plot.png')
    plt.show()

def evaluate_model(model, X_test, y_test, label="Test Set"):
    y_pred = model.predict(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    y_pred_labels = (y_pred > optimal_threshold).astype(int)

    auc = roc_auc_score(y_test, y_pred)
    print(f"{label} AUC: {auc:.4f}")
    print(confusion_matrix(y_test, y_pred_labels))
    print(classification_report(y_test, y_pred_labels))
    save_metrics_to_file(y_test, y_pred, Config.PROJECT_ROOT / 'src/models/order_habit/metrics.txt')
    return auc

def plot_confusion_matrix(y_true, y_pred_labels):
    cm = confusion_matrix(y_true, y_pred_labels)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig(Config.PROJECT_ROOT / 'src/models/order_habit/Confusion_Matrix.png')
    plt.show()

def calculate_feature_importance(model, X_test, y_test, feature_names):
    baseline_auc = roc_auc_score(y_test, model.predict(X_test))
    importances = []
    for i in range(X_test.shape[1]):
        X_test_permuted = X_test.copy()
        np.random.shuffle(X_test_permuted[:, i])
        permuted_auc = roc_auc_score(y_test, model.predict(X_test_permuted))
        importances.append(baseline_auc - permuted_auc)
    for feature, imp in zip(feature_names, importances):
        print(f"{feature}: {imp:.4f}")

def save_metrics_to_file(y_true, y_pred_probs, output_path):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs)
    auc_score = roc_auc_score(y_true, y_pred_probs)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    y_pred_labels = (y_pred_probs > optimal_threshold).astype(int)

    report = classification_report(y_true, y_pred_labels)
    cm = confusion_matrix(y_true, y_pred_labels)

    with open(output_path, 'w') as f:
        f.write(f"AUC: {auc_score:.4f}\n")
        f.write(f"Optimal Threshold: {optimal_threshold:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(cm))
        f.write("\n\nClassification Report:\n")
        f.write(report)
if __name__ == "__main__":
    agg = preprocess_data()
    X_scaled, y, feature_names = prepare_features(agg)
    X_train, X_test, y_train, y_test, X_train_min, y_train_min = manual_train_test_split(X_scaled, y)
    X_resampled, y_resampled = balance_with_smote(X_train, y_train)

    weights = dict(zip(np.unique(y_train), compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)))
    model = build_model(X_resampled.shape[1])
    early_stop = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)

    history = model.fit(X_resampled, y_resampled, validation_data=(X_test, y_test),
                        class_weight=weights, epochs=EPOCHS, batch_size=BATCH_SIZE,
                        callbacks=[early_stop])

    model.save(Config.PROJECT_ROOT / 'src/models/order_habit/model.h5')
    plot_training(history)

    # Evaluate on original test
    evaluate_model(model, X_test, y_test)

    # Extended test set with extra zeros
    extra_zeros = 10
    X_extra_zeros = X_train_min[:extra_zeros]
    y_extra_zeros = y_train_min[:extra_zeros]
    X_test_ext = np.vstack([X_test, X_extra_zeros])
    y_test_ext = pd.concat([y_test, y_extra_zeros])
    evaluate_model(model, X_test_ext, y_test_ext, label="Extended Test")

    # Feature importance
    calculate_feature_importance(model, X_test, y_test, feature_names)


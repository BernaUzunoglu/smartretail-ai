import os
import joblib
import random
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from data.product_purchase_data import get_category_features, get_customer_features, get_customer_category_spend
from core.config import Config

# Constants
SEED = 42
TEST_SIZE = 0.2
EPOCHS = 40
BATCH_SIZE = 256
PATIENCE = 7
EMBEDDING_SIZE = 24

# Seeding
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.config.experimental.enable_op_determinism()


def encode_dataframes():
    """Encode all DataFrames and return them"""
    # Load data
    customer_category_spend = get_customer_category_spend()
    customer_features = get_customer_features()
    category_features = get_category_features()

    # Create encoders
    customer_encoder = LabelEncoder()
    category_encoder = LabelEncoder()

    # Encode customer IDs
    customer_ids = customer_encoder.fit_transform(customer_features.index)
    customer_features.index = customer_ids
    customer_category_spend.index = customer_encoder.transform(customer_category_spend.index)

    # Encode category names
    category_names = category_encoder.fit_transform(category_features.index)
    category_features.index = category_names
    customer_category_spend.columns = category_encoder.transform(customer_category_spend.columns)

    return {
        'customer_category_spend': customer_category_spend,
        'customer_features': customer_features,
        'category_features': category_features,
        'customer_encoder': customer_encoder,
        'category_encoder': category_encoder
    }


def preprocess_data():
    """Load and preprocess all data"""
    # Load and encode data
    data = encode_dataframes()
    customer_category_spend = data['customer_category_spend']
    customer_features = data['customer_features']
    category_features = data['category_features']
    customer_encoder = data['customer_encoder']
    category_encoder = data['category_encoder']

    # Store last_order separately
    last_order = customer_features['last_order']
    customer_features = customer_features.drop('last_order', axis=1)

    # Normalize numerical features
    customer_features_normalized = (customer_features - customer_features.mean()) / customer_features.std()
    category_features_normalized = (category_features - category_features.mean()) / category_features.std()

    # Preprocess customer-category matrix
    customer_category_spend = normalize_and_clean_data(customer_category_spend)

    threshold = customer_category_spend.quantile(0.75).mean()

    return {
        'customer_category_spend': customer_category_spend,
        'customer_features': customer_features_normalized,
        'category_features': category_features_normalized,
        'customer_encoder': customer_encoder,
        'category_encoder': category_encoder,
        'threshold': threshold
    }


def normalize_and_clean_data(df):
    """Advanced data preprocessing"""
    # Z-score normalization
    normalized_df = (df - df.mean()) / df.std()

    # Outlier cleaning (clip values with z-score > 3)
    normalized_df = normalized_df.clip(-3, 3)

    # Log transformation - signed log for negative values
    return np.sign(normalized_df) * np.log1p(np.abs(normalized_df))


def prepare_features(data):
    """Prepare features for model training"""
    customer_category_spend = data['customer_category_spend']
    customer_features = data['customer_features']
    category_features = data['category_features']
    threshold = data['threshold']

    X_user_id = []
    X_category_id = []
    X_user_features = []
    X_category_features = []
    y = []

    for i, user_id in enumerate(customer_category_spend.index):
        user_feat = customer_features.loc[user_id].values

        for j, category_id in enumerate(customer_category_spend.columns):
            category_feat = category_features.loc[category_id].values

            X_user_id.append(user_id)
            X_category_id.append(category_id)
            X_user_features.append(user_feat)
            X_category_features.append(category_feat)
            y.append(1 if customer_category_spend.iloc[i, j] > threshold else 0)

    X_user_id = np.array(X_user_id)
    X_category_id = np.array(X_category_id)
    X_user_features = np.array(X_user_features)
    X_category_features = np.array(X_category_features)
    y = np.array(y)

    num_users = len(np.unique(X_user_id))
    num_categories = len(np.unique(X_category_id))
    user_features_dim = X_user_features.shape[1]
    category_features_dim = X_category_features.shape[1]

    return {
        'X_user_id': X_user_id,
        'X_category_id': X_category_id,
        'X_user_features': X_user_features,
        'X_category_features': X_category_features,
        'y': y,
        'num_users': num_users,
        'num_categories': num_categories,
        'user_features_dim': user_features_dim,
        'category_features_dim': category_features_dim
    }


def split_train_test(features_data):
    """Split data into train and test sets"""
    X_user_id = features_data['X_user_id']
    X_category_id = features_data['X_category_id']
    X_user_features = features_data['X_user_features']
    X_category_features = features_data['X_category_features']
    y = features_data['y']

    indices = np.arange(len(y))
    indices_train, indices_test = train_test_split(
        indices, test_size=TEST_SIZE, random_state=SEED, stratify=y, shuffle=True
    )

    train_data = {
        'X_user_id': X_user_id[indices_train],
        'X_category_id': X_category_id[indices_train],
        'X_user_features': X_user_features[indices_train],
        'X_category_features': X_category_features[indices_train],
        'y': y[indices_train]
    }

    test_data = {
        'X_user_id': X_user_id[indices_test],
        'X_category_id': X_category_id[indices_test],
        'X_user_features': X_user_features[indices_test],
        'X_category_features': X_category_features[indices_test],
        'y': y[indices_test]
    }

    return train_data, test_data


def balance_with_undersampling(train_data):
    """Balance train data using Random Under-Sampling"""
    X_user_id = train_data['X_user_id']
    X_category_id = train_data['X_category_id']
    X_user_features = train_data['X_user_features']
    X_category_features = train_data['X_category_features']
    y = train_data['y']

    # Combine data for undersampling
    X_combined = np.column_stack([
        np.arange(len(y)),  # Indices
        np.zeros(len(y))  # Dummy feature
    ])

    rus = RandomUnderSampler(random_state=SEED)
    X_resampled, y_resampled = rus.fit_resample(X_combined, y)

    # Get resampled indices
    resampled_indices = X_resampled[:, 0].astype(int)

    # Use indices to select data
    balanced_data = {
        'X_user_id': X_user_id[resampled_indices],
        'X_category_id': X_category_id[resampled_indices],
        'X_user_features': X_user_features[resampled_indices],
        'X_category_features': X_category_features[resampled_indices],
        'y': y_resampled
    }

    return balanced_data


def build_model(model_params):
    """Create and compile the neural collaborative filtering model"""
    num_users = model_params['num_users']
    num_categories = model_params['num_categories']
    user_features_dim = model_params['user_features_dim']
    category_features_dim = model_params['category_features_dim']

    # Set seed for kernel initializers
    kernel_init = tf.keras.initializers.GlorotUniform(seed=SEED)

    # User input and embedding
    user_id_input = tf.keras.layers.Input(shape=(1,), name='user_id_input')
    user_embedding = tf.keras.layers.Embedding(
        num_users,
        EMBEDDING_SIZE,
        embeddings_regularizer=tf.keras.regularizers.l2(0.05),
        embeddings_initializer=kernel_init,
        name='user_embedding'
    )(user_id_input)
    user_vec = tf.keras.layers.Flatten(name='user_flatten')(user_embedding)

    # User features input
    user_features_input = tf.keras.layers.Input(shape=(user_features_dim,), name='user_features_input')

    # Category input and embedding
    category_id_input = tf.keras.layers.Input(shape=(1,), name='category_id_input')
    category_embedding = tf.keras.layers.Embedding(
        num_categories,
        EMBEDDING_SIZE,
        embeddings_regularizer=tf.keras.regularizers.l2(0.05),
        embeddings_initializer=kernel_init,
        name='category_embedding'
    )(category_id_input)
    category_vec = tf.keras.layers.Flatten(name='category_flatten')(category_embedding)

    # Category features input
    category_features_input = tf.keras.layers.Input(shape=(category_features_dim,), name='category_features_input')

    # Concatenate all features
    concat = tf.keras.layers.Concatenate()([
        user_vec, user_features_input, category_vec, category_features_input
    ])

    # First layer
    dense1 = tf.keras.layers.Dense(
        64,
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(0.05),
        kernel_initializer=kernel_init
    )(concat)
    batch_norm1 = tf.keras.layers.BatchNormalization()(dense1)
    dropout1 = tf.keras.layers.Dropout(0.3, seed=SEED)(batch_norm1)

    # Second layer
    dense2 = tf.keras.layers.Dense(
        32,
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(0.05),
        kernel_initializer=kernel_init
    )(dropout1)
    batch_norm2 = tf.keras.layers.BatchNormalization()(dense2)
    dropout2 = tf.keras.layers.Dropout(0.2, seed=SEED)(batch_norm2)

    # Output layer
    output = tf.keras.layers.Dense(
        1,
        activation='sigmoid',
        kernel_initializer=kernel_init
    )(dropout2)

    model = tf.keras.Model(
        inputs=[user_id_input, user_features_input, category_id_input, category_features_input],
        outputs=output
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )

    return model


def plot_training_history(history):
    """Visualize training history"""
    plt.figure(figsize=(12, 4))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(Config.PROJECT_ROOT / 'src/models/product_purchase_potential/training_history.png')
    plt.close()


def plot_roc_curve(y_true, y_pred_proba):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")

    plt.savefig(Config.PROJECT_ROOT / 'src/models/product_purchase_potential/roc_curve.png')
    plt.close()


def plot_confusion_matrix(y_true, y_pred):
    """Visualize confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    plt.savefig(Config.PROJECT_ROOT / 'src/models/product_purchase_potential/confusion_matrix.png')
    plt.close()


def find_optimal_threshold(y_true, y_pred_proba):
    """Find optimal threshold using ROC curve"""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold


def evaluate_model(model, test_data, label="Test Set"):
    """Evaluate model and return metrics"""
    X_user_id = test_data['X_user_id']
    X_category_id = test_data['X_category_id']
    X_user_features = test_data['X_user_features']
    X_category_features = test_data['X_category_features']
    y_true = test_data['y']

    y_pred_proba = model.predict([X_user_id, X_user_features, X_category_id, X_category_features])

    # Find optimal threshold
    optimal_threshold = find_optimal_threshold(y_true, y_pred_proba)
    y_pred = (y_pred_proba >= optimal_threshold).astype(int)

    # Calculate AUC
    auc_score = auc(*roc_curve(y_true, y_pred_proba)[:2])
    print(f"{label} AUC: {auc_score:.4f}")
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred))

    # Save metrics
    save_metrics_to_file(y_true, y_pred_proba, optimal_threshold,
                         Config.PROJECT_ROOT / 'src/models/product_purchase_potential/metrics.txt')

    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred)

    # Plot ROC curve
    plot_roc_curve(y_true, y_pred_proba)

    return optimal_threshold


def save_metrics_to_file(y_true, y_pred_proba, threshold, output_path):
    """Save evaluation metrics to file"""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    auc_score = auc(fpr, tpr)
    y_pred = (y_pred_proba >= threshold).astype(int)

    report = classification_report(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    with open(output_path, 'w') as f:
        f.write(f"AUC: {auc_score:.4f}\n")
        f.write(f"Optimal Threshold: {threshold:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(cm))
        f.write("\n\nClassification Report:\n")
        f.write(report)


if __name__ == "__main__":
    try:
        # Step 1: Preprocess data
        data = preprocess_data()

        # Step 2: Prepare features
        features_data = prepare_features(data)

        # Step 3: Split data into train and test sets
        train_data, test_data = split_train_test(features_data)

        # Step 4: Balance train data
        balanced_train_data = balance_with_undersampling(train_data)

        # Step 5: Build model
        model_params = {
            'num_users': features_data['num_users'],
            'num_categories': features_data['num_categories'],
            'user_features_dim': features_data['user_features_dim'],
            'category_features_dim': features_data['category_features_dim']
        }
        model = build_model(model_params)

        # Step 6: Create callbacks for training
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=PATIENCE,
            restore_best_weights=True,
            min_delta=0.001
        )

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=0.00001
        )

        # Step 7: Train model
        history = model.fit(
            [
                balanced_train_data['X_user_id'],
                balanced_train_data['X_user_features'],
                balanced_train_data['X_category_id'],
                balanced_train_data['X_category_features']
            ],
            balanced_train_data['y'],
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=0.2,
            shuffle=True,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )

        # Step 8: Plot training history
        plot_training_history(history)

        # Step 9: Evaluate model
        optimal_threshold = evaluate_model(model, test_data)

        # Step 10: Save model and artifacts
        model.save(Config.PROJECT_ROOT / 'src/models/product_purchase_potential/model.h5')
        joblib.dump(data['customer_encoder'],
                    Config.PROJECT_ROOT / 'src/models/product_purchase_potential/customer_encoder.joblib')
        joblib.dump(data['category_encoder'],
                    Config.PROJECT_ROOT / 'src/models/product_purchase_potential/category_encoder.joblib')

        # Save normalized features
        data['customer_features'].to_parquet(
            Config.PROJECT_ROOT / 'src/models/product_purchase_potential/customer_features.parquet')
        data['category_features'].to_parquet(
            Config.PROJECT_ROOT / 'src/models/product_purchase_potential/category_features.parquet')

        # Save optimal threshold
        with open(Config.PROJECT_ROOT / 'src/models/product_purchase_potential/optimal_threshold.txt', 'w') as f:
            f.write(str(optimal_threshold))

        print("Model training and evaluation completed successfully!")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback

        traceback.print_exc()
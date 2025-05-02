import os
import joblib
import random
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from core.config import Config
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from data.product_purchase_data import get_category_features, get_customer_features, get_customer_category_spend

# Global seed settings for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

# Ensure TensorFlow operations are deterministic
tf.config.experimental.enable_op_determinism()

#
def encode_dataframes():
    """Tüm DataFrame'leri kodlar ve döndürür"""
    # Verileri yükle
    customer_category_spend = get_customer_category_spend()
    customer_features = get_customer_features()
    category_features = get_category_features()

    # Encoder'ları oluştur
    customer_encoder = LabelEncoder()
    category_encoder = LabelEncoder()

    # Müşteri ID'lerini kodla
    customer_ids = customer_encoder.fit_transform(customer_features.index)
    customer_features.index = customer_ids
    customer_category_spend.index = customer_encoder.transform(customer_category_spend.index)

    # Kategori isimlerini kodla
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


def calculate_customer_features(df):
    """Müşteri bazlı özellikler oluştur"""
    # Müşteri alışveriş sıklığı
    customer_frequency = df.groupby('customer_id').size()
    
    # Müşteri toplam harcama (sipariş miktarı * birim fiyat)
    customer_total_spend = df.groupby('customer_id').apply(lambda x: (x['order_unit_price'] * x['quantity']).sum())
    
    # Müşteri ortalama harcama
    customer_avg_spend = df.groupby('customer_id').apply(lambda x: (x['order_unit_price'] * x['quantity']).mean())
    
    # Müşteri son alışveriş tarihi
    customer_last_order = df.groupby('customer_id')['order_date'].max()
    
    # Özellikleri birleştir
    customer_features = pd.DataFrame({
        'frequency': customer_frequency,
        'total_spend': customer_total_spend,
        'avg_spend': customer_avg_spend,
        'last_order': customer_last_order
    })
    
    return customer_features

def calculate_category_features(df):
    """Kategori bazlı özellikler oluştur"""
    # Kategori popülerliği
    category_popularity = df.groupby('category_name').size()
    
    # Kategori ortalama fiyat
    category_avg_price = df.groupby('category_name')['product_unit_price'].mean()
    
    # Kategori toplam satış
    category_total_sales = df.groupby('category_name').apply(lambda x: (x['order_unit_price'] * x['quantity']).sum())
    
    # Özellikleri birleştir
    category_features = pd.DataFrame({
        'popularity': category_popularity,
        'avg_price': category_avg_price,
        'total_sales': category_total_sales
    })
    
    return category_features

def calculate_class_weights(y):
    """Sınıf ağırlıklarını hesapla"""
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y),
        y=y
    )
    return dict(enumerate(class_weights))

def apply_smote(X_user, X_category, y):
    """SMOTE ile veri dengesizliğini gider"""
    # X verilerini birleştir
    X_combined = np.column_stack((X_user, X_category))
    
    # SMOTE uygula
    smote = SMOTE(random_state=SEED)
    X_resampled, y_resampled = smote.fit_resample(X_combined, y)
    
    # X verilerini tekrar ayır
    X_user_resampled = X_resampled[:, 0]
    X_category_resampled = X_resampled[:, 1]
    
    return X_user_resampled, X_category_resampled, y_resampled

def find_optimal_threshold(y_true, y_pred_proba):
    """ROC eğrisi kullanarak optimal threshold değerini bul"""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold

def plot_roc_curve(y_true, y_pred_proba):
    """ROC eğrisini çiz"""
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

def create_neural_cf_model(num_users, num_categories, user_features_dim, category_features_dim, embedding_size=24):
    # Set seed for kernel initializers
    kernel_init = tf.keras.initializers.GlorotUniform(seed=SEED)
    
    # Kullanıcı girişi ve embedding
    user_id_input = tf.keras.layers.Input(shape=(1,), name='user_id_input')
    user_embedding = tf.keras.layers.Embedding(num_users, embedding_size, 
                                             embeddings_regularizer=tf.keras.regularizers.l2(0.05),
                                             embeddings_initializer=kernel_init,
                                             name='user_embedding')(user_id_input)
    user_vec = tf.keras.layers.Flatten(name='user_flatten')(user_embedding)
    
    # Kullanıcı özellikleri girişi
    user_features_input = tf.keras.layers.Input(shape=(user_features_dim,), name='user_features_input')
    
    # Kategori girişi ve embedding
    category_id_input = tf.keras.layers.Input(shape=(1,), name='category_id_input')
    category_embedding = tf.keras.layers.Embedding(num_categories, embedding_size,
                                                 embeddings_regularizer=tf.keras.regularizers.l2(0.05),
                                                 embeddings_initializer=kernel_init,
                                                 name='category_embedding')(category_id_input)
    category_vec = tf.keras.layers.Flatten(name='category_flatten')(category_embedding)
    
    # Kategori özellikleri girişi
    category_features_input = tf.keras.layers.Input(shape=(category_features_dim,), name='category_features_input')
    
    # Tüm özellikleri birleştir
    concat = tf.keras.layers.Concatenate()([user_vec, user_features_input, 
                                          category_vec, category_features_input])
    
    # İlk katman
    dense1 = tf.keras.layers.Dense(64, activation='relu',
                                kernel_regularizer=tf.keras.regularizers.l2(0.05),
                                kernel_initializer=kernel_init)(concat)
    batch_norm1 = tf.keras.layers.BatchNormalization()(dense1)
    dropout1 = tf.keras.layers.Dropout(0.3, seed=SEED)(batch_norm1)
    
    # İkinci katman
    dense2 = tf.keras.layers.Dense(32, activation='relu',
                                kernel_regularizer=tf.keras.regularizers.l2(0.05),
                                kernel_initializer=kernel_init)(dropout1)
    batch_norm2 = tf.keras.layers.BatchNormalization()(dense2)
    dropout2 = tf.keras.layers.Dropout(0.2, seed=SEED)(batch_norm2)
    
    # Çıkış katmanı
    output = tf.keras.layers.Dense(1, activation='sigmoid',
                                kernel_initializer=kernel_init)(dropout2)
    
    model = tf.keras.Model(inputs=[user_id_input, user_features_input,
                                 category_id_input, category_features_input], 
                         outputs=output)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model

def plot_training_history(history):
    """Eğitim geçmişini görselleştirir"""
    plt.figure(figsize=(12, 4))
    
    # Loss grafiği
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Accuracy grafiği
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

def plot_confusion_matrix(y_true, y_pred):
    """Karmaşıklık matrisini görselleştirir"""
    cm = confusion_matrix(y_true, y_pred.round())
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    plt.savefig(Config.PROJECT_ROOT / 'src/models/product_purchase_potential/confusion_matrix.png')
    plt.close()

def plot_category_distribution(customer_features):
    """Kategori dağılımını görselleştirir"""
    category_sums = (customer_features > 0).sum()
    
    plt.figure(figsize=(12, 6))
    category_sums.plot(kind='bar')
    plt.title('Number of Customers per Category')
    plt.xlabel('Category')
    plt.ylabel('Number of Customers')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.savefig(Config.PROJECT_ROOT / 'src/models/product_purchase_potential/category_distribution.png')
    plt.close()

def preprocess_data(customer_features):
    """Gelişmiş veri ön işleme"""
    # Z-score normalizasyonu
    customer_features = (customer_features - customer_features.mean()) / customer_features.std()
    
    # Outlier temizleme (z-score > 3 olan değerler)
    customer_features = customer_features.clip(-3, 3)
    
    # Log transformation - negatif değerler için signed log
    customer_features = np.sign(customer_features) * np.log1p(np.abs(customer_features))
    
    return customer_features

def main():
    try:
        # Set random seeds again for good measure
        np.random.seed(SEED)
        tf.random.set_seed(SEED)
        random.seed(SEED)
        
        # Veri yükleme ve kodlama
        print("Veriler yükleniyor ve kodlanıyor...")
        data = encode_dataframes()
        customer_category_spend = data['customer_category_spend']
        customer_features = data['customer_features']
        category_features = data['category_features']
        customer_encoder = data['customer_encoder']
        category_encoder = data['category_encoder']
        print("Veriler yüklendi ve kodlandı.")
        
        print("\nVeri boyutları:")
        print(f"Müşteri-kategori matrisi: {customer_category_spend.shape}")
        print(f"Müşteri özellikleri: {customer_features.shape}")
        print(f"Kategori özellikleri: {category_features.shape}")
        
        # Özellik normalizasyonu
        print("\nÖzellikler normalize ediliyor...")
        # last_order sütununu ayır
        last_order = customer_features['last_order']
        customer_features = customer_features.drop('last_order', axis=1)
        
        # Sayısal sütunları normalize et
        customer_features_normalized = (customer_features - customer_features.mean()) / customer_features.std()
        category_features_normalized = (category_features - category_features.mean()) / category_features.std()
        
        # Temel müşteri-kategori etkileşim matrisini oluştur
        customer_category_spend = preprocess_data(customer_category_spend)
        
        print("Özellikler normalize edildi.")
        
        # Veri hazırlama
        print("\nEğitim verileri hazırlanıyor...")
        X_user_id = []
        X_category_id = []
        X_user_features = []
        X_category_features = []
        y = []
        
        threshold = customer_category_spend.quantile(0.75).mean()
        
        for i, user_id in enumerate(customer_category_spend.index):
            user_feat = customer_features_normalized.loc[user_id].values
            
            for j, category_id in enumerate(customer_category_spend.columns):
                category_feat = category_features_normalized.loc[category_id].values
                
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
        
        print("\nVeri boyutları:")
        print(f"X_user_id: {X_user_id.shape}")
        print(f"X_category_id: {X_category_id.shape}")
        print(f"X_user_features: {X_user_features.shape}")
        print(f"X_category_features: {X_category_features.shape}")
        print(f"y: {y.shape}")
        
        print("Veri hazırlama tamamlandı.")
        
        # Train-test split
        print("\nTrain-test split yapılıyor...")
        indices = np.arange(len(y))
        indices_train, indices_test = train_test_split(indices, test_size=0.2, random_state=SEED, stratify=y, shuffle=True)
        
        X_user_id_train = X_user_id[indices_train]
        X_user_id_test = X_user_id[indices_test]
        X_category_id_train = X_category_id[indices_train]
        X_category_id_test = X_category_id[indices_test]
        X_user_features_train = X_user_features[indices_train]
        X_user_features_test = X_user_features[indices_test]
        X_category_features_train = X_category_features[indices_train]
        X_category_features_test = X_category_features[indices_test]
        y_train = y[indices_train]
        y_test = y[indices_test]
        
        print("Train-test split tamamlandı.")
        
        # Random Under-Sampling
        print("\nVeri dengesizliği gideriliyor...")
        # Önce veriyi birleştir
        X_combined = np.column_stack([
            np.arange(len(y_train)),  # İndeksler
            np.zeros(len(y_train))    # Dummy feature
        ])
        
        rus = RandomUnderSampler(random_state=SEED)
        X_resampled, y_train = rus.fit_resample(X_combined, y_train)
        
        # Resampled indeksleri al
        resampled_indices = X_resampled[:, 0].astype(int)
        
        # İndeksleri kullanarak verileri seç
        X_user_id_train = X_user_id_train[resampled_indices]
        X_category_id_train = X_category_id_train[resampled_indices]
        X_user_features_train = X_user_features_train[resampled_indices]
        X_category_features_train = X_category_features_train[resampled_indices]
        
        print(f"Veri dengesizliği giderildi. Yeni veri boyutu: {len(y_train)}")
        
        # Model oluşturma
        print("\nModel oluşturuluyor...")
        num_users = len(np.unique(X_user_id))
        num_categories = len(np.unique(X_category_id))
        user_features_dim = X_user_features.shape[1]
        category_features_dim = X_category_features.shape[1]
        
        print(f"Toplam müşteri sayısı: {num_users}")
        print(f"Toplam kategori sayısı: {num_categories}")
        print(f"Müşteri özellik boyutu: {user_features_dim}")
        print(f"Kategori özellik boyutu: {category_features_dim}")
        
        model = create_neural_cf_model(
            num_users=num_users,
            num_categories=num_categories,
            user_features_dim=user_features_dim,
            category_features_dim=category_features_dim
        )
        print("Model oluşturuldu.")
        
        # Early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=7,
            restore_best_weights=True,
            min_delta=0.001
        )
        
        # Reduce learning rate
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=0.00001
        )
        
        # Model eğitimi
        print("\nModel eğitiliyor...")
        history = model.fit(
            [X_user_id_train, X_user_features_train,
             X_category_id_train, X_category_features_train],
            y_train,
            epochs=40,
            batch_size=256,
            validation_split=0.2,
            shuffle=True,  # Explicitly set shuffle
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        print("Model eğitimi tamamlandı.")
        
        # Eğitim geçmişini görselleştir
        plot_training_history(history)
        print("Eğitim geçmişi görselleştirildi.")
        
        # Model değerlendirme
        print("\nModel değerlendiriliyor...")
        y_pred_proba = model.predict(
            [X_user_id_test, X_user_features_test,
             X_category_id_test, X_category_features_test]
        )
        
        # ROC eğrisi
        plot_roc_curve(y_test, y_pred_proba)
        print("ROC eğrisi çizildi.")
        
        # Optimal threshold
        optimal_threshold = find_optimal_threshold(y_test, y_pred_proba)
        print(f"Optimal threshold: {optimal_threshold:.3f}")
        
        # Confusion matrix
        y_pred = (y_pred_proba >= optimal_threshold).astype(int)
        plot_confusion_matrix(y_test, y_pred)
        print("Confusion matrix çizildi.")
        
        # Sınıflandırma raporu
        print("\nSınıflandırma Raporu:")
        print(classification_report(y_test, y_pred))
        
        # Model kaydetme
        model.save(Config.PROJECT_ROOT / 'src/models/product_purchase_potential/model.h5')
        joblib.dump(customer_encoder,
                    Config.PROJECT_ROOT / 'src/models/product_purchase_potential/customer_encoder.joblib')
        joblib.dump(category_encoder,
                    Config.PROJECT_ROOT / 'src/models/product_purchase_potential/category_encoder.joblib')

        # ✅ Yeni: Özellikleri de kaydet
        customer_features_normalized.to_parquet(
            Config.PROJECT_ROOT / 'src/models/product_purchase_potential/customer_features.parquet')
        category_features_normalized.to_parquet(
            Config.PROJECT_ROOT / 'src/models/product_purchase_potential/category_features.parquet')

        # Optimal threshold değerini kaydet
        with open(os.path.join(Config.PROJECT_ROOT / 'src/models/product_purchase_potential/optimal_threshold.txt'), 'w') as f:
            f.write(str(optimal_threshold))
        
        print("Model, encoder'lar ve optimal threshold başarıyla kaydedildi.")
        
    except Exception as e:
        print(f"Hata oluştu: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()


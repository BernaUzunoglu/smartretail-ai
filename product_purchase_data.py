import pandas as pd
from sqlalchemy import create_engine, text
import os
from sklearn.preprocessing import LabelEncoder

def get_db_connection():
    """Veritabanı bağlantısı oluşturur"""
    # Veritabanı bağlantı bilgileri
    DB_USER = os.getenv('DB_USER', 'postgres')
    DB_PASSWORD = os.getenv('DB_PASSWORD', '123456')
    DB_HOST = os.getenv('DB_HOST', 'localhost')
    DB_PORT = os.getenv('DB_PORT', '5432')
    DB_NAME = os.getenv('DB_NAME', 'GYK')
    
    # SQLAlchemy bağlantı URL'si
    connection_url = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    
    # Engine oluştur
    engine = create_engine(connection_url)
    return engine

def get_customer_category_spend():
    """Müşteri-kategori bazında harcama verilerini getirir"""
    engine = get_db_connection()
    
    # SQL sorgusu
    query = text("""
        SELECT 
            o.customer_id,
            c.category_name,
            SUM(od.unit_price * od.quantity) as total_spend
        FROM orders o
        JOIN order_details od ON o.order_id = od.order_id
        JOIN products p ON od.product_id = p.product_id
        JOIN categories c ON p.category_id = c.category_id
        GROUP BY o.customer_id, c.category_name
    """)
    
    # Sorguyu çalıştır ve DataFrame'e dönüştür
    df = pd.read_sql(query, engine)
    
    # Pivot tablo oluştur
    pivot_df = df.pivot(
        index='customer_id',
        columns='category_name',
        values='total_spend'
    ).fillna(0.0)
    
    return pivot_df

def get_customer_features():
    """Müşteri özelliklerini getirir"""
    engine = get_db_connection()
    
    query = text("""
        WITH customer_metrics AS (
            SELECT 
                o.customer_id,
                COUNT(DISTINCT o.order_id) as frequency,
                SUM(od.unit_price * od.quantity) as total_spend,
                AVG(od.unit_price * od.quantity) as avg_spend,
                MAX(o.order_date) as last_order
            FROM orders o
            JOIN order_details od ON o.order_id = od.order_id
            GROUP BY o.customer_id
        )
        SELECT * FROM customer_metrics
    """)
    
    df = pd.read_sql(query, engine)
    
    # customer_id'yi index yap
    df.set_index('customer_id', inplace=True)
    
    return df

def get_category_features():
    """Kategori özelliklerini getirir"""
    engine = get_db_connection()
    
    query = text("""
        WITH category_metrics AS (
            SELECT 
                c.category_name,
                COUNT(DISTINCT od.order_id) as popularity,
                AVG(p.unit_price) as avg_price,
                SUM(od.unit_price * od.quantity) as total_sales
            FROM categories c
            JOIN products p ON c.category_id = p.category_id
            JOIN order_details od ON p.product_id = od.product_id
            GROUP BY c.category_name
        )
        SELECT * FROM category_metrics
    """)
    
    df = pd.read_sql(query, engine)
    
    # category_name'i index yap
    df.set_index('category_name', inplace=True)
    
    return df

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
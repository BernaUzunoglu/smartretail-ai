# SmartRetail AI 🛒🤖

**Northwind veritabanı** üzerinde geliştirilmiş bu proje, derin öğrenme modelleri kullanarak perakende sektöründeki müşteri davranışlarını analiz eder ve tahminler üretir. Projede 3 temel problem çözülmüştür ve tüm modeller REST API formatında sunulmuştur.

---

## 📌 Proje Başlıkları

### 1. Sipariş Verme Alışkanlığı Tahmini
> Müşterilerin harcama geçmişi, sipariş sayısı ve ortalama sipariş büyüklüğüne göre önümüzdeki 6 ay içinde tekrar sipariş verip vermeyeceğini tahmin eden bir model.

- Kullanılan Tablolar: `Customers`, `Orders`, `Order Details`
- Özellikler: Toplam harcama, sipariş sayısı, son sipariş tarihi
- Model: Binary classification (Deep Learning)
- Ar-Ge: Mevsimsellik analizi, class imbalance çözümü (SMOTE)

### 2. Ürün İade Risk Skoru
> Siparişlerdeki indirim oranı, miktar ve harcama tutarına göre siparişin iade edilme riskini tahmin eden bir model.

- Kullanılan Tablolar: `Order Details`, `Orders`
- Özellikler: Discount, Quantity, Total Price
- Etiketleme: Sahte iade verisi (high discount + low total)
- Ar-Ge: Cost-sensitive learning, SHAP ile yorumlama

### 3. Yeni Ürün Satın Alma Potansiyeli
> Müşterinin geçmiş kategorik satın alma alışkanlıklarına göre yeni çıkan bir ürünü satın alma potansiyelini tahmin eden öneri sistemi.

- Kullanılan Tablolar: `Products`, `Categories`, `Orders`, `Order Details`
- Özellikler: Kategori bazlı harcama eğilimleri
- Model: Neural Collaborative Filtering (NCF)
- Ar-Ge: Multi-label prediction, öneri sistemleri

---

## 🛠 Kullanılan Teknolojiler

- Python (TensorFlow, Keras, Scikit-learn, Pandas, NumPy)
- Flask (REST API)
- SHAP / LIME (Model yorumlama)
- SQLite (Northwind veritabanı)
- Jupyter Notebook (EDA & Geliştirme)

---

## Proje Yapısı
````bash
northwind_dl_project/
│
├── api/                          # API katmanı
│   ├── app.py                    # FastAPI ana uygulama
│   ├── endpoints/                # API endpoint'leri
│   │   ├── order_habit.py        # Sipariş alışkanlığı tahmini
│   │   ├── return_risk.py        # İade risk skoru
│   │   └── new_product.py        # Yeni ürün satın alma potansiyeli
│   └── schemas.py                # Pydantic şemaları
│
├── core/                         # Çekirdek iş mantığı
│   ├── config.py                 # Proje konfigürasyonu
│   ├── database.py               # DB bağlantı ve sorgular
│   └── utils.py                  # Yardımcı fonksiyonlar
│
├── models/                       # Eğitilmiş modeller ve pipeline'lar
│   ├── order_habit/              # 1. model için
│   │   ├── model.h5             
│   │   └── preprocessor.pkl      
│   ├── return_risk/              # 2. model için
│   └── new_product/             # 3. model için
│
├── notebooks/                    # Jupyter notebook'lar
│   ├── 1_order_habit.ipynb       # Veri keşfi ve model eğitimi
│   ├── 2_return_risk.ipynb       
│   └── 3_new_product.ipynb      
│
├── services/                     # İş servisleri
│   ├── prediction_service.py     # Tahmin servisi temel sınıfı
│   ├── order_service.py          # 1. model servisi
│   ├── return_service.py         # 2. model servisi
│   └── product_service.py        # 3. model servisi
│
├── tests/                        # Testler
│   ├── unit/                     # Unit testler
│   └── integration/              # Entegrasyon testleri
│
├── requirements.txt              # Python bağımlılıkları
├── Dockerfile                    # Containerizasyon
└── README.md                     # Proje dokümantasyonu
````


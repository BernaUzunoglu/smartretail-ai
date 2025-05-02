```bash

📁 smartretail-ai/
├── 📚 docs/                         # Projeye ait açıklayıcı dokümanlar ve Ar-Ge notları
│   ├── 📝 Order_Habit_Arge.md       # Sipariş alışkanlığı tahminine yönelik yapılan araştırma ve teknik analizler
│   └── 📝 Project_Documentation.md  # Proje genel tanımı, kullanım senaryoları, mimari ve metodolojik belgeler
│
├── 🧠 src/                          # Projenin ana kaynak kodlarını içeren klasör
│   ├── 🌐 api/                      # API uç noktaları ve uygulama çalıştırıcı dosyası
│   │   ├── 📂 endpoints/            # Farklı servisler için API endpoint tanımları (örneğin /predict, /score vs.)
│   │   ├── 🚀 app.py                # FastAPI veya Flask tabanlı uygulamanın giriş noktası
│   │   └── 📄 schemas.py            # API için giriş-çıkış (input/output) veri modelleri (Pydantic ile)
│   │
│   ├── ⚙️ core/                     # Ortak ayarlar, yapılandırmalar ve yardımcı araçlar
│   │   ├── 🛠️ config.py             # Ortam değişkenleri, yol ayarları, model yolları gibi genel konfigürasyonlar
│   │   └── 🧰 utils.py              # Sık kullanılan yardımcı fonksiyonlar (örnek: veri temizleme, tarih dönüştürme)
│   │
│   ├── 🗃 data/                     # Veritabanı işlemleri ve veri çekme dosyaları
│   │   ├── 🗄 database.py           # Veritabanı bağlantısı ve ORM tanımları
│   │   ├── 📊 fetch_customer_order_summary.py  # Müşteri sipariş özetlerini çeken script
│   │   ├── 📦 fetch_orders.py       # Sipariş verilerini çeken script
│   │   └── 🧾 product_purchase_data.py  # Ürün satın alma potansiyeli hesaplama için veri hazırlığı
│   │
│   └── 🧪 models/                   # Farklı tahmin modellerinin dosyaları
│       ├── 📁 order_habit/          # Sipariş alışkanlığı tahmini için model ve eğitim dosyaları
│       ├── 📁 product_purchase_potential/  # Ürün satın alma potansiyeli tahmini için modeller
│       └── 📁 return_risk/          # Ürün iade riski tahmini için modeller
│
├── 📒 notebooks/                    # Geliştirme sürecinde kullanılan Jupyter notebook'lar
│   ├── 🆕 new_product/
│   │   └── 🧮 product_purchase_potential.py  # Yeni ürün satın alma potansiyeli tahmini için notebook
│   │
│   ├── 📊 order_habit/              # Sipariş alışkanlığı modeline ait adım adım geliştirme süreçleri
│   │   ├── 1️⃣ 01_baseline_model.py                # Temel model kurulumu
│   │   ├── 2️⃣ 02_add_smote.py                     # SMOTE ile veri dengeleme
│   │   ├── 3️⃣ 03_manual_split_smote.py            # El ile veri ayırma + SMOTE
│   │   ├── 4️⃣ 04_add_features_monthly_daily.py    # Aylık ve günlük özellik mühendisliği
│   │   ├── 5️⃣ 05_early_stopping_dropout_auc.py    # Dropout ve erken durdurma ile model iyileştirme
│   │   ├── 6️⃣ 06_feature_importance_auc.py        # Özellik önem derecelendirmesi (AUC bazlı)
│   │   ├── 7️⃣ 07_temporal_features_extended.py    # Genişletilmiş zaman tabanlı özellikler
│   │   ├── 8️⃣ 08_temporal_features_v2.py          # Alternatif zaman temelli özellik seti
│   │   └── 🏁 final_order_habit_model.py            # Son versiyon model
│   │
│   └── ♻️ product_return_risk/
│       ├── 🔁 return_risk.py          # Temel iade riski tahmin modeli
│       └── 🔂 return_risk_smote.py    # SMOTE uygulamasıyla geliştirilmiş iade riski modeli
│
├── 🛰 services/                     # Modeli dış dünyaya sunan servis katmanı
│   ├── 📡 order_habit_service.py        # Sipariş alışkanlığı modelini servisleştiren dosya
│   ├── 📡 product_purchase_service.py   # Ürün satın alma modelini servisleştiren dosya
│   └── 📡 return_risk_service.py        # İade riski modelini servisleştiren dosya
│
├── 🚀 run.py                      # Projenin ana çalıştırma noktası (örn. FastAPI sunucusu burada başlar)
├── 🔐 .env                        # Ortam değişkenleri (gizli anahtar, veri yolu vb.)
├── 📄 .env.example                # Ortam değişkenleri için örnek şablon dosya
├── 📄 .gitignore                  # Git'e dahil edilmeyecek dosyalar
├── 📘 README.md                   # Projenin genel açıklaması ve kurulum dökümantasyonu
└── 📦 requirements.txt            # Gerekli Python kütüphaneleri listesi



````
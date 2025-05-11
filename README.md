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

## 📁 Proje Yapısı ve Mimarisi

Bu proje, modüler, ölçeklenebilir ve üretime uygun bir makine öğrenimi tabanlı mikroservis mimarisi üzerine inşa edilmiştir. Aşağıdaki klasör yapısı, her bileşenin görevine göre ayrıldığı, sürdürülebilir ve geliştirilebilir bir yapı sunmaktadır:

### 🔹 `docs/`
Projeye ait teknik dökümantasyon ve Ar-Ge notlarını içerir.
- `Project_Documentation.md`: Proje genel tanımı, kullanım senaryoları, mimari yapı ve metodolojik açıklamaları içerir.
- `Order_Habit_Arge.md`: Sipariş alışkanlığı tahminiyle ilgili yapılan araştırmaları ve deneysel analizleri içerir.

### 🔹 `src/`
Uygulamanın çekirdek kaynak kodlarını barındırır.
- `api/`: FastAPI tabanlı REST API yapısı.  
  - `app.py`: Uygulamanın ana giriş noktası.  
  - `endpoints/`: Tahmin servisleri için endpoint tanımları.  
  - `schemas.py`: Girdi/çıktı veri modelleri (Pydantic ile).

- `core/`: Genel yapılandırma ve yardımcı araçlar.  
  - `config.py`: Ortam ayarları, model yolları ve diğer sabit tanımlar.  
  - `utils.py`: Sık kullanılan yardımcı fonksiyonlar.

- `data/`: Veritabanı bağlantıları ve veri toplama scriptleri.  
  - `database.py`: ORM yapılandırmaları ve bağlantı yönetimi.  
  - `fetch_customer_order_summary.py`, `fetch_orders.py`: Veri çekme scriptleri.  
  - `product_purchase_data.py`: Ürün satın alma potansiyeli verilerinin hazırlanması.

- `models/`: Tahmin modellerinin ve eğitim süreçlerinin bulunduğu klasör.  
  - `order_habit/`: Sipariş alışkanlığı tahmin modeli.  
  - `product_purchase_potential/`: Ürün satın alma potansiyeli tahmin modeli.  
  - `return_risk/`: Ürün iade riski tahmin modeli.

### 🔹 `notebooks/`
Model geliştirme sürecinde kullanılan Jupyter Notebook’ları içerir.
- `order_habit/`: Sipariş alışkanlığı tahmini adımlarını içeren notebook'lar.  
- `new_product/`: Yeni ürünler için satın alma potansiyeli analizleri.  
- `product_return_risk/`: İade riski tahminine dair SMOTE ve modelleme çalışmaları.

### 🔹 `services/`
Eğitilen modellerin servisleştirilerek dış dünyaya sunulmasını sağlar.
- `order_habit_service.py`: Sipariş alışkanlığı modeli servisi.  
- `product_purchase_service.py`: Ürün satın alma modeli servisi.  
- `return_risk_service.py`: İade riski modeli servisi.

### 🔹 Diğer Ana Dosyalar
- `run.py`: FastAPI uygulamasını başlatan çalıştırıcı dosya.  
- `.env` / `.env.example`: Ortam değişkenleri için yapılandırma dosyaları.  
- `.gitignore`: Versiyon kontrolüne dahil edilmeyecek dosyalar.  
- `requirements.txt`: Projede kullanılan tüm Python bağımlılıkları.  
- `README.md`: Projenin tanıtım ve kurulum kılavuzu.

---

Bu yapı sayesinde:

- ✅ **Yeniden Kullanılabilirlik:** Her bileşen tek sorumluluk prensibine göre düzenlenmiştir.  
- 🔧 **Bakım Kolaylığı:** Modüler yapı sayesinde hata ayıklama ve güncelleme işlemleri izole biçimde yapılabilir.  
- 🚀 **Genişletilebilirlik:** Yeni modeller, servisler veya API uçları kolayca eklenebilir.  
- 🏗 **Üretime Uygunluk:** API ve servis yapısı, CI/CD süreçlerine kolay entegrasyon sağlar.

📌 **Detaylı proje yapısı için:**  
👉 [docs/Project_Structure.md](https://github.com/BernaUzunoglu/smartretail-ai/blob/main/docs/Project_Structure.md)


## API ÇALIŞTIRMA
Kök dizizn içerisinde 
```` bash
uvicorn src.api.app:app --reload
````


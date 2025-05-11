# 📊 Northwind Derin Öğrenme Projeleri

Bu dokümantasyon, Northwind veritabanı kullanılarak geliştirilen üç farklı derin öğrenme tabanlı tahmin modelini kapsamaktadır. Modellerin her biri, müşteri davranışlarını analiz ederek şirket stratejilerine katkı sağlamayı amaçlamaktadır. Proje sonunda her model, bir API aracılığıyla erişilebilir hale getirilmiştir.

---

## 🧠 Proje 1: Sipariş Verme Alışkanlığı Tahmini

### 🎯 Amaç
Müşterilerin toplam harcaması, sipariş sayısı ve ortalama sipariş büyüklüğüne göre önümüzdeki 6 ay içinde tekrar sipariş verip vermeyeceğini tahmin etmek.

### 🔍 Kullanılan Tablolar
- `Customers`
- `Orders`
- `Order Details`

### ⚙️ Özellik Mühendisliği
- Toplam Harcama: `SUM(UnitPrice * Quantity * (1 - Discount))`
- Sipariş Sayısı: `COUNT(OrderID)`
- Ortalama Sipariş Tutarı
- Son Sipariş Tarihi
- 6 aydan sonra sipariş verdi mi? `last_order_date` üzerinden 6 ay kuralı ile etiketleme (0: vermedi, 1: verdi) → **Hedef Etiket (0/1)**

### 🧪 Ar-Ge Konuları
- **Zamansal Özellikler (Temporal Features):** Siparişlerin ay bazında dağılımına bakılarak mevsimsellik analizi.
- **Veri Büyütme (Data Augmentation):** Müşteri profilleri farklı varyasyonlarla çoğaltılarak modelin genelleme gücü artırıldı.
- **Sınıf Dengesizliği (Class Imbalance):** Az sayıda tekrar sipariş veren müşteriler için `SMOTE` ve `class_weight` kullanıldı.

### 🧠 Kullanılan Model
- Derin sinir ağı (Dense Layers ile MLP)
- Aktivasyon: `ReLU`, Çıkış Katmanı: `Sigmoid`
- Kayıp Fonksiyonu: `Binary Crossentropy`

---

## 🔁 Proje 2: Ürün İade Risk Skoru

### 🎯 Amaç
İndirim oranı, sipariş miktarı ve harcama bilgilerine göre bir siparişin iade edilme riskini tahmin etmek.
Müşterilerin daha önceki siparişlerindeki indirim oranı, ürün miktarı ve harcama miktarına göre bir siparişin iade edilme riskini tahmin eden bir derin öğrenme modeli oluştur.

### 🔍 Kullanılan Tablolar
- `Orders`
- `Order Details`


## 📊 Özellikler (Features)

| Özellik         | Açıklama                                             |
|------------------|------------------------------------------------------|
| `discount_rate`  | Siparişte uygulanan indirim oranı (0-1 arası)       |
| `unit_price`     | Ürünün birim fiyatı                                 |
| `quantity`       | Siparişteki ürün adedi                              |
| `total_spend`    | Toplam harcama = `unit_price * quantity * (1 - discount_rate)` |

📌 **Etiketleme (Labeling):**  
- **Yüksek indirim** + **düşük harcama** → “İade riski yüksek” kabul edilir.

### 🧪 Ar-Ge Konuları
- **Cost-sensitive Learning:** Modelin iade sınıfına daha fazla önem vermesi için sınıf ağırlıkları belirlendi.
- **XAI (Explainable AI):** `SHAP` ile modelin neden bir siparişi riskli gördüğü açıklanabilir hale getirildi.

### 🧠 Kullanılan Model
- Derin sinir ağı (MLP)
- Ağırlıklı Binary Crossentropy Loss
- SHAP analizleriyle görsel açıklamalar

---

## 🆕 Proje 3: Yeni Ürün Satın Alma Potansiyeli

### 🎯 Amaç
Müşterilerin geçmişteki ürün kategorisi tercihlerini analiz ederek yeni bir ürünü satın alma olasılıklarını tahmin etmek.

### 🔍 Kullanılan Tablolar
- `Orders`
- `Order Details`
- `Products`
- `Categories`

### ⚙️ Özellik Mühendisliği
- Müşteri başına kategori bazlı harcama
- Kategori çeşitliliği ve toplam harcama oranları
- Yeni ürünü önerilecek kategoriye ait mi?

### 🧪 Ar-Ge Konuları
- **Öneri Sistemleri:** Neural Collaborative Filtering denemeleri
- **Multi-label Prediction:** Aynı anda birden fazla ürün önerimi
- **AutoEncoder ile Müşteri Temsil Öğrenimi**

### 🧠 Kullanılan Model
- Derin öğrenme tabanlı öneri sistemi (NNCF)
- Çok etiketli `sigmoid` çıkış ve `binary crossentropy` loss

---

## 🖥️ API Servisleri

Her model ayrı bir REST API servisi olarak sunulmuştur:

| Model | Endpoint | Method | Girdi Formatı | Açıklama |
|-------|----------|--------|----------------|----------|
| Sipariş Tahmini | `/predict-order` | POST | JSON | Müşterinin tekrar sipariş verip vermeyeceğini tahmin eder |
| İade Riski | `/predict-return-risk` | POST | JSON | Siparişin iade edilip edilmeyeceğini tahmin eder |
| Yeni Ürün Potansiyeli | `/predict-new-product` | POST | JSON | Müşterinin yeni ürünü satın alıp almayacağını tahmin eder |

### Örnek Girdi Formatı (JSON)
```json
{
  "total_spent": 1200,
  "order_count": 5,
  "avg_order_value": 240
}
````

---

## 🧪 Değerlendirme Metrikleri

* **Accuracy**
* **Precision / Recall / F1-score**
* **AUC-ROC**
* **Confusion Matrix**
* **Explainability (SHAP visualizations)**

---

## 🛠️ Kullanılan Teknolojiler

* Python, Pandas, Scikit-learn
* TensorFlow / Keras
* FastAPI (API Servisleri)
* SQLite (Northwind DB)
* SHAP, SMOTE (imbalanced-learn), matplotlib

---

## 📁 Proje Yapısı



---

## 🔚 Sonuç ve Katkı Alanları

Bu projeler, küçük ama yapısal bir veritabanı olan Northwind üzerinde uygulanabilir derin öğrenme uygulamaları geliştirmek isteyen araştırmacılar ve geliştiriciler için rehber niteliğindedir. Ayrıca, gerçek dünyadaki pazarlama, stok yönetimi ve müşteri analitiği gibi iş senaryolarına kolayca uyarlanabilir.


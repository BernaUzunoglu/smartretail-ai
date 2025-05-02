# 📦 Northwind Derin Öğrenme Projeleri – API Formatında

## 1. 🔁 Sipariş Verme Alışkanlığı Tahmini

### 📌 Problem:
Bir müşterinin önümüzdeki 6 ay içinde tekrar sipariş verip vermeyeceğini tahmin et.

### 🗂️ Kullanılan Tablolar:
- `Customers`
- `Orders`
- `Order_Details`

### 📊 Özellikler (Features):
- `total_spend` = toplam harcama
- `order_count` = verilen sipariş sayısı
- `avg_order_value` = ortalama sipariş tutarı
- `last_order_date` üzerinden 6 ay kuralı ile etiketleme (0: vermedi, 1: verdi)

### 🔬 Ar-Ge Konuları:
- **Temporal Features:** Sipariş tarihinden ay, mevsim gibi bilgiler çıkar. Mevsimsellik etkisi var mı? (Örn: Yaz aylarında sipariş artıyor mu?)
- **Data Augmentation:** Müşteri datasını arttırarak daha büyük bir veri seti oluşturup modelin başarısını gözlemle.
- **Class Imbalance:** SMOTE, class_weight , Eğer az kişi sipariş veriyorsa, class_weight veya SMOTE gibi yöntemlerle çözüm üret.

### 🧠 Model:
- Giriş: `[total_spend, order_count, avg_order_value, month, season]`
- Çıkış: `[0, 1]` (sipariş verdi/vermedi)

### 🛠️ API Endpoint:
```http
POST /api/prediction/order-return-probability
```

#### 🧾 Örnek Gönderi:
```json
{
  "customerId": "ALFKI",
  "totalSpend": 1200,
  "orderCount": 5,
  "avgOrderValue": 240,
  "lastOrderMonth": 7,
  "lastOrderSeason": "Summer"
}
```

#### ✅ Örnek Yanıt:
```json
{
  "willReorderNext6Months": true,
  "probability": 0.78
}
```

---

## # ⚠️ Ürün İade Risk Skoru

## 📌 Problem Tanımı  
Bir siparişin iade edilme riskini tahmin et.  
Müşterilerin daha önceki siparişlerindeki **indirim oranı**, **ürün miktarı** ve **harcama miktarına** göre bir siparişin iade edilme riskini tahmin eden bir **derin öğrenme modeli** oluştur.

---

## 🗂️ Kullanılan Veri Tabloları  
- `Order_Details`

---

## 📊 Özellikler (Features)

| Özellik         | Açıklama                                             |
|------------------|------------------------------------------------------|
| `discount_rate`  | Siparişte uygulanan indirim oranı (0-1 arası)       |
| `unit_price`     | Ürünün birim fiyatı                                 |
| `quantity`       | Siparişteki ürün adedi                              |
| `total_spend`    | Toplam harcama = `unit_price * quantity * (1 - discount_rate)` |

📌 **Etiketleme (Labeling):**  
- **Yüksek indirim** + **düşük harcama** → “İade riski yüksek” kabul edilir.

---

## 🔬 Ar-Ge Konuları  

### ✅ Cost-sensitive Learning
- `class_weight`: **İade edilen ürünlerin maliyeti daha yüksek** olduğundan, model bu sınıfı daha ciddi şekilde değerlendirmeli.
- **Amaç:** İade edilen ürünlere yanlış tahmin yapmanın cezasını artırmak.

### ✅ Explainable AI
- **SHAP / LIME** gibi açıklanabilir yapay zeka yöntemleri kullanılacak.
- **Amaç:** "Model neden bu siparişi riskli buldu?" sorusuna yanıt vermek.

---

## 🧠 Model Özeti  

- **Girdi (Input):**
  - `discount`
  - `quantity`
  - `total_spend`

- **Çıktı (Output):**
  - `risk_score` (0 ile 1 arasında iade riski)

---

## 🛠️ API Endpoint  

```http
POST /api/prediction/return-risk
````

# Örnek Gönderi (Request Body)
```json
{
  "orderId": 10248,
  "discount": 0.15,
  "quantity": 5,
  "unitPrice": 20
}
````
# Örnek Yanıt (Response)
```json
{
  "returnRiskScore": 0.83,
  "explanation": "High discount with low total spend indicates likely return."
}

```

---

## 3. 🛍️ Yeni Ürün Satın Alma Potansiyeli

### 📌 Problem:
Bir müşterinin yeni çıkan ürünü satın alma olasılığını tahmin et.
Müşterilerin geçmiş satın alma kategorilerine (örneğin "Beverages", "Confections") bakarak, yeni çıkan bir ürünü satın alma ihtimallerini tahmin eden bir sinir ağı modeli geliştir.

### 🗂️ Kullanılan Tablolar:
- `Customers`
- `Orders`
- `Order_Details`
- `Products`
- `Categories`

### 📊 Özellikler (Features):
Müşterinin hangi kategorilerde ne kadar harcama yaptığı gibi özellikler üret.

- Müşterinin her kategoriye yaptığı toplam harcama
- Sık alışveriş yaptığı kategori
- Yeni ürünün kategorisi

### 🔬 Ar-Ge Konuları:
- **Recommendation Systems:** Neural Collaborative Filtering , Deep Learning tabanlı ürün öneri sistemleri araştır (örneğin Neural Collaborative Filtering, AutoEncoders).
- **Multi-label Prediction:** Aynı anda birden fazla yeni ürün önerimi , Aynı anda birkaç ürünü birden önerebilecek bir sistem geliştir.

### 🧠 Model:
- Giriş: `[beverages_spend, dairy_spend, ..., target_category_id]`
- Çıkış: `[purchase_probability: 0-1]`

### 🛠️ API Endpoint:
```http
POST /api/recommendation/new-product
```

#### 🧾 Örnek Gönderi:
```json
{
  "customerId": "VINET",
  "spendingByCategory": {
    "Beverages": 500,
    "Confections": 200,
    "Produce": 100
  },
  "targetCategory": "Beverages"
}
```

#### ✅ Örnek Yanıt:
```json
{
  "purchaseProbability": 0.69,
  "recommended": true
}
```

---
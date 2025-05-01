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
- **Temporal Features:** Sipariş tarihinden ay, mevsim gibi bilgiler çıkar.
- **Data Augmentation:** Kopyalanmış ve hafif bozulmuş müşteri verileriyle veri çoğaltma
- **Class Imbalance:** SMOTE, class_weight

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

## 2. ⚠️ Ürün İade Risk Skoru

### 📌 Problem:
Bir siparişin iade edilme riskini tahmin et.

### 🗂️ Kullanılan Tablolar:
- `Order_Details`

### 📊 Özellikler (Features):
- `discount_rate`
- `unit_price`
- `quantity`
- `total_spend = unit_price * quantity * (1 - discount_rate)`
- Etiketleme: Eğer yüksek indirim + düşük harcama → “iade riski yüksek” kabul et.

### 🔬 Ar-Ge Konuları:
- **Cost-sensitive Learning:** `class_weight` = iade edilenlerin kaybı daha yüksek
- **Explainable AI:** SHAP ile karar açıklamaları

### 🧠 Model:
- Giriş: `[discount, quantity, total_spend]`
- Çıkış: `[risk_score: 0-1]`

### 🛠️ API Endpoint:
```http
POST /api/prediction/return-risk
```

#### 🧾 Örnek Gönderi:
```json
{
  "orderId": 10248,
  "discount": 0.15,
  "quantity": 5,
  "unitPrice": 20
}
```

#### ✅ Örnek Yanıt:
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

### 🗂️ Kullanılan Tablolar:
- `Customers`
- `Orders`
- `Order_Details`
- `Products`
- `Categories`

### 📊 Özellikler (Features):
- Müşterinin her kategoriye yaptığı toplam harcama
- Sık alışveriş yaptığı kategori
- Yeni ürünün kategorisi

### 🔬 Ar-Ge Konuları:
- **Recommendation Systems:** Neural Collaborative Filtering
- **Multi-label Prediction:** Aynı anda birden fazla yeni ürün önerimi

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
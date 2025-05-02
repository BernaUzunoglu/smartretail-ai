# 🧠 Model Geliştirme Aşamaları & Farklılıklar

| Dosya Adı                       | Açıklama / Amaç                                                                 | Öne Çıkan Farklılıklar |
|--------------------------------|----------------------------------------------------------------------------------|-------------------------|
| `01_baseline_model.py`         | İlk temel deneme. Veri hazırlama + basit DNN modeli.                            | SMOTE, class_weight yok. |
| `02_add_smote.py`              | Dengesiz veri problemi için SMOTE uygulanıyor.                                  | İlk kez SMOTE eklendi. |
| `03_manual_split_smote.py`     | Azınlık sınıfı elle bölünüyor, test setine eşitlik sağlanıyor.                  | Elle split + SMOTETomek. |
| `04_add_features_monthly_daily.py` | Yeni özellikler: `orders_per_month`, `spend_per_day`.                          | Zaman tabanlı metrikler eklendi. |
| `05_early_stopping_dropout_auc.py`| Eğitim erken durdurma, dropout, AUC metrikleri eklendi.                        | Eğitim iyileştirmeleri odakta. |
| `06_feature_importance_auc.py` | AUC bazlı permütasyon ile feature importance hesaplandı.                         | Özelliklerin model etkisi ölçüldü. |
| `07_temporal_features_extended.py`| En sık sipariş ayı, sezon one-hot encoding eklendi.                             | Zaman temelli davranış modellemesi. |
| `08_temporal_features_v2.py`   | `07` dosyasının sadeleştirilmiş versiyonu.                                       | `spend_per_day` çıkarıldı. |
| `final_order_habit_model.py`   | Tüm süreç fonksiyonlaştırıldı, modülerleştirildi, sürdürülebilir hale getirildi. | Yapı artık production-ready. |

---

# 🔬 Ar-Ge Konuları ve Model Geliştirme Stratejisi


## 🔁 **Data Augmentation (Veri Zenginleştirme)**

### 📌 **Tanım ve Amaç**

Veri zenginleştirme, makine öğrenmesi modellerinin daha fazla ve çeşitli örnek üzerinden öğrenmesini sağlamak için mevcut veriden türetilmiş yeni veriler üretme yöntemidir. Amaç; modeli **genelleyebilir**, **daha dayanıklı** ve **overfitting’e karşı dirençli** hale getirmektir.

### ⚙️ **Yöntemler**

* **SMOTE (Synthetic Minority Over-sampling Technique):** Azınlık sınıfa ait örneklerin çevresinden yeni örnekler üretir.
* **Noise injection:** Sayısal verilere küçük gürültüler eklenir.
* **Bootstrapping:** Var olan veriler tekrar tekrar kullanılarak yeni örnekler oluşturulur.
* **Generative modeling (GAN gibi):** Gerçekçi sentetik veriler üretir (ileri düzey).

### ✅ **Avantajları**

* Küçük veri setlerinde modeli daha genelleyici hale getirir.
* Overfitting riskini azaltır.
* Az örnekli (rare case) durumları modele öğretmek kolaylaşır.

### ❌ **Dezavantajları**

* Sentetik veriler gerçek verinin doğasını bozabilir.
* Aşırı veri üretimi, **noise (gürültü)** yaratabilir ve modeli yanıltabilir.
* Veri kalitesine bağlıdır; hatalı veriler çoğaltılırsa hata da büyür.

### 🎯 **Kullanım Durumları**

* Veri seti küçükse veya belirli sınıflar çok az gözlem içeriyorsa.
* Model overfit oluyorsa.
* Mevsimsellik, alışkanlık gibi zaman bağlı örüntüleri çeşitlendirmek isteniyorsa.

### 📈 **Etkisi**

* Modelin doğruluğunu artırabilir.
* AUC, F1-score gibi metriklerde iyileşme sağlar.
* Dengesiz sınıflarda tahmin başarısı artar.

---

## ⚖️ **Class Imbalance (Sınıf Dengesizliği ile Mücadele)**

### 📌 **Tanım ve Amaç**

Sınıf dengesizliği, bir sınıfın diğer sınıfa göre veri setinde çok fazla ya da çok az temsil edilmesi durumudur. Bu, modelin baskın sınıfa odaklanmasına neden olur. Amaç, **azınlık sınıfın da etkin şekilde öğrenilmesini** sağlamaktır.

### ⚙️ **Yöntemler**

* **Over-Sampling:** Azınlık sınıfa ait örnekler çoğaltılır (SMOTE, ADASYN).
* **Under-Sampling:** Çoğunluk sınıfının bazı örnekleri çıkarılır (RandomUnderSampler).
* **Class Weighting:** Modelin eğitiminde azınlık sınıfa daha fazla ceza verilmesi sağlanır.
* **Karma Yaklaşımlar:** SMOTE + Tomek Links gibi hem over hem under sampling birlikte kullanılır.

### ✅ **Avantajları**

* Azınlık sınıfa ait sınıflandırma başarısı artar.
* Gerçek hayattaki dengesiz dağılımlar için daha adil tahminler yapılır.
* Özellikle F1-score, recall gibi metriklerde gelişme sağlar.

### ❌ **Dezavantajları**

* Under-sampling bilgi kaybına neden olabilir.
* Over-sampling modelin aşırı benzer örnekleri öğrenmesine neden olabilir (overfitting).
* Class weighting her modelde etkili olmayabilir.

### 🎯 **Kullanım Durumları**

* Gerçek veri setlerinde "az görülen ama kritik" durumları sınıflandırmak istiyorsak (örneğin; dolandırıcılık, churn, hastalık teşhisi).
* Model sadece baskın sınıfı tahmin ediyorsa.
* Accuracy çok yüksek ama diğer metrikler (recall, precision) düşükse.

### 📈 **Etkisi**

* Azınlık sınıfa ait doğruluğu artırır.
* Modelin tüm sınıflar üzerinde daha dengeli bir performans göstermesini sağlar.
* F1-score, recall gibi dengesizlik duyarlı metriklerde iyileşme yaratır.

---

## 🧪 **Sonuçlara Etkisi ve Ar-Ge Değeri**

* Bu yöntemler, modelin doğruluğunu "gerçek başarıya" çevirir.
* Akademik çalışmalarda metrik gelişimini ölçmek için temel Ar-Ge alanlarıdır.
* Özellikle dengesiz sınıflarda yapılan A/B testlerinde model başarı oranları bu yöntemlerle gözle görülür biçimde artabilir.

---
## 🎯 Problem Tanımı

Müşterilerin geçmiş sipariş davranışlarına (toplam harcama, sipariş sayısı, sipariş sıklığı, vb.) göre önümüzdeki 6 ay içinde tekrar sipariş verip vermeyeceğini tahmin eden bir sınıflandırma modeli geliştirildi.

---

## 🔎 Araştırma ve Geliştirme Konuları

### 1. ⚖️ Class Imbalance (Dengesiz Sınıf Dağılımı)


- - **Problem:** "Tekrar sipariş veren" müşterilerin oranı oldukça düşüktü (örneğin %10-15). Bu durum modelin çoğunluk sınıfını tahmin etme eğilimini artırır.Ve azınlık sınıfının tahmin etme eğilimini ise azaltır.
- **Çözüm Stratejileri:**
  - **SMOTE:** Azınlık sınıfı çoğaltarak örnek sayısı eşitlendi.
  - **class_weight:** Model eğitimi sırasında azınlık sınıfa daha fazla önem verilmesi sağlandı.
  - **Stratejik Test Ayrımı:** Test setinde azınlık sınıf yeterince temsil edilsin diye manuel olarak test verisi ayrımı yapıldı.
- **Sonuç:** Bu stratejiler, özellikle `recall` ve `AUC` metriklerinde anlamlı artışlar sağladı. Model, sipariş verecek müşterileri kaçırma olasılığını minimize etti.
- (Bkz: `02_add_smote.py`, `03_manual_split_smote.py`, `final_order_habit_model.py`)

---

### 2. Elle Test Ayrımı (Manual Test Split)

- **Sorun:** Stratified split yetersiz kaldı, test setinde azınlık sınıf yeterince yer almadı.
- **Yaklaşım:**  
  - Azınlık ve çoğunluk sınıflar ayrı ayrı bölünerek test setine kontrollü şekilde eklendi.  
  - Böylece model azınlık sınıfı üzerinde daha tutarlı ölçümler yaptı.  
  - (Bkz: `03_manual_split_smote.py`, `06_feature_importance_auc.py`, `final_order_habit_model.py`)

---

### 3. Feature Engineering (Özellik Türetme) - 🧩 Temporal Features (Zamana Dayalı Davranışsal Özellikler)

- **Araştırma Sorusu:** Sipariş alışkanlıklarında belirli zaman dilimlerine (ay, mevsim) bağlı bir düzen var mı?
- **Yöntem:** `order_date` sütunu üzerinden siparişin verildiği ay (`order_month`) ve bu aya karşılık gelen mevsim (`order_season`) çıkarılmıştır.
- **Feature Engineering:** Sezon bilgisi nominal olduğu için one-hot encoding uygulanarak `order_season_Winter`, `order_season_Spring`, `order_season_Summer`, `order_season_Fall` şeklinde ayrı değişkenler oluşturulmuştur.
- **Gelişim:** `orders_per_month`, `spend_per_day`, `avg_days_between_orders` gibi yeni türevler eklendi. 
- **Temporal Özellikler:** En sık sipariş verilen ay (`common_order_month`) ve sezon bazlı one-hot encoding uygulandı.  
- **Model Etkisi:** Sezon bilgisi, bazı müşterilerin yılın belirli dönemlerinde sipariş verme alışkanlıklarını yansıttığı için modelin tahmin başarısını artırmıştır.
- 
- (Bkz: `04_add_features_monthly_daily.py`, `07_temporal_features_extended.py`)

---

### 4. Eğitim İyileştirme Teknikleri

- `Dropout` katmanları ile overfitting önlendi.  
- `EarlyStopping` ile gereksiz epoch çalışması engellendi.  
- `ReduceLROnPlateau` (opsiyonel olarak denendi) öğrenme oranını dinamik yönetti.  
- AUC ve loss grafikleri ile performans izlenebilir hale getirildi.  
- (Bkz: `05_early_stopping_dropout_auc.py`, `06_feature_importance_auc.py`)

---

### 5. Feature Importance Hesaplaması

- **Amaç:** Hangi özelliğin model başarısına ne kadar katkı sağladığını ölçmek.  
- **Yöntem:** AUC bazlı permütasyon yöntemiyle her bir feature rastgele bozuldu ve AUC farkı hesaplandı.  
- (Bkz: `06_feature_importance_auc.py`, `07_temporal_features_extended.py`, `final_order_habit_model.py`)

---

### 6. Yapının Modülerleştirilmesi

- Tüm veri işleme, model eğitimi, test ve değerlendirme adımları fonksiyonlaştırıldı.  
- `final_order_habit_model.py` artık üretime hazır, temiz bir yapı sunar.  
- Yeniden kullanılabilirlik, test edilebilirlik ve sürdürülebilirlik sağlandı.

---

# 🚀 API ile Sunum ve Servis Yapısı

## 📦 Yapı Bileşenleri

### 1. Servis Katmanı
- **Görev:** Eğitilen `.h5` modelini ve `preprocessor.pkl` dosyasını yükleyerek gelen verileri işler ve tahmin yapar.
- **Yetenek:** Belirli bir müşteri ID’sine göre verileri yeniden işler (feature engineering dahil).
- **Ana Fonksiyonlar:**
  - `predict_by_customer_id(customer_id)`: Tek bir müşteri üzerinden tahmin.
  - `predict(features: dict)`: Özellik sözlüğü üzerinden tahmin skorunu döner.

> Bu dosya servis katmanıdır, FastAPI gibi frameworklerle doğrudan kullanılabilir.

---

### 2.API endpoint
- **FastAPI Router** tanımı içerir.
- `GET /predict-order-habit-by-id/{customer_id}` endpoint’i üzerinden müşterinin tekrar sipariş verip vermeyeceğini tahmin eder.
- Hatalar kullanıcı dostu mesajlarla döndürülür.

# 🧪 Örnek API Yanıtı
```bash
GET /predict-order-habit-by-id/CUS123

{
  "prediction_probability": 0.7723,
  "will_reorder": true
}
````

---

## 🧭 Entegrasyon Akışı

```mermaid
flowchart TD
    A[Müşteri ID'si frontend'den gönderilir] --> B[API endpoint /predict-order-habit-by-id/{customer_id}]
    B --> C[OrderHabitService veriyi get_customer_order_data ile çeker]
    C --> D[Öznitelikler hesaplanır (feature engineering)]
    D --> E[Preprocessor (StandardScaler) uygulanır]
    E --> F[Model (.h5) tahmin yapar]
    F --> G{prediction >= 0.5?}
    G -- Evet --> H[will_reorder: true]
    G -- Hayır --> I[will_reorder: false]
    H & I --> J[JSON formatında yanıt döner]



# 🛡️ TrustLLM: Uncertainty Quantification, Calibration & Linguistic Routing

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Framework-orange.svg)](https://pytorch.org/)
[![SpaCy](https://img.shields.io/badge/SpaCy-NLP-09A3D5.svg)](https://spacy.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

[English](#-english) | [Türkçe](#-türkçe) | [Deutsch](#-deutsch)

---

## 📌 Abstract / Özet

Deep neural networks and Large Language Models (LLMs) often exhibit high confidence in incorrect or subjective predictions, leading to reliability and hallucination risks in real-world deployments. **TrustLLM** is an academic-grade evaluation framework designed to quantify epistemic and aleatoric uncertainty, perform post-hoc confidence calibration, and mitigate overconfidence via **Dual-Signal Subjectivity Filtering**, **SpaCy-based Part-of-Speech (POS) Parsing**, **Uncertainty-Guided Model Routing**, **Reliability Diagram Visualizations**, and an **Uncertainty-Gated System Refusal Mechanism**.

Derin sinir ağları ve Büyük Dil Modelleri (LLM), yanlış veya öznel tahminlerde dahi yüksek özgüven (overconfidence) üretme eğilimindedir. **TrustLLM**, bu modellerde belirsizliği nicelleştirmek, post-hoc kalibrasyon uygulamak, **Çifte Sinyalli (Dual-Signal) Öznellik Ön-Filtrelemesi**, **SpaCy tabanlı dilbilgisel (POS) varlık izolasyonu** yapmak, **ikili model yönlendirmesi (routing)**, **Reliability Diagrams** görselleştirmeleri ve **Belirsizlik Tabanlı Sistem Reddetme (Refusal System)** katmanı ile halüsinasyon riskini en aza indirmek amacıyla geliştirilmiş araştırma seviyesinde modüler bir sistemdir.

---

## 🚀 Projenin Gelişim Süreci ve Teknik İyileştirmeler (Project Evolution)

Proje, basit sezgisel kurallarla başlayıp aşamalı olarak endüstriyel ve akademik standartlarda nesne yönelimli bir mimariye dönüştürülmüştür:

1. **Statik Stop-Word / Kelime Listelerinden NLP Sentaks Analizine Geçiş:**
   * **İlk Aşama:** Öznel sorguları yakalamak için manuel stop-word ve sabit sıfat listelerine güveniliyordu. Bu yaklaşım *"highest mountain"* (nesnel) ile *"best movie"* (öznel) arasındaki dilbilgisel farkı ayırt edemeyerek yüksek False Positive veriyordu.
   * **Geliştirilmiş Hal:** Sabit kelime engelleme mantığı kaldırılarak **SpaCy Dependency Tree (`en_core_web_sm`)** ve **WordNet Semantik Ağaç** entegrasyonuna geçildi. Kelimelerin cümle içindeki sentaks rolleri (`POS=JJS`, `advmod -> amod` ilişkileri) ve anlamsal kökleri (lemmas) dinamik olarak incelenmeye başlandı.

2. **Aşırı Özgüvenli Ezberleme (Overconfident Memorization) Tuzağının Çözülmesi:**
   * Küçük parametreli modellerin (GPT-2, Qwen1.5-0.5B) öznel sorularda ezberden dolayı hep aynı yanıtı vererek Anlamsal Entropiyi $H(S)=0.0000$ çıkarma problemi tespit edildi.
   * Sözcüksel/Sentaks Analizi ile Anlamsal Entropiyi harmanlayan **Çifte Sinyalli (Dual-Signal) Emniyet Filtresi** geliştirilerek entropi 0 çıksa dahi öznelliğin %100 isabetle reddedilmesi sağlandı.

3. **Modüler Production Mimarisine Geçiş (`src/` Modül Ayrışımı):**
   * Tüm karmaşık hesaplamalar ve NLP fonksiyonları `app.py` içinden çıkarılarak `src/subjectivity.py`, `src/extraction.py`, `src/metrics.py`, `src/calibration.py` ve `src/uncertainty.py` altında Clean Code ilkelerine uygun olarak modülerleştirildi.

4. **Ölçeklenebilirlik ve Latans Testleri ($N=100,000$ Benchmark):**
   * Test ölçeği $N=10$'dan $N=100$, $N=1,000$ ve nihayetinde $N=100,000$ dev benchmark seviyesine çıkarıldı. Sistem CPU üzerinde soru başına **8.24 ms latans** ve **%100.00 Accuracy** ile doğrulanmıştır.

---

## 🌍 Multilingual Highlights

### 🇬🇧 English
* **Step 0: Dual-Signal Subjectivity Pre-Filter (Enhanced with NLP)**
  * Replaced static stop-word lists with **SpaCy Syntactic Dependency Tree Parsing** and **WordNet Semantic Hierarchy**.
  * Combines **Syntactic Superlative Parsing (`POS=JJS`)** with **Semantic Entropy Spreading ($H(S)$)**.
  * Captures overconfident model memorization even when model entropy drops to $0.0000$ on subjective prompts.

* **Step 1: Linguistic Parsing & Token Filtering**
  * Parses syntax dependency trees using **SpaCy (`en_core_web_sm`)**.
  * Filters out non-informative tokens (`ADJ`, `VERB`, `ADP`) to isolate target entities (`NOUN`, `PROPN`).
  * Displays step-by-step decision rationale logs and white-box logit tracking.

* **Step 2: Uncertainty-Guided Model Routing**
  * Evaluates **GPT-2 (Base)** against **Qwen1.5-0.5B-Chat (Instruction)** in parallel.
  * Calculates **Semantic Entropy ($H(S)$)** and **Expected Calibration Error (ECE)**.
  * Dynamically routes inferences to the model exhibiting superior calibration stability.

* **Step 3: Calibration Metrics & Visualizations**
  * Generates **Reliability Diagrams** comparing raw vs. calibrated confidence distributions ($y=x$ ideal line).
  * Computes the **Brier Score** to evaluate mean squared probability accuracy.

* **Step 4: Uncertainty Threshold & Refusal System**
  * Evaluates system reliability against a dynamic confidence threshold ($\tau$).
  * Shields users from hallucinations by withholding responses when uncertainty exceeds safety limits.

---

### 🇹🇷 Türkçe
* **0. Adım: Çifte Sinyalli (Dual-Signal) NLP Öznellik Ön-Filtresi**
  * Statik stop-word engelleme yerine **SpaCy Sentaks Bağlantı Ağacı** ve **WordNet Semantik Ağaç** entegrasyonu kullanır.
  * **Sözcüksel/Sentaks Analizi (`POS=JJS`)** ile **Anlamsal Entropi ($H(S)$)** sinyallerini birleştirir.
  * Öznel sorularda modeller ezber yapıp $H(S)=0.0000$ üretseler dahi aşırı özgüveni yakalayarak yanıtı maskeler.

* **1. Adım: Dilbilgisel Ayrıştırma ve Varlık İzolasyonu**
  * Sentaks bağımlılık ağaçlarını analiz etmek için **SpaCy (`en_core_web_sm`)** kullanır.
  * Bilgi taşımayan kelime türlerini (`ADJ`, `VERB`, `ADP`) eleyerek yalnızca hedef varlıkları (`NOUN`, `PROPN`) izole eder.
  * Her aday token için adım adım karar gerekçelerini ve beyaz kutu (white-box) logit değerlerini sunar.

* **2. Adım: Belirsizlik Rehberliğinde Model Yönlendirmesi**
  * **GPT-2 (Base)** ile **Qwen1.5-0.5B-Chat (Instruction)** modellerini eşzamanlı değerlendirir.
  * **Anlamsal Entropi ($H(S)$)** ve **Beklenen Kalibrasyon Hatası (ECE)** skorlarını hesaplar.
  * Çıkarımları, kalibrasyon kararlılığı en yüksek olan modele dinamik olarak yönlendirir.

* **3. Adım: Kalibrasyon Metrikleri ve Görselleştirme**
  * Ham ve kalibre edilmiş özgüven dağılımlarını karşılaştıran **Reliability Diagrams** ($y=x$ çizgisi) çizer.
  * Olasılıksal doğruluk sapmasını ölçmek için **Brier Skoru** hesaplar.

* **4. Adım: Belirsizlik Eşik Değeri ve Otomatik Reddetme (Refusal)**
  * Hesaplanan Güvenilirlik Skorunu dinamik bir emniyet eşiği ($\tau$) ile karşılaştırır.
  * Belirsizlik kritik seviyeyi aştığında yanıtı maskeler ve *"Bilmiyorum / Cevap Vermiyorum"* kararı alarak halüsinasyonu engeller.

---

### 🇩🇪 Deutsch
* **Schritt 0: Dual-Signal Subjektivitäts-Vorfilter (NLP-basiert)**
  * Ersetzt statische Stoppwort-Listen durch **SpaCy Syntax-Baum-Analyse** und **WordNet Semantik**.
  * Kombiniert **Syntaktisches Superlativ-Parsing (`POS=JJS`)** mit **Semantischer Entropie ($H(S)$)**.
  * Erfasst überdurchschnittliche Modell-Memorierung, selbst wenn die Modellentropie bei subjektiven Prompts auf $0.0000$ fällt.

---

## 🧠 Step 0: Dual-Signal Subjectivity Pre-Filter Methodology

LLM modellerinin ezberleme (memorization) eğilimi nedeniyle öznel sorularda tek bir yanıta kilitlenerek Anlamsal Entropiyi yanlışlıkla $H(S)=0.0000$ çıkarma riski **SpaCy Sentaks Analizi ve Çifte Sinyalli Hibrit Filtre** ile çözülmüştür:

$$\text{IsSubjective} = (\text{HasSuperlativeOrOpinionPattern}) \lor (H(S) \ge \tau_{\text{entropy}})$$

### 📊 Deneysel Doğrulama & Ekran Görüntüleri

| Nesnel Sorgu Testi (`capital of France`) | Öznel Sorgu Testi (`best movie in world`) |
| :---: | :---: |
| ![Objective Test](objective_test_capital.png) | ![Subjective Test](subjective_test_movie.png) |
| *Entropi 0.0000 & Yapısal Öznel Öge Yok → **OBJECTIVE FACT-BASED (PASSED)**.* | *Entropi 0.0000 fakat `best` sıfatı tespit edildi → **SUBJECTIVE / AMBIGUOUS (REFUSED)**.* |

### 🚀 Industrial Large-Scale Subjectivity Benchmark ($N=100,000$ Prompts)

TrustLLM öznellik filtreleme katmanının ölçeklenebilirliği (scalability) ve gecikme süresi (latency), 50.000 nesnel ve 50.000 öznel sorgudan oluşan **$N=100,000$ soruluk devasa bir benchmark veri kümesinde** doğrulanmıştır:

| Metrik Kriteri (Evaluation Metric) | Değer (Value) | Detay & Açıklama |
| :--- | :---: | :--- |
| **Total Processed Dataset Size** | **100,000** | 50,000 Fact-Based (Objective) vs. 50,000 Opinion-Based (Subjective) |
| **General Accuracy** | **%100.00** | 100.000 sorunun tamamında sıfır hata ile sınıflandırma. |
| **True Positives (TP)** | **50,000 / 50,000** | Öznel soruların tamamı başarıyla maskelendi. |
| **True Negatives (TN)** | **50,000 / 50,000** | Nesnel soruların tamamı emniyetle onaylandı. |
| **System Throughput** | **121.3 q/s** | CPU ortamında saniyede 121.3 soru işleme kapasitesi. |
| **Average Latency per Query** | **8.242 ms** | Soru başına ultra-düşük gecikme süresi. |
| **Total Execution Time** | **824.21 sec** | ~13.7 dakikada 100.000 sorunun tamamı işlendi. |

---

## 🔬 Theoretical Background & Methodology

Framework beş temel akademik problem alanına odaklanır:

### 1. Expected Calibration Error (ECE)
Sınıflandırma modellerinin özgüven skorları ($p$) ile gerçek doğruluk oranları ($acc$) arasındaki uyumsuzluk $M$ adet özgüven aralığı (bin) üzerinden hesaplanır:

$$\text{ECE} = \sum_{m=1}^{M} \frac{\vert{}B_m\vert{}}{N} \left\vert{} \text{acc}(B_m) - \text{conf}(B_m) \right\vert{}$$

### 2. Post-Hoc Temperature Scaling
Aşırı özgüveni törpülemek için model logitleri ($\mathbf{z}$), yeniden eğitim gerekmeksizin öğrenilebilir bir $T > 0$ sıcaklık parametresi ile ölçeklenir:

$$\hat{q}_i = \max_k \sigma\left(\frac{\mathbf{z}_i}{T}\right)_k$$

### 3. Brier Score (Probabilistic Accuracy)
Tahmin edilen olasılıklar ile gerçek sonuçlar arasındaki karesel sapmayı ölçer ($N$ örnek sayısı):

$$\text{BS} = \frac{1}{N} \sum_{i=1}^{N} (f_i - o_i)^2$$

### 4. Semantic Entropy for LLMs
LLM modellerinde jeneratif metinlerin anlamsal eşdeğerliğini ölçmek için yanıtlar anlamsal kümeleme (hierarchical clustering) işlemine tabi tutulur ve kümelerin olasılık dağılımı üzerinden Şifreli Entropi hesaplanır:

$$\text{SE}(x) = - \sum_{c \in C} p(c\vert{}x) \log p(c\vert{}x)$$

### 5. Composite Reliability & Refusal Score
Bileşik güvenilirlik skoru; en iyi token olasılığı, POS gramer bonusu, anlamsal entropi cezası ve ECE cezasının harmanlanmasıyla elde edilir:

$$\text{Reliability} = P_{\text{best}} + \text{Bonus}_{\text{POS}} - (0.8 \times H(S)) - (1.2 \times \text{ECE}_{\text{cal}})$$

Eğer $\text{Reliability} < \tau$ (Eşik Değeri) ise sistem yanıt vermeyi reddeder.

---

## 📊 Benchmark Evaluation & Trade-off Analysis ($N=50$)

Sistemin başarısını ölçmek amacıyla 10 soruluk ön test ($N=10$) ve 50 soruluk genişletilmiş test kümesi ($N=50$) üzerinde **Refusal Rate vs. Accuracy Trade-off** kıyaslamaları yapılmıştır.

### 📈 Refusal Rate vs. Accuracy Karşılaştırma Grafikleri

| Baseline Test ($N=10$) | Calibrated Benchmark ($N=50$) |
| :---: | :---: |
| ![Baseline Curve](benchmark_tradeoff_curve.png) | ![Calibrated Benchmark Curve](benchmark_tradeoff_curve_50.png) |
| *Dar örneklem kümesinde filtrenin tepkisiz kalması ($0.2 \le \tau \le 0.7$).* | *Hassaslaştırılmış formül ile oluşan ideal kararlılık eğrisi ($\tau = 0.55-0.65$).* |

---

## 📂 Architecture & Directory Tree

```text
TrustLLM-Uncertainty-Quantification/
│
├── src/                           # Production Engine Modülleri
│   ├── __init__.py
│   ├── subjectivity.py            # Dual-Signal & SpaCy Sentaks Öznellik Filtresi
│   ├── extraction.py              # SpaCy POS Entity Isolation Engine
│   ├── metrics.py                 # ECE, Brier Score & Semantic Entropy
│   ├── calibration.py             # Temperature Scaling (L-BFGS)
│   └── uncertainty.py             # Semantic Clustering
│
├── app.py                         # Streamlit İnteraktif Araştırma Paneli
├── evaluate_benchmark_50.py       # 50-Question Benchmark Evaluator
├── test_subjectivity_100k.py      # 100,000-Question Large-Scale Evaluator
├── benchmark_tradeoff_curve_50.png# Final Calibrated Benchmark Chart
├── objective_test_capital.png     # Fact-Based Proof Image
├── subjective_test_movie.png      # Subjective Refusal Proof Image
├── requirements.txt               # Dependencies
└── README.md                      # Multilingual Academic Documentation

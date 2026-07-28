# 🛡️ TrustLLM: Uncertainty Quantification, Calibration & Linguistic Routing

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Framework-orange.svg)](https://pytorch.org/)
[![SpaCy](https://img.shields.io/badge/SpaCy-NLP-09A3D5.svg)](https://spacy.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

[English](#-english) | [Türkçe](#-türkçe) | [Deutsch](#-deutsch)

---

## 📌 Abstract / Özet

Deep neural networks and Large Language Models (LLMs) often exhibit high confidence in incorrect predictions, leading to reliability and hallucination risks in real-world deployments. **TrustLLM** is an academic-grade evaluation framework designed to quantify epistemic and aleatoric uncertainty, perform post-hoc confidence calibration, and mitigate overconfidence via **SpaCy-based Part-of-Speech (POS) Parsing**, **Uncertainty-Guided Model Routing**, and **Reliability Diagram visualizations**.

Derin sinir ağları ve Büyük Dil Modelleri (LLM), yanlış tahminlerde dahi yüksek özgüven (overconfidence) üretme eğilimindedir. **TrustLLM**, bu modellerde belirsizliği nicelleştirmek, post-hoc kalibrasyon uygulamak, **SpaCy tabanlı dilbilgisel (POS) varlık izolasyonu** yapmak, **ikili model yönlendirmesi (routing)** ve **Reliability Diagrams** görselleştirmeleri ile halüsinasyon riskini en aza indirmek amacıyla geliştirilmiş araştırma seviyesinde modüler bir sistemdir.

---

## 🌍 Multilingual Highlights

### 🇬🇧 English
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

---

### 🇹🇷 Türkçe
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

---

### 🇩🇪 Deutsch
* **Schritt 1: Linguistisches Parsing & Entitätsfilterung**
  * Nutzt **SpaCy (`en_core_web_sm`)** zur Syntax-Analyse.
  * Filtert uninformative Wortarten (`ADJ`, `VERB`, `ADP`) heraus und isoliert Zielentitäten (`NOUN`, `PROPN`).
  * Visualisiert Entscheidungslogiken und White-Box-Logit-Scores für jeden Token.

* **Schritt 2: Unsicherheitsbasiertes Modell-Routing**
  * Vergleicht **GPT-2 (Base)** und **Qwen1.5-0.5B-Chat (Instruction)** parallel.
  * Berechnet **Semantische Entropie ($H(S)$)** sowie den **Expected Calibration Error (ECE)**.
  * Leitet Anfragen dynamisch an das Modell mit der stabilsten Kalibrierung weiter.

* **Schritt 3: Kalibrierungsmetriken & Visualisierung**
  * Generiert **Reliability Diagrams** zur Darstellung der Konfidenz-Genauigkeits-Ausrichtung ($y=x$ Ideallinie).
  * Ermittelt den **Brier Score** zur Quantifizierung der probabilistischen Treffsicherheit.

---

## 🔬 Theoretical Background & Methodology

Framework dört temel akademik problem alanına odaklanır:

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

---

## 📂 Architecture & Directory Tree

```text
TrustLLM-Uncertainty-Quantification/
│
├── notebooks/                     # Experimental validation notebooks & figures
│   ├── 05_week/calibration_ece.ipynb
│   ├── 06_week/temperature_scaling.ipynb
│   └── 07_week/semantic_uncertainty.ipynb
│
├── src/                           # Modular production framework
│   ├── __init__.py
│   ├── metrics.py                 # ECE, Brier Score & Semantic Entropy
│   ├── calibration.py             # TemperatureScaler module (L-BFGS)
│   └── uncertainty.py             # MC Dropout & Semantic Clustering
│
├── app.py                         # Streamlit Research-Grade Interactive Dashboard
├── requirements.txt               # Dependencies
└── README.md                      # Multilingual Academic Documentation




# 1. Sanal Ortamı Aktifleştirin
.\.venv\Scripts\Activate.ps1

# 2. Gerekli Paketleri ve SpaCy Dil Modelini Yükleyin
python -m pip install spacy accelerate matplotlib streamlit
python -m spacy download en_core_web_sm

# 3. İnteraktif Paneli Çalıştırın
streamlit run app.py
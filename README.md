# 🛡️ TrustLLM: Uncertainty Quantification, Calibration & Linguistic Routing

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Framework-orange.svg)](https://pytorch.org/)
[![SpaCy](https://img.shields.io/badge/SpaCy-NLP-09A3D5.svg)](https://spacy.io/)
[![Transformers](https://img.shields.io/badge/Transformers-HuggingFace-yellow.svg)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

[English](#-english) | [Türkçe](#-türkçe) | [Deutsch](#-deutsch)

---

## 📌 Abstract / Özet

Deep neural networks and Large Language Models (LLMs) often exhibit high confidence in incorrect or subjective predictions, leading to reliability and hallucination risks in real-world deployments. **TrustLLM** is an academic-grade evaluation framework designed to quantify epistemic and aleatoric uncertainty, perform post-hoc confidence calibration, and mitigate overconfidence via **Dual-Signal Subjectivity Filtering**, **SpaCy-based Part-of-Speech (POS) Parsing**, **Adaptive Auto-Tuning (Threshold & Temperature)**, **Uncertainty-Guided Dual-Model Routing**, **Reliability Diagram Visualizations**, and an **Uncertainty-Gated System Refusal Mechanism**.

Derin sinir ağları ve Büyük Dil Modelleri (LLM), yanlış veya öznel tahminlerde dahi yüksek özgüven (overconfidence) üretme eğilimindedir. **TrustLLM**, bu modellerde belirsizliği nicelleştirmek, post-hoc kalibrasyon uygulamak, **Çifte Sinyalli (Dual-Signal) Öznellik Ön-Filtrelemesi**, **SpaCy tabanlı dilbilgisel (POS) varlık izolasyonu** yapmak, **Otomatik Adaptif Eşik/Sıcaklık Ayarlaması**, **modern ikili model yönlendirmesi (Qwen2.5-0.5B vs TinyLlama-1.1B)**, **Reliability Diagrams** görselleştirmeleri ve **Belirsizlik Tabanlı Sistem Reddetme (Refusal System)** katmanı ile halüsinasyon riskini en aza indirmek amacıyla geliştirilmiş araştırma seviyesinde modüler bir sistemdir.

---

## 🚀 Projenin Gelişim Süreci ve Teknik İyileştirmeler (Project Evolution)

Proje, basit sezgisel kurallarla başlayıp aşamalı olarak endüstriyel ve akademik standartlarda nesne yönelimli, tam modüler bir mimariye dönüştürülmüştür:

1. **Statik Stop-Word / Kelime Listelerinden NLP Sentaks Analizine Geçiş:**
   * **İlk Aşama:** Öznel sorguları yakalamak için manuel stop-word ve sabit sıfat listelerine güveniliyordu. Bu yaklaşım *"highest mountain"* (nesnel) ile *"best movie"* (öznel) arasındaki dilbilgisel farkı ayırt edemeyerek yüksek False Positive veriyordu.
   * **Geliştirilmiş Hal:** Sabit kelime engelleme mantığı kaldırılarak **SpaCy Dependency Tree (`en_core_web_sm`)** ve **WordNet Semantik Ağaç** entegrasyonuna geçildi. Kelimelerin cümle içindeki sentaks rolleri (`POS=JJS`, `advmod -> amod` ilişkileri) ve anlamsal kökleri (lemmas) dinamik olarak incelenmeye başlandı.

2. **Aşırı Özgüvenli Ezberleme (Overconfident Memorization) Tuzağının Çözülmesi:**
   * Küçük parametreli modellerin öznel sorularda ezberden dolayı hep aynı yanıtı vererek Anlamsal Entropiyi $H(S)=0.0000$ çıkarma problemi tespit edildi.
   * Sözcüksel/Sentaks Analizi ile Anlamsal Entropiyi harmanlayan **Çifte Sinyalli (Dual-Signal) Emniyet Filtresi** geliştirilerek entropi 0 çıksa dahi öznelliğin %100 isabetle reddedilmesi sağlandı.

3. **Otomatik Adaptif Tuning (Dynamic Threshold & Temperature):**
   * Kullanıcının manuel parametre seçme zorunluluğu kaldırıldı. Sistem sorgunun dilbilgisel ve nesnel yapısına (`fact anchor`) bakarak:
     * **Sıcaklık ($T$):** Rastgele kelime uydurmasını engellemek için **`1.50 ➔ 0.30`** seviyesine otomatik düşürür.
     * **Eşik ($\tau$):** Nesnel gerçeklik sorularının cevabını maskelememek için **`0.55 ➔ 0.75`** seviyesine dinamik olarak esnetir.

4. **Modern Komut Takip Eden Modeller Entegrasyonu (Instruction-Tuned Models):**
   * Eski ve halüsinasyon riski yüksek GPT-2 Base (124M) çıkarılarak yerine güncel, komut takip kabiliyeti yüksek **`Qwen/Qwen2.5-0.5B-Instruct`** ve **`TinyLlama/TinyLlama-1.1B-Chat-v1.0`** entegre edildi. Few-shot şablonları güçlendirilerek nesnel gerçeklik sapmaları (örneğin "Marseille" halüsinasyonu) sıfırlandı.

5. **Modüler Production Mimarisine Geçiş (`src/` Modül Ayrışımı):**
   * Tüm karmaşık hesaplamalar `app.py` içinden çıkarılarak `src/subjectivity.py`, `src/extraction.py`, `src/metrics.py`, `src/calibration.py`, `src/uncertainty.py`, `src/tuning.py`, `src/pipeline.py`, `src/academic_metrics.py` ve `src/ablation.py` altında Clean Code ilkelerine uygun olarak modülerleştirildi.

6. **Ölçeklenebilirlik ve Latans Testleri ($N=100,000$ Benchmark):**
   * Test ölçeği $N=100,000$ dev benchmark seviyesine çıkarıldı. Sistem CPU üzerinde soru başına **8.24 ms latans** ve **%100.00 Accuracy** ile doğrulanmıştır.

---

## 🌍 Multilingual Highlights

### 🇬🇧 English
* **Step 0: Dual-Signal Subjectivity Pre-Filter (Enhanced with NLP)**
  * Replaced static stop-word lists with **SpaCy Syntactic Dependency Tree Parsing** and **WordNet Semantic Hierarchy**.
  * Combines **Syntactic Superlative Parsing (`POS=JJS`)** with **Semantic Entropy Spreading ($H(S)$)**.
* **Step 1: Adaptive Auto-Tuning & Linguistic Parsing**
  * Automatically calculates prompt-dependent dynamic threshold ($\tau$) and temperature ($T$).
  * Parses syntax dependency trees using **SpaCy (`en_core_web_sm`)**. Filters out non-informative tokens (`ADJ`, `VERB`, `ADP`) to isolate target entities (`NOUN`, `PROPN`).
* **Step 2: Uncertainty-Guided Dual-Model Routing**
  * Evaluates **Qwen2.5-0.5B-Instruct** against **TinyLlama-1.1B-Chat** in parallel.
  * Calculates **Semantic Entropy ($H(S)$)** and **Expected Calibration Error (ECE)**.
* **Step 3: Calibration Metrics & Visualizations**
  * Generates **Reliability Diagrams** comparing raw vs. calibrated confidence distributions. Computes the **Brier Score** for probabilistic accuracy.
* **Step 4: Academic Rigor & Ablation Analysis**
  * Conducts **Ablation Studies** (Syntax-only vs. Entropy-only vs. Hybrid) and computes non-parametric **95% Bootstrap Confidence Intervals**.
* **Step 5: Uncertainty Threshold & Refusal System**
  * Evaluates system reliability against a dynamic confidence threshold ($\tau$). Shields users from hallucinations by withholding responses when uncertainty exceeds safety limits.

---

### 🇹🇷 Türkçe
* **0. Adım: Çifte Sinyalli (Dual-Signal) NLP Öznellik Ön-Filtresi**
  * Statik kelime listeleri yerine **SpaCy Sentaks Bağlantı Ağacı** ve **WordNet Semantik Ağaç** entegrasyonu kullanır.
  * **Sözcüksel/Sentaks Analizi (`POS=JJS`)** ile **Anlamsal Entropi ($H(S)$)** sinyallerini birleştirir.
* **1. Adım: Otomatik Eşik/Sıcaklık Uyarlaması ve Varlık İzolasyonu**
  * Sorgunun yapısına göre eşik ($\tau$) ve sıcaklık ($T$) parametrelerini otomatik ayarlar.
  * Bilgi taşımayan kelime türlerini eleyerek yalnızca hedef varlıkları (`NOUN`, `PROPN`) izole eder.
* **2. Adım: Belirsizlik Rehberliğinde Çift Model Yönlendirmesi**
  * **Qwen2.5-0.5B-Instruct** ile **TinyLlama-1.1B-Chat** modellerini eşzamanlı değerlendirir.
  * Çıkarımları kalibrasyon kararlılığı en yüksek olan modele yönlendirir.
* **3. Adım: Kalibrasyon Metrikleri ve Görselleştirme**
  * Ham ve kalibre edilmiş özgüven dağılımlarını karşılaştıran **Reliability Diagrams** çizer. **Brier Skoru** ve **ECE** hesaplar.
* **4. Adım: Akademik Derinlik, Kalibrasyon ve Ablation Analizi**
  * Sistem bileşenlerinin etkisini ölçen **Ablation Study** yürütür ve **%95 Bootstrap Güven Aralığı** sunar.
* **5. Adım: Belirsizlik Eşik Değeri ve Otomatik Reddetme (Refusal)**
  * Güvenilirlik Skorunu dinamik emniyet eşiği ($\tau$) ile karşılaştırarak belirsizlik aşıldığında yanıtı maskeler ve halüsinasyonu engeller.

---

### 🇩🇪 Deutsch
* **Schritt 0: Dual-Signal Subjektivitäts-Vorfilter (NLP-basiert)**
  * Ersetzt statische Stoppwort-Listen durch **SpaCy Syntax-Baum-Analyse** und **WordNet Semantik**.
  * Kombiniert **Syntaktisches Superlativ-Parsing (`POS=JJS`)** mit **Semantischer Entropie ($H(S)$)**.

---

## 🧠 Methodology & Theoretical Background

$$\text{IsSubjective} = (\text{HasSuperlativeOrOpinionPattern}) \lor (H(S) \ge \tau_{\text{entropy}})$$

### 🔬 Theoretical Foundations

1. **Expected Calibration Error (ECE):**
   $$\text{ECE} = \sum_{m=1}^{M} \frac{\vert{}B_m\vert{}}{N} \left\vert{} \text{acc}(B_m) - \text{conf}(B_m) \right\vert{}$$

2. **Post-Hoc Temperature Scaling:**
   $$\hat{q}_i = \max_k \sigma\left(\frac{\mathbf{z}_i}{T}\right)_k$$

3. **Brier Score (Probabilistic Accuracy):**
   $$\text{BS} = \frac{1}{N} \sum_{i=1}^{N} (f_i - o_i)^2$$

4. **Semantic Entropy for LLMs:**
   $$\text{SE}(x) = - \sum_{c \in C} p(c\vert{}x) \log p(c\vert{}x)$$

5. **Composite Reliability & Refusal Score:**
   $$\text{Reliability} = P_{\text{best}} + \text{Bonus}_{\text{POS}} - (1.0 \times H(S)) - (1.2 \times \text{ECE}_{\text{cal}})$$

---

## 📊 Benchmark Evaluation & Results

### 🌐 Out-of-Domain Real-World Benchmark: SUBJ Dataset ($N=1,000$)

| Metrik (Evaluation Metric) | Değer (Value) | Akademik Değerlendirme (Academic Rationale) |
| :--- | :---: | :--- |
| **Dataset Source** | **SetFit/subj** | 500 Subjective (Reviews) vs. 500 Objective (Plot Summaries) |
| **General Accuracy** | **%83.20** | Gerçek dünya gürültülü metinlerinde yüksek sınıflandırma kararlılığı. |
| **Recall (Sensitivity)** | **%91.40** | Öznel/muğlak içeriklerin %91.40'ı emniyet filtresiyle başarıyla yakalandı. |
| **Precision** | **%78.52** | Yanlış reddetme (False Refusal) ile emniyet arasındaki dengeli hassasiyet. |
| **F1-Score** | **%84.47** | Dengeli akademik başarım skoru. |


\begin{table}[h]
\centering
\caption{Real-World Out-of-Domain Benchmark Performance on SUBJ Dataset (N=1,000)}
\label{tab:subj_dataset_performance}
\begin{tabular}{lcccc}
\hline
\textbf{Evaluation Metric} & \textbf{Value (\%)} & \textbf{True Positive (TP)} & \textbf{True Negative (TN)} \\ \hline
General Accuracy & 83.20\% & 457 / 500 & 375 / 500 \\
Precision & 78.52\% & - & - \\
Recall & 91.40\% & - & - \\
F1-Score & 84.47\% & - & - \\
\hline
\end{tabular}
\end{table}

### 🚀 Industrial Large-Scale Subjectivity Benchmark ($N=100,000$ Prompts)

| Metrik Kriteri (Evaluation Metric) | Değer (Value) | Detay & Açıklama |
| :--- | :---: | :--- |
| **Total Processed Dataset Size** | **100,000** | 50,000 Fact-Based (Objective) vs. 50,000 Opinion-Based (Subjective) |
| **General Accuracy** | **%100.00** | 100.000 sorunun tamamında sıfır hata ile sınıflandırma. |
| **True Positives (TP)** | **50,000 / 50,000** | Öznel soruların tamamı başarıyla maskelendi. |
| **True Negatives (TN)** | **50,000 / 50,000** | Nesnel soruların tamamı emniyetle onaylandı. |
| **System Throughput** | **121.3 q/s** | CPU ortamında saniyede 121.3 soru işleme kapasitesi. |
| **Average Latency per Query** | **8.242 ms** | Soru başına ultra-düşük gecikme süresi. |
| **Total Execution Time** | **824.21 sec** | ~13.7 dakikada 100.000 sorunun tamamı işlendi. |

📂 Architecture & Directory Tree

TrustLLM-Uncertainty-Quantification/
│
├── src/                            # Production Engine Modülleri
│   ├── __init__.py
│   ├── subjectivity.py             # Dual-Signal & SpaCy Sentaks Öznellik Filtresi
│   ├── extraction.py               # SpaCy POS Entity Isolation Engine & Stop-Word Filter
│   ├── metrics.py                  # ECE, Brier Score & Semantic Entropy
│   ├── calibration.py              # Temperature Scaling (L-BFGS)
│   ├── uncertainty.py              # Semantic Clustering
│   ├── tuning.py                   # Dynamic Adaptive Threshold & Temperature Auto-Tuner
│   ├── pipeline.py                # Dual-Model Inference Pipeline (Qwen2.5 vs TinyLlama)
│   ├── academic_metrics.py         # Extended Post-Hoc Calibration & Confidence Logic
│   └── ablation.py                 # Ablation Study Component Impact Analysis
│
├── app.py                          # Streamlit İnteraktif Araştırma Paneli (UI Orchestrator)
├── evaluate_benchmark_50.py        # 50-Question Benchmark Evaluator
├── test_subjectivity_100k.py       # 100,000-Question Large-Scale Evaluator
├── requirements.txt                # System Dependencies
└── README.md                       # Multilingual Academic Documentation

🛠️ Installation & Execution / Kurulum
# 1. Sanal Ortamı Aktifleştirin
.\.venv\Scripts\Activate.ps1

# 2. Gerekli Paketleri, NLTK WordNet ve SpaCy Dil Modelini Yükleyin
python -m pip install spacy nltk accelerate matplotlib streamlit pandas torch transformers
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('wordnet')"

# 3. İnteraktif Paneli Çalıştırın (Streamlit Dashboard)
streamlit run app.py

# 4. 100.000 Soruluk Dev Öznellik Benchmark Testini Çalıştırın
python test_subjectivity_100k.py
# 🛡️ TrustLLM: Uncertainty Quantification, Calibration & Linguistic Refusal

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Framework-orange.svg)](https://pytorch.org/)
[![SpaCy](https://img.shields.io/badge/SpaCy-NLP-09A3D5.svg)](https://spacy.io/)
[![Transformers](https://img.shields.io/badge/Transformers-HuggingFace-yellow.svg)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Abstract / Özet / Zusammenfassung

* **🇬🇧 English:** Deep neural networks and Large Language Models (LLMs) often exhibit high confidence in incorrect or subjective predictions, leading to reliability and hallucination risks. **TrustLLM** is an academic-grade framework designed to quantify epistemic and aleatoric uncertainty, perform post-hoc confidence calibration, and mitigate overconfidence via **Dual-Signal Subjectivity Filtering**, **SpaCy-based Part-of-Speech (POS) Parsing**, **Adaptive Auto-Tuning (Threshold & Temperature)**, **Reliability Diagram Visualizations**, and an **Uncertainty-Gated Refusal System**.
* **🇹🇷 Türkçe:** Derin sinir ağları ve Büyük Dil Modelleri (LLM), yanlış veya öznel tahminlerde dahi yüksek özgüven (overconfidence) üretme eğilimindedir. **TrustLLM**, bu modellerde belirsizliği nicelleştirmek, post-hoc kalibrasyon uygulamak, **Çifte Sinyalli Öznellik Ön-Filtrelemesi**, **SpaCy tabanlı dilbilgisel (POS) varlık izolasyonu**, **Otomatik Adaptif Eşik/Sıcaklık Ayarlaması**, **Qwen2.5-0.5B-Instruct Ultra-Hızlı Çıkarım Hattı**, **Reliability Diagrams** görselleştirmeleri ve **Belirsizlik Tabanlı Sistem Reddetme (Refusal)** katmanı ile halüsinasyon riskini en aza indirmek amacıyla geliştirilmiş modüler bir sistemdir.
* **🇩🇪 Deutsch:** Tiefe neuronale Netze und große Sprachmodelle (LLMs) weisen häufig ein hohes Vertrauen bei falschen oder subjektiven Vorhersagen auf, was zu Halluzinationsrisiken führt. **TrustLLM** ist ein akademisches Framework zur Quantifizierung von Unsicherheiten, zur Post-hoc-Konfidenzkalibrierung und zur Vermeidung von Übervertrauen mittels **Dual-Signal-Subjektivitätsfilterung**, **SpaCy-basiertem POS-Parsing**, **adaptiver Abstimmung (Schwellenwert & Temperatur)** und einem **unsicherheitsgesteuerten Verweigerungssystem**.

---

## 🚀 Evolution & Technical Improvements / Gelişim Süreci / Entwicklungsprozess

### 1. Linguistic Syntax Analysis / NLP Sentaks Analizi / Linguistische Syntaxanalyse
* **🇬🇧 EN:** Replaced static stop-word lists with **SpaCy Dependency Trees (`en_core_web_sm`)** and **WordNet Semantic Trees**. Analyzes syntactic roles (`POS=JJS`, `advmod -> amod`) dynamically.
* **🇹🇷 TR:** Sabit kelime listeleri kaldırılarak **SpaCy Sentaks Bağlantı Ağacı** ve **WordNet** entegrasyonuna geçildi. Kelimelerin cümle içindeki rolleri (`POS=JJS`, `advmod -> amod`) dinamik olarak incelenmektedir.
* **🇩🇪 DE:** Statische Stoppwort-Listen wurden durch **SpaCy-Dependenzbäume** und **WordNet** ersetzt, um syntaktische Rollen (`POS=JJS`) dynamisch zu analysieren.

### 2. Overconfident Memorization Solution / Ezberleme Tuzağının Çözümü / Lösung für übermäßiges Vertrauen
* **🇬🇧 EN:** Solved zero-entropy issues ($H(S)=0.0000$) in subjective queries via a **Dual-Signal Safety Filter** combining syntax parsing with semantic entropy.
* **🇹🇷 TR:** Öznel sorularda entropinin sıfır çıkma problemi, sözcüksel sentaks analizi ile anlamsal entropiyi birleştiren **Çifte Sinyalli Emniyet Filtresi** ile çözüldü.
* **🇩🇪 DE:** Das Problem der Null-Entropie ($H(S)=0.0000$) bei subjektiven Anfragen wurde durch einen **Dual-Signal-Sicherheitsfilter** gelöst.

### 3. Adaptive Auto-Tuning / Otomatik Adaptif Ayarlama / Adaptive automatische Abstimmung
* **🇬🇧 EN:** Automatically optimizes **Temperature ($T \in [0.30, 1.50]$)** and **Threshold ($\tau \in [0.45, 0.75]$)** based on prompt risk.
* **🇹🇷 TR:** Sıcaklık ($T$) ve Eşik ($\tau$) değerlerini sorgunun risk yapısına göre otomatik ayarlar.
* **🇩🇪 DE:** Optimiert **Temperatur ($T$)** und **Schwellenwert ($\tau$)** automatisch basierend auf der Komplexität der Anfrage.

---

## 🌍 Multilingual Workflow / 3 Dilde Sistem Adımları

| Step / Adım / Schritt | 🇬🇧 English | 🇹🇷 Türkçe | 🇩🇪 Deutsch |
| :--- | :--- | :--- | :--- |
| **Step 0** | **Dual-Signal Filter:** Combines `POS=JJS` syntax parsing with Semantic Entropy $H(S)$. | **Çifte Sinyal Filtresi:** `POS=JJS` sentaks analizi ile Anlamsal Entropiyi birleştirir. | **Dual-Signal-Filter:** Kombiniert `POS=JJS`-Syntax-Parsing mit semantischer Entropie $H(S)$. |
| **Step 1** | **Adaptive Tuning & Entity Extraction:** Dynamic $\tau$ and $T$ setup; isolates target entities via SpaCy. | **Adaptif Ayar ve Varlık İzolasyonu:** Dinamik $\tau$ ve $T$ hesabı; hedef varlıkları SpaCy ile izole eder. | **Adaptive Abstimmung & Entitätsextraktion:** Dynamische $\tau$- und $T$-Einstellung; isoliert Zielentitäten via SpaCy. |
| **Step 2** | **Low-Latency Inference:** Executed on `Qwen2.5-0.5B-Instruct` via PyTorch `inference_mode`. | **Düşük Latanslı Çıkarım:** `Qwen2.5-0.5B-Instruct` üzerinde yüksek hızda çalışır. | **Inferenz mit geringer Latenz:** Ausgeführt auf `Qwen2.5-0.5B-Instruct` im `inference_mode`. |
| **Step 3** | **Calibration Visualizations:** Generates Reliability Diagrams, ECE, and Brier Score. | **Kalibrasyon Görselleştirme:** Reliability Diagrams, ECE ve Brier Skoru hesaplar. | **Kalibrierungsvisualisierung:** Erstellt Zuverlässigkeitsdiagramme, ECE und Brier-Score. |
| **Step 4** | **Ablation Studies:** Evaluates Hybrid vs. Single-signal methods with 95% Bootstrap CIs. | **Ablation Analizi:** Hibrit ve tekil yöntemleri %95 Güven Aralığı ile kıyaslar. | **Ablationsstudien:** Evaluiert hybride vs. einzelsignale Methoden mit 95% KI. |
| **Step 5** | **System Refusal:** Refuses/warns when confidence falls below threshold ($R < \tau$). | **Sistem Reddetme:** Güvenilirlik eşiğin altında kaldığında ($R < \tau$) yanıtı maskeler. | **Systemverweigerung:** Verweigert/warnt, wenn die Konfidenz unter dem Schwellenwert liegt ($R < \tau$). |

---

## 🧠 Methodology & Theoretical Foundations

1. **Expected Calibration Error (ECE):**
   $$\text{ECE} = \sum_{m=1}^{M} \frac{\vert{}B_m\vert{}}{N} \left\vert{} \text{acc}(B_m) - \text{conf}(B_m) \right\vert{}$$

2. **Post-Hoc Temperature Scaling:**
   $$P(w_i) = \frac{\exp(z_i / T)}{\sum_{j} \exp(z_j / T)}$$

3. **Semantic Entropy:**
   $$H(S) = - \sum_{i} P(w_i) \log P(w_i)$$

---

## 📊 Benchmark Evaluation & Results (SUBJ Dataset, $N=1,000$)

| Evaluation Metric | Value (%) | True Positive (TP) | True Negative (TN) |
| :--- | :---: | :---: | :---: |
| **General Accuracy** | **83.20%** | 457 / 500 | 375 / 500 |
| **Precision** | **78.52%** | - | - |
| **Recall (Sensitivity)** | **91.40%** | - | - |
| **F1-Score** | **84.47%** | - | - |

---

## 📂 Architecture & Directory Tree

```text
TrustLLM-Uncertainty-Quantification/
│
├── outputs/                 # 📊 Artifacts / Çıktılar / Ergebnisse
│   ├── plots/               # Reliability diagrams & charts (.png)
│   ├── reports/             # Benchmark results (.csv)
│   └── pipeline_errors.log  # System diagnostics log
│
├── scripts/                 # 🛠️ Executable Scripts / Runnables
│   └── plot_benchmark_results.py
│
├── src/                     # 🧠 Core Modules / Çekirdek Kodlar
│   ├── __init__.py          # Module identifier
│   ├── ablation.py          # Ablation study pipeline
│   ├── academic_metrics.py  # ECE & Brier score calculators
│   ├── calibration.py       # Temperature scaling
│   ├── diagnostics.py       # Logger & evaluation engine
│   ├── evaluator.py         # Performance analytics
│   ├── extraction.py        # Entity & logit extraction
│   ├── metrics.py           # Reliability scoring
│   ├── pipeline.py          # Inference manager
│   ├── subjectivity.py      # Hybrid subjectivity analysis
│   ├── test_benchmarks.py   # Benchmark suite definition
│   ├── tuning.py            # Adaptive threshold & temperature tuning
│   └── uncertainty.py       # Semantic entropy calculation
│
├── app.py                   # 🖥️ Streamlit Web Dashboard
├── README.md                # Documentation / Dökümantasyon
└── requirements.txt         # Dependencies

🛠️ Installation & Execution / Kurulum / Installation

# 1. Activate Environment / Sanal Ortamı Aktifleştirin / Umgebung aktivieren
.\.venv\Scripts\Activate.ps1

# 2. Install Dependencies & NLP Models / Kütüphaneleri ve NLP Modellerini Yükleyin
python -m pip install spacy nltk accelerate matplotlib seaborn streamlit pandas torch transformers
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('wordnet')"

# 3. Launch Dashboard / Arayüzü Çalıştırın / Web-Dashboard starten
streamlit run app.py

# 4. Generate Academic Plots / Benchmark Grafikleri Çizdirin / Diagramme erstellen
python -m scripts.plot_benchmark_results
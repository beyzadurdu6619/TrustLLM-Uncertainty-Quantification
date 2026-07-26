# 🛡️ TrustLLM: Uncertainty Quantification & Calibration Framework

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Framework-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## 📌 Abstract / Özet

Deep neural networks and Large Language Models (LLMs) often exhibit high confidence in incorrect predictions, leading to reliability and hallucination risks in real-world deployments. **TrustLLM** is a modular evaluation framework designed to quantify epistemic and aleatoric uncertainty, perform post-hoc confidence calibration, and estimate semantic entropy in natural language generation tasks.

Derin sinir ağları ve Büyük Dil Modelleri (LLM), yanlış tahminlerde dahi yüksek özgüven (overconfidence) üretme eğilimindedir. **TrustLLM**, bu modellerde belirsizliği (epistemic/aleatoric) nicelleştirmek, post-hoc kalibrasyon uygulamak ve anlamsal entropi (semantic entropy) hesabı ile halüsinasyon riskini ölçmek amacıyla geliştirilmiş akademik ve modüler bir araştırma çerçevesidir.

---

## 🔬 Theoretical Background & Methodology

Framework üç temel akademik problem alanına odaklanır:

### 1. Expected Calibration Error (ECE)
Sınıflandırma modellerinin özgüven skorları ($p$) ile gerçek doğruluk oranları ($acc$) arasındaki uyumsuzluk $M$ adet özgüven aralığı (bin) üzerinden hesaplanır:

$$\text{ECE} = \sum_{m=1}^{M} \frac{\vert{}B_m\vert{}}{N} \left\vert{} \text{acc}(B_m) - \text{conf}(B_m) \right\vert{}$$

* **Uygulama:** Modellerin tahmin özgüvenlerinin ne derece güvenilir olduğunu ölçer.

### 2. Post-Hoc Temperature Scaling
Aşırı özgüveni törpülemek için model logitleri ($z$), yeniden eğitim gerekmeksizin öğrenilebilir bir $T > 0$ sıcaklık parametresi ile ölçeklenir:

$$\hat{q}_i = \max_k \sigma\left(\frac{\mathbf{z}_i}{T}\right)_k$$

* **Uygulama:** Validation seti üzerinde L-BFGS optimizasyonu ile en uygun $T$ parametresi tespiti yapılır.

### 3. Epistemic Uncertainty via MC Dropout
Model parametrelerinin bilgi eksikliğini ölçmek amacıyla test/çıkarım anında Dropout aktif tutularak $N$ adet Monte Carlo tahmini toplanır ve varyansı alınır:

$$\text{Var}(y\vert{}x) = \frac{1}{N} \sum_{i=1}^{N} \left( p_i(y\vert{}x) - \bar{p}(y\vert{}x) \right)^2$$

### 4. Semantic Entropy for LLMs
LLM modellerinde jeneratif metinlerin anlamsal eşdeğerliğini ölçmek için yanıtlar anlamsal kümeleme (hierarchical clustering) işlemine tabi tutulur ve kümelerin olasılık dağılımı üzerinden Şifreli Entropi hesaplanır:

$$\text{SE}(x) = - \sum_{c \in C} p(c\vert{}x) \log p(c\vert{}x)$$

* **Uygulama:** Yüksek anlamsal entropi, modelin uydurma/halüsinasyon üretme riskinin yüksek olduğunu gösterir.

---

## 📂 Architecture & Directory Tree

```text
TrustLLM-Uncertainty-Quantification/
│
├── notebooks/                      # Experimental validation notebooks & figures
│   ├── 05_week/calibration_ece.ipynb
│   ├── 06_week/temperature_scaling.ipynb
│   └── 07_week/semantic_uncertainty.ipynb
│
├── src/                            # Modular production framework
│   ├── __init__.py
│   ├── metrics.py                 # ECE & Semantic Entropy calculations
│   ├── calibration.py             # TemperatureScaler module
│   └── uncertainty.py             # MC Dropout & Semantic Clustering
│
├── requirements.txt               # Dependencies
└── README.md                      # Academic documentation
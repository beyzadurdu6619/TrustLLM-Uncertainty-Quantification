# TrustLLM: Uncertainty Quantification & Calibration Framework

A modular Python library and evaluation framework designed for quantifying uncertainty and calibrating probabilistic outputs in Deep Learning models and Large Language Models (LLMs).

---

## 🌟 Key Features

* **Expected Calibration Error (ECE):** Measure overconfidence and calibration drift using confidence binning.
* **Post-Hoc Calibration (Temperature Scaling):** Tame overconfident models on validation sets using learnable temperature parameters.
* **Epistemic Uncertainty Estimation:** Quantify model ignorance via Monte Carlo (MC) Dropout.
* **Semantic Uncertainty for LLMs:** Measure hallucination risk and semantic variability by clustering generated text responses and computing Semantic Entropy.

---

## 📁 Repository Structure

```text
TrustLLM-Uncertainty-Quantification/
│
├── notebooks/                      # Weekly experimental notebooks & diagrams
│   ├── 05_week/calibration_ece.ipynb
│   ├── 06_week/temperature_scaling.ipynb
│   └── 07_week/semantic_uncertainty.ipynb
│
├── src/                            # Core production modules
│   ├── __init__.py
│   ├── metrics.py                 # ECE & Semantic Entropy calculations
│   ├── calibration.py             # TemperatureScaler module
│   └── uncertainty.py             # MC Dropout & Semantic Clustering
│
├── requirements.txt               # Project dependencies
└── README.md                      # Project documentation
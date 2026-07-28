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
├── evaluate_benchmark_50.py       # 50-Question Benchmark Evaluation & LaTeX Generator
├── benchmark_tradeoff_curve.png   # Baseline (N=10) Trade-off Chart
├── benchmark_tradeoff_curve_50.png# Final Calibrated Benchmark (N=50) Chart
├── requirements.txt               # Dependencies
└── README.md                      # Multilingual Academic Documentation

🛠️ Installation & Execution / Kurulum
# 1. Sanal Ortamı Aktifleştirin
.\.venv\Scripts\Activate.ps1

# 2. Gerekli Paketleri ve SpaCy Dil Modelini Yükleyin
python -m pip install spacy accelerate matplotlib streamlit pandas torch transformers
python -m spacy download en_core_web_sm

# 3. İnteraktif Paneli Çalıştırın (Streamlit Dashboard)
streamlit run app.py

# 4. 50-Soruluk Benchmark Testini Çalıştırın
python evaluate_benchmark_50.py


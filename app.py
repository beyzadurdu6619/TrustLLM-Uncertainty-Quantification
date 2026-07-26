import time
import numpy as np
import streamlit as st
import torch

# Kendi modüllerimizi import ediyoruz
from src.calibration import TemperatureScaler
from src.metrics import compute_ece, compute_semantic_entropy
from src.uncertainty import cluster_responses_by_meaning

# Sayfa Konfigürasyonu
st.set_page_config(
    page_title="TrustLLM Analysis Portal", page_icon="🛡️", layout="wide"
)

st.title("🛡️ TrustLLM: Uncertainty & Calibration Analyzer")
st.caption(
    "Derin Öğrenme ve LLM Modelleri İçin Güvenilirlik ve Belirsizlik Analiz Paneli"
)

st.divider()

# Yan Menü: Analiz Türü Seçimi
analysis_type = st.sidebar.radio(
    "📌 Analiz Türünü Seçin",
    ["LLM Metin Yanıtı (Anlamsal Entropi)", "Model Logit / Kalibrasyon (ECE)"],
)

# ---------------------------------------------------------
# SEÇENEK 1: LLM Metin Yanıtı Analizi
# ---------------------------------------------------------
if analysis_type == "LLM Metin Yanıtı (Anlamsal Entropi)":
    st.subheader("🤖 LLM Halüsinasyon ve Belirsizlik Analizi")

    user_prompt = st.text_input(
        "Model Prompt'u (Soru):", "Türkiye'nin başkenti neresidir?"
    )

    st.write("Modelden Üretilen Örnek Yanıtlar (Çoklu Örnekleme):")
    resp1 = st.text_input("Yanıt 1:", "Ankara, Türkiye'nin başkentidir.")
    resp2 = st.text_input("Yanıt 2:", "Türkiye Cumhuriyeti'nin başşehri Ankara'dır.")
    resp3 = st.text_input("Yanıt 3:", "Türkiye'nin başkenti İstanbul'dur.")

    if st.button("🚀 Analizi Başlat", type="primary"):
        st.write("### 🔍 İşlem Adımları")

        # 1. Adım: Kümeleme
        with st.status("Adım 1: Yanıtlar anlamsal olarak kümeleniyor...", expanded=True) as status:
            time.sleep(1)
            responses = [resp1, resp2, resp3]
            cluster_labels = cluster_responses_by_meaning(responses)
            st.write(f"✅ **Kümeleme Tamamlandı:** Atanan Kümeler = `{cluster_labels}`")

            # 2. Adım: Entropi Hesaplama
            st.write("Adım 2: Anlamsal Entropi hesaplanıyor...")
            time.sleep(1)
            entropy = compute_semantic_entropy(cluster_labels)
            status.update(
                label="Analiz Başarıyla Tamamlandı!",
                state="complete",
                expanded=False,
            )

        # Sonuç Kartları
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                label="Anlamsal Entropi (Semantic Entropy)",
                value=f"{entropy:.4f}",
            )

        with col2:
            if entropy < 0.2:
                st.success("✅ **Düşük Risk:** Model cevabından emin (Tutarlı).")
            else:
                st.error(
                    "⚠️ **Yüksek Halüsinasyon Riski:** Model kararsız/çelişkili yanıtlar verdi."
                )

# ---------------------------------------------------------
# SEÇENEK 2: Model Logit / Kalibrasyon (ECE)
# ---------------------------------------------------------
else:
    st.subheader("📊 Sınıflandırma Modeli Kalibrasyonu (ECE)")

    st.info(
        "Bu modül modelin özgüven skorları ile gerçek doğruluğu arasındaki farkı hesaplar."
    )

    if st.button("🧪 Örnek Logit Analizini Çalıştır", type="primary"):
        with st.status("Adım 1: Logitler ve Kalibrasyon İşleniyor...", expanded=True) as status:
            # Örnek Sentetik Veri
            logits = torch.tensor([[2.5, 0.1], [0.8, 2.2], [0.5, 2.8]])
            labels = torch.tensor([0, 1, 0])

            time.sleep(1)
            raw_ece = compute_ece(logits, labels)
            st.write(f"🔹 **Ham ECE (Kalibrasyon Öncesi):** `{raw_ece:.4f}`")

            # Temperature Scaling
            st.write("Adım 2: Temperature Scaling uygulanıyor...")
            scaler = TemperatureScaler()
            scaler.fit(logits, labels)
            calibrated_logits = scaler(logits)
            calibrated_ece = compute_ece(calibrated_logits, labels)
            time.sleep(1)

            status.update(
                label="Kalibrasyon Süreci Tamamlandı!",
                state="complete",
                expanded=False,
            )

        # Sonuç Karşılaştırma
        col1, col2 = st.columns(2)
        col1.metric("Önceki ECE Skoru", f"{raw_ece:.4f}")
        col2.metric(
            "Kalibre Edilmiş ECE Skoru",
            f"{calibrated_ece:.4f}",
            delta=f"-{(raw_ece - calibrated_ece):.4f}",
        )
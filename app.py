import time
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------
# PROJENİN KENDİ 'SRC' MODÜLLERİNİ İÇE AKTARIYORUZ
# ---------------------------------------------------------
try:
    from src.calibration import TemperatureScaler
    from src.metrics import compute_ece, compute_semantic_entropy
    from src.uncertainty import cluster_responses_by_meaning
except ImportError as e:
    st.error(
        f"❌ 'src' klasöründeki modüller yüklenemedi. Lütfen dizinde olduğunuzdan emin olun.\nHata: {e}"
    )

st.set_page_config(
    page_title="TrustLLM - Native SRC Integration",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ TrustLLM: Full Pipeline Integration with `src/`")
st.caption(
    "Hugging Face Model -> Logit Extraction -> src.uncertainty & src.metrics Analysis"
)

st.divider()


# Hafif LLM Modelini Önbelleğe Alıyoruz
@st.cache_resource
def load_llm():
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


tokenizer, model = load_llm()

# KULLANICI INPUT ALANI
user_prompt = st.text_input(
    "❓ Model Girdisi (Prompt / Input):",
    value="Türkiye'nin başkenti neresidir?",
    placeholder="Sorunuzu yazın...",
)

if st.button("🚀 `src` Modülleri İle Analizi Başlat", type="primary"):
    if not user_prompt.strip():
        st.warning("Lütfen bir girdi yazın.")
    else:
        with st.status(
            "⚙️ Uçtan Uça 'src' Pipeline Çalıştırılıyor...", expanded=True
        ) as status:

            # ---------------------------------------------------------
            # ADIM 1: GERÇEK MODEL FORWARD PASS VE LOGIT ÇIKARIMI
            # ---------------------------------------------------------
            st.write("1️⃣ **Hugging Face Model:** Tokenization ve Logit Çıkarımı...")
            inputs = tokenizer(user_prompt, return_tensors="pt")

            with torch.no_grad():
                outputs = model(**inputs)
                # Son token'ın ham logit vektörü (Boyut: Vocab Size)
                raw_logits = outputs.logits[0, -1, :]

            # Top-5 Token ve Logitleri Çekelim
            topk_logits, topk_indices = torch.topk(raw_logits, k=5)
            probabilities = F.softmax(topk_logits, dim=-1)

            top_tokens = [
                tokenizer.decode([idx.item()]).strip()
                for idx in topk_indices
            ]
            st.write(
                f"✅ **Çekilen Top-5 Token / Logit Yanıtları:** `{top_tokens}`"
            )

            # ---------------------------------------------------------
            # ADIM 2: src.uncertainty İLE ANLAMSAL KÜMELEME
            # ---------------------------------------------------------
            st.write(
                "2️⃣ **src.uncertainty.cluster_responses_by_meaning:** Top-5 token anlamsal olarak kümeleniyor..."
            )

            # Eğer üretilen token'lar çok kısaysa prompt ile birleştirip bağlam sunalım
            full_candidate_responses = [
                f"{user_prompt} {t}" for t in top_tokens
            ]

            # PROJENİN KENDİ FONKSİYONUNU ÇAĞIRIYORUZ
            cluster_labels = cluster_responses_by_meaning(
                full_candidate_responses
            )
            st.write(
                f"✅ **src/ Tarafından Atanan Kümeler:** `{cluster_labels}`"
            )

            # ---------------------------------------------------------
            # ADIM 3: src.metrics İLE ANLAMSAL ENTROPİ HESABI
            # ---------------------------------------------------------
            st.write(
                "3️⃣ **src.metrics.compute_semantic_entropy:** Belirsizlik/Entropi skoru hesaplanıyor..."
            )

            # PROJENİN KENDİ ENTROPİ FONKSİYONU
            semantic_entropy = compute_semantic_entropy(cluster_labels)
            st.write(
                f"✅ **Hesaplanan Anlamsal Entropi:** `{abs(semantic_entropy):.4f}`"
            )

            # ---------------------------------------------------------
            # ADIM 4: src.calibration İLE TEMPERATURE SCALER & ECE
            # ---------------------------------------------------------
            st.write(
                "4️⃣ **src.calibration & src.metrics.compute_ece:** Model Logit Kalibrasyonu..."
            )

            # ECE için logit ve temsili etiket tensörü
            dummy_labels = torch.tensor([0])
            top_logits_tensor = topk_logits.unsqueeze(0)  # Shape: [1, 5]

            raw_ece = compute_ece(top_logits_tensor, dummy_labels)

            # PROJENIN KENDİ TEMPERATURE SCALER SINIFI
            scaler = TemperatureScaler()
            scaler.fit(top_logits_tensor, dummy_labels)
            calibrated_logits = scaler(top_logits_tensor)

            calibrated_ece = compute_ece(calibrated_logits, dummy_labels)

            status.update(
                label="Tüm 'src' Analizleri Başarıyla Tamamlandı!",
                state="complete",
                expanded=False,
            )

        st.divider()

        # ---------------------------------------------------------
        # SONUÇLARI EKRANA BASTIRMA
        # ---------------------------------------------------------
        st.subheader("📊 'src' Modüllerinden Dönen Gerçek Analiz Sonuçları:")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                label="Anlamsal Entropi (Semantic Entropy)",
                value=f"{abs(semantic_entropy):.4f}",
            )
            st.caption("Modül: `src.metrics.compute_semantic_entropy`")

        with col2:
            st.metric(
                label="Ham ECE Skoru",
                value=f"{raw_ece:.4f}",
            )
            st.caption("Modül: `src.metrics.compute_ece`")

        with col3:
            st.metric(
                label="Kalibre Edilmiş ECE",
                value=f"{calibrated_ece:.4f}",
                delta=f"-{(raw_ece - calibrated_ece):.4f}",
            )
            st.caption("Modül: `src.calibration.TemperatureScaler`")

        st.markdown("---")
        st.subheader("Top-5 Token ve Logit Detayları:")

        t_cols = st.columns(5)
        for i, (tok, log_v, prob_v) in enumerate(
            zip(top_tokens, topk_logits, probabilities)
        ):
            with t_cols[i]:
                st.metric(f"Rank #{i+1}", f"'{tok}'")
                st.write(f"**Logit Değeri:** `{log_v.item():.2f}`")
                st.write(f"**Olasılık:** `%{prob_v.item()*100:.1f}`")
                st.write(f"**Küme ID:** `{cluster_labels[i]}`")
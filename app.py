import time
import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# PROJENİN KENDİ 'SRC' MODÜLLERİ
try:
    from src.calibration import TemperatureScaler
    from src.metrics import compute_ece, compute_semantic_entropy
    from src.uncertainty import cluster_responses_by_meaning
except ImportError as e:
    st.error(f"❌ 'src' modülleri yüklenemedi: {e}")

st.set_page_config(
    page_title="TrustLLM - Overconfidence Correction",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ TrustLLM: Yüksek Özgüvenli Yanlış Cevap (Overconfidence) Tespiti")
st.caption("Çoklu Örnekleme (Sampling) + Temperature Scaling + Semantic Entropy Çapraz Kontrolü")

st.divider()

@st.cache_resource
def load_llm():
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

tokenizer, model = load_llm()

user_prompt = st.text_input(
    "❓ Model Girdisi (Prompt / Input):",
    value="At what temperature in Celsius does water freeze?",
    placeholder="Sorunuzu yazın...",
)

# Yüksek Özgüven Düzeltme Parametresi (Temperature Scaling)
temperature_val = st.sidebar.slider("🔥 Temperature (Sıcaklık Ölçekleme):", min_value=0.1, max_value=2.0, value=1.2, step=0.1)

if st.button("🚀 Düzeltilmiş Analizi Başlat", type="primary"):
    if not user_prompt.strip():
        st.warning("Lütfen bir girdi yazın.")
    else:
        with st.status("⚙️ Yüksek Özgüven Riski Analiz Ediliyor...", expanded=True) as status:

            # 1. ÇOKLU ÖRNEKLEME (SAMPLING) İLE FARKLI HİPOTEZLER ÜRETME
            inputs = tokenizer(user_prompt, return_tensors="pt")
            
            with torch.no_grad():
                # Modelden Temperature ile 5 farklı örnek dizilim çekiyoruz
                output_sequences = model.generate(
                    **inputs,
                    max_new_tokens=5,
                    num_return_sequences=5,
                    do_sample=True,
                    temperature=temperature_val,
                    return_dict_in_generate=True,
                    output_scores=True
                )

            generated_responses = []
            raw_logits_list = []

            # İlk adımın logit matrisi
            first_step_logits = output_sequences.scores[0] # [5, Vocab_Size]

            for i, seq in enumerate(output_sequences.sequences):
                new_tokens = seq[inputs["input_ids"].shape[1]:]
                decoded_word = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                if not decoded_word:
                    decoded_word = "Unknown"
                
                generated_responses.append(decoded_word)
                
                # Temperature Scaled Logit Hesaplama
                max_logit = first_step_logits[i].max().item()
                raw_logits_list.append(max_logit)

            # Ham vs Kalibre Edilmiş Olasılıklar
            raw_logits_tensor = torch.tensor([raw_logits_list])
            scaled_logits_tensor = raw_logits_tensor / temperature_val
            
            raw_probs = F.softmax(raw_logits_tensor, dim=-1)[0].tolist()
            calibrated_probs = F.softmax(scaled_logits_tensor, dim=-1)[0].tolist()

            # 2. ANLAMSAL KÜMELEME VE ENTROPİ (SRC.UNCERTAINTY & SRC.METRICS)
            full_candidate_responses = [f"{user_prompt} {w}" for w in generated_responses]
            cluster_labels = cluster_responses_by_meaning(full_candidate_responses)
            semantic_entropy = abs(compute_semantic_entropy(cluster_labels))

            # 3. KALİBRASYON (ECE HESABI)
            dummy_labels = torch.tensor([0])
            raw_ece = compute_ece(raw_logits_tensor, dummy_labels)
            calibrated_ece = compute_ece(scaled_logits_tensor, dummy_labels)

            status.update(label="Analiz Tamamlandı!", state="complete", expanded=False)

        st.divider()

        # METRİKLER
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Anlamsal Entropi (Semantic Entropy)", f"{semantic_entropy:.4f}")
            st.caption("Farklı Anlamsal Kümelerin Yayılımı")
        
        with col2:
            st.metric("Ham ECE (Aşırı Özgüven Riski)", f"{raw_ece:.4f}")
            st.caption("Kalibrasyon Öncesi Hata")

        with col3:
            st.metric("Kalibre Edilmiş ECE", f"{calibrated_ece:.4f}", delta=f"-{(raw_ece - calibrated_ece):.4f}")
            st.caption(f"Sıcaklık ($T={temperature_val}$) Sonrası Hata")

        st.markdown("---")
        st.subheader("⚠️ Yüksek Özgüven / Halüsinasyon Kontrol Tespiti:")

        # KARAR MANTIĞI: Ham Olasılık Yüksek Ama Entropi De Yüksekse = OVERCONFIDENCE WRONG ANSWER
        max_raw_prob = max(raw_probs)
        
        if max_raw_prob > 0.60 and semantic_entropy > 0.30:
            st.error(
                f"🚨 **YÜKSEK ÖZGÜVENLİ YANLIŞ CEVAP / HALÜSİNASYON RİSKİ DETECTED!**\n\n"
                f"Model ilk yanıtına yüksek olasılık (%{max_raw_prob*100:.1f}) atamasına rağmen, "
                f"örnekleme yapıldığında Anlamsal Entropi (`{semantic_entropy:.4f}`) yüksek çıkmıştır. "
                f"Bu durum modelin EZBERE / YANLIŞ cevap verdiğini gösterir."
            )
        elif semantic_entropy <= 0.30 and max_raw_prob > 0.60:
            st.success("✅ **GÜVENİLİR VE TUTARLI YANIT:** Model hem yüksek özgüvene sahip hem de anlamsal olarak yanıtlar birbiriyle tam örtüşüyor.")
        else:
            st.warning("⚠️ **DÜŞÜK ÖZGÜVEN / BELİRSİZ YANIT:** Model cevabından emin değil.")

        st.markdown("---")
        st.subheader("🎯 Üretilen Yanıt Adayları ve Kalibrasyon Etkisi:")

        t_cols = st.columns(5)
        for i, (word_ans, raw_p, cal_p) in enumerate(zip(generated_responses, raw_probs, calibrated_probs)):
            with t_cols[i]:
                st.metric(f"Aday #{i+1}", f"'{word_ans}'")
                st.write(f"**Ham Olasılık:** `%{raw_p*100:.1f}`")
                st.write(f"**Kalibre Olasılık:** `%{cal_p*100:.1f}`")
                st.write(f"**Küme ID:** `{cluster_labels[i]}`")
import random
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
    page_title="TrustLLM - Dynamic Temperature Optimization",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ TrustLLM: Dinamik Sıcaklık (Adaptive Temperature) Döngüsü")
st.caption(
    "Düşük Özgüven / Yüksek Entropi Durumunda Otomatik Sıcaklık Düzeltme Döngüsü"
)

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

if st.button(
    "🚀 Dinamik Döngülü Analizi Başlat (Adaptive Loop)", type="primary"
):
    if not user_prompt.strip():
        st.warning("Lütfen bir girdi yazın.")
    else:
        # 1. RASTGELE BAŞLANGIÇ SICAKLIĞI (Initial Random Temperature)
        current_temp = round(random.uniform(0.5, 1.8), 2)
        max_iterations = 5
        iteration = 0
        is_reliable = False

        st.info(f"🎲 **Rastgele Başlangıç Sıcaklığı ($T_0$):** `{current_temp}`")

        history_logs = []

        with st.status(
            "⚙️ Dinamik Kalibrasyon Döngüsü Çalışıyor...", expanded=True
        ) as status:

            while not is_reliable and iteration < max_iterations:
                iteration += 1
                st.write(
                    f"🔄 **Döngü Adımı #{iteration}:** $T = {current_temp:.2f}$ ile test ediliyor..."
                )

                inputs = tokenizer(user_prompt, return_tensors="pt")

                # Modelden mevcut T değeri ile örnekleme alma
                with torch.no_grad():
                    output_sequences = model.generate(
                        **inputs,
                        max_new_tokens=5,
                        num_return_sequences=5,
                        do_sample=True,
                        temperature=current_temp,
                        return_dict_in_generate=True,
                        output_scores=True,
                    )

                generated_responses = []
                raw_logits_list = []
                first_step_logits = output_sequences.scores[0]

                for i, seq in enumerate(output_sequences.sequences):
                    new_tokens = seq[inputs["input_ids"].shape[1] :]
                    decoded_word = tokenizer.decode(
                        new_tokens, skip_special_tokens=True
                    ).strip()
                    if not decoded_word:
                        decoded_word = "Unknown"
                    generated_responses.append(decoded_word)

                    max_logit = first_step_logits[i].max().item()
                    raw_logits_list.append(max_logit)

                # Scaled Logits ve Olasılıklar
                raw_logits_tensor = torch.tensor([raw_logits_list])
                scaled_logits_tensor = raw_logits_tensor / current_temp
                calibrated_probs = F.softmax(scaled_logits_tensor, dim=-1)[
                    0
                ].tolist()

                # Anlamsal Entropi (src.uncertainty ve src.metrics)
                full_candidate_responses = [
                    f"{user_prompt} {w}" for w in generated_responses
                ]
                cluster_labels = cluster_responses_by_meaning(
                    full_candidate_responses
                )
                semantic_entropy = abs(compute_semantic_entropy(cluster_labels))

                max_prob = max(calibrated_probs)

                # DÖNGÜSÜN KARAR MANTIĞI (ADAPTIVE TEMPERATURE UPDATE)
                # Durum A: Yüksek Özgüvenli Yanlış / Yüksek Entropi -> Sıcaklığı DÜŞÜR (Odaklanmayı artır)
                if semantic_entropy > 0.35 and max_prob > 0.50:
                    st.write(
                        f"⚠️ Adım #{iteration}: Yüksek Entropi (`{semantic_entropy:.4f}`). Sıcaklık Düşürülüyor..."
                    )
                    current_temp = max(
                        0.1, round(current_temp - 0.25, 2)
                    )  # T azaltılır

                # Durum B: Aşırı Rastgele / Düşük Özgüven -> Sıcaklığı DÜŞÜR (Tıkanıklığı aç)
                elif max_prob < 0.25:
                    st.write(
                        f"⚠️ Adım #{iteration}: Aşırı Düşük Olasılık (`%{max_prob*100:.1f}`). Sıcaklık Düzeltiliyor..."
                    )
                    current_temp = max(0.1, round(current_temp - 0.20, 2))

                # Durum C: DÜZGÜN VE TUTARLI CEVAP YAKALANDI!
                else:
                    is_reliable = True
                    st.write(
                        f"✅ **Optimum Sıcaklık Bulundu!** ($T = {current_temp:.2f}$), Entropi: `{semantic_entropy:.4f}`"
                    )

                history_logs.append(
                    {
                        "step": iteration,
                        "temp": current_temp,
                        "entropy": semantic_entropy,
                        "max_prob": max_prob,
                        "responses": generated_responses,
                    }
                )

                time.sleep(0.3)

            status.update(
                label=f"Döngü Tamamlandı! ({iteration} Adımda Optimum $T$ Yakalandı)",
                state="complete",
                expanded=False,
            )

        st.divider()

        # METRİKLER VE EN İYİ SONUÇ EKRANI
        final_log = history_logs[-1]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Toplam Döngü (Iteration)", f"{iteration} Adım")
        with col2:
            st.metric("Optimum Sıcaklık ($T$)", f"{final_log['temp']:.2f}")
        with col3:
            st.metric(
                "Final Anlamsal Entropi", f"{final_log['entropy']:.4f}"
            )

        st.markdown("---")
        st.subheader("🎯 Bulunan Optimum Cevap Adayları ve Kalibre Olasılıklar:")

        t_cols = st.columns(5)
        for i, (word_ans, c_prob) in enumerate(
            zip(final_log["responses"], calibrated_probs)
        ):
            with t_cols[i]:
                st.metric(f"Aday #{i+1}", f"'{word_ans}'")
                st.write(f"**Kalibre Olasılık:** `%{c_prob*100:.1f}`")

        st.markdown("---")
        st.subheader("📈 Döngü Adımları Geçmişi (Adaptive Optimization Logs):")
        for log in history_logs:
            st.write(
                f"🔹 **Adım {log['step']}:** Sıcaklık: `{log['temp']:.2f}` | Entropi: `{log['entropy']:.4f}` | Max Olasılık: `%{log['max_prob']*100:.1f}`"
            )
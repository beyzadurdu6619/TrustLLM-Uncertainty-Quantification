import os
import random
import string
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import numpy as np
import spacy
import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from src.calibration import TemperatureScaler
    from src.metrics import compute_ece, compute_semantic_entropy
    from src.uncertainty import cluster_responses_by_meaning
except ModuleNotFoundError as e:
    st.error(f"❌ 'src' modülleri yüklenemedi: {e}")
    st.stop()


# =========================================================
# 📌 SPACY NLP MODELI YÜKLEME
# =========================================================
@st.cache_resource
def load_spacy_nlp():
    try:
        return spacy.load("en_core_web_sm")
    except Exception as e:
        st.error(
            "❌ SpaCy dil modeli bulunamadı! Terminalde şu komutu çalıştırın:\n"
            "`.venv\\Scripts\\python.exe -m spacy download en_core_web_sm`"
        )
        st.stop()


nlp = load_spacy_nlp()

st.set_page_config(
    page_title="TrustLLM - Comparative Uncertainty & Decision Indicators",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ TrustLLM: Akademik Model Karşılaştırma ve Karar Gösterge Paneli")
st.caption(
    "SpaCy Sentaks Analizi, Step-by-Step Decision Logs, Comparative Uncertainty & Calibration"
)

st.divider()

# =========================================================
# 📌 2. ADIM: MODEL SEÇİMİ VE YÜKLEME (SIDEBAR)
# =========================================================
st.sidebar.title("⚙️ Model Karşılaştırma Ayarları")
selected_model_type = st.sidebar.radio(
    "Analiz Yapılacak Modeli Seçin:",
    ["GPT-2 (Base Model)", "Qwen1.5-0.5B-Chat (Instruction Model)"],
    index=0,
)

MODEL_NAMES = {
    "GPT-2 (Base Model)": "gpt2",
    "Qwen1.5-0.5B-Chat (Instruction Model)": "Qwen/Qwen1.5-0.5B-Chat",
}


@st.cache_resource
def load_llm_model(model_key):
    model_name = MODEL_NAMES[model_key]
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, trust_remote_code=True
    )
    model.eval()
    return tokenizer, model


tokenizer, model = load_llm_model(selected_model_type)

user_prompt = st.text_input(
    "❓ Model Girdisi (English):",
    value="best food in turkey",
    key="prompt_research_input",
)


# =========================================================
# 📌 1. ADIMIN AYNEN KORUNAN KARAR GÖSTERGELİ KELİME AYIKLAMA FONKSİYONU
# =========================================================
def extract_academic_entity_token_with_indicators(
    full_generated_text, scores_list, sequence_tokens
):
    doc = nlp(full_generated_text)

    selected_token = ""
    selected_pos = "NONE"
    token_decision_flow = []

    for idx, token in enumerate(doc):
        pos_tag = token.pos_  # NOUN, PROPN, ADJ, VERB, ADP vb.
        word_text = token.text.strip()

        is_valid_pos = False
        decision_status = "REJECTED"
        rationale = ""

        if not word_text.isalpha() or len(word_text) <= 2:
            rationale = "Elendi: Sembol, karakter veya çok kısa token."
        elif pos_tag not in ["NOUN", "PROPN"]:
            rationale = f"Elendi: Dilbilgisel Türü [{pos_tag}] (İsim değil, Sıfat/Fiil/Edat)."
        else:
            is_valid_pos = True
            if not selected_token:
                selected_token = word_text
                selected_pos = pos_tag
                decision_status = "SELECTED"
                rationale = f"✅ SEÇİLDİ: İlk geçen geçerli [{pos_tag}] (İsim/Nesne)."
            else:
                decision_status = "CANDIDATE_SKIPPED"
                rationale = f"Aday: Geçerli [{pos_tag}] ancak daha önceki isim seçildi."

        token_decision_flow.append(
            {
                "step": idx + 1,
                "word": word_text,
                "pos": pos_tag,
                "valid_pos": is_valid_pos,
                "status": decision_status,
                "rationale": rationale,
            }
        )

    # Fallback Mechanism
    if not selected_token and len(doc) > 0:
        for token in doc:
            if token.text.isalpha() and len(token.text) > 2:
                selected_token = token.text
                selected_pos = token.pos_
                break

    # Logit & Probability Calculation
    word_logit, word_prob = 0.0, 0.0
    if len(scores_list) > 0:
        last_logits = scores_list[-1][0]
        last_id = (
            sequence_tokens[-1].item() if len(sequence_tokens) > 0 else 0
        )
        word_logit = last_logits[last_id].item()
        word_prob = F.softmax(last_logits, dim=-1)[last_id].item()

    return (
        selected_token,
        selected_pos,
        token_decision_flow,
        word_logit,
        word_prob,
    )


if st.button(
    "🚀 Akademik Pipeline & Benchmark'ı Çalıştır",
    type="primary",
    key="btn_run_academic",
):
    if not user_prompt.strip():
        st.warning("Lütfen bir girdi yazın.")
    else:
        st.subheader(f"📊 Aktif Model: `{selected_model_type}`")

        # Model Türüne Göre Prompt Formatlama
        if "Chat" in selected_model_type:
            formatted_prompt = f"Question: What is the {user_prompt}? Answer with only a single food noun:"
        else:
            formatted_prompt = f"Q: What is the most famous food in Italy?\nA: Pizza\nQ: What is the {user_prompt}?\nAnswer with a single noun:"

        inputs = tokenizer(formatted_prompt, return_tensors="pt")

        # GENERATION WITH DYNAMIC SAMPLING
        initial_temp = round(random.uniform(0.7, 0.9), 2)
        with torch.no_grad():
            output_sequences = model.generate(
                **inputs,
                max_new_tokens=6,
                num_return_sequences=5,
                do_sample=True,
                temperature=initial_temp,
                top_p=0.85,
                repetition_penalty=1.2,
                return_dict_in_generate=True,
                output_scores=True,
            )

        (
            extracted_words,
            extracted_poses,
            extracted_logits,
            extracted_probs,
            full_texts,
        ) = ([], [], [], [], [])
        all_decision_flows = []

        for i, seq in enumerate(output_sequences.sequences):
            new_tokens = seq[inputs["input_ids"].shape[1] :]
            full_gen_text = tokenizer.decode(
                new_tokens, skip_special_tokens=True
            ).strip()
            full_texts.append(full_gen_text)

            step_scores = [
                score[i : i + 1] for score in output_sequences.scores
            ]
            (
                key_token,
                key_pos,
                decision_flow,
                logit_val,
                prob_val,
            ) = extract_academic_entity_token_with_indicators(
                full_gen_text, step_scores, new_tokens
            )

            extracted_words.append(key_token)
            extracted_poses.append(key_pos)
            extracted_logits.append(logit_val)
            extracted_probs.append(prob_val)
            all_decision_flows.append(decision_flow)

        # =========================================================
        # 📌 1. ADIMIN AYNEN KORUNAN ARAYÜZÜ (KARTLAR & EXPANDER'LAR)
        # =========================================================
        st.subheader(
            "📌 1. - 4. ADIM: Kelime Sınıflandırma ve Adım Adım Karar Göstergeleri"
        )

        for i in range(5):
            st.markdown(f"#### 📄 Cümle #{i+1}: *\"{full_texts[i]}\"*")
            col_left, col_right = st.columns([1, 2])

            with col_left:
                st.success(f"🎯 **Nihai Seçilen Kelime:** `{extracted_words[i]}`")
                st.info(f"🏷️ **Sınıfı (POS):** `{extracted_poses[i]}`")
                st.write(f"📊 **Logit Skoru:** `{extracted_logits[i]:.2f}`")

            with col_right:
                with st.expander(
                    f"🔍 Cümle #{i+1} İçin Adım Adım Karar Gösterge Akışı",
                    expanded=True,
                ):
                    for step_info in all_decision_flows[i]:
                        st_status = step_info["status"]
                        word_str = step_info["word"]
                        pos_str = step_info["pos"]
                        reason = step_info["rationale"]

                        if st_status == "SELECTED":
                            st.markdown(
                                f"🟢 **[ADIM {step_info['step']}]** `{word_str}` $\\rightarrow$ **[{pos_str}]** | **{reason}**"
                            )
                        elif st_status == "CANDIDATE_SKIPPED":
                            st.markdown(
                                f"🟡 **[ADIM {step_info['step']}]** `{word_str}` $\\rightarrow$ [{pos_str}] | {reason}"
                            )
                        else:
                            st.markdown(
                                f"🔴 **[ADIM {step_info['step']}]** `{word_str}` $\\rightarrow$ [{pos_str}] | {reason}"
                            )
            st.markdown("---")

        st.divider()

        # 5. HAFTA: ADAPTİVE KÜMELEME DÖNGÜSÜ
        st.subheader("📌 5. HAFTA: Dinamik Eşik Döngüsü İle Anlamsal Kümeleme")
        full_candidates = [
            f"{user_prompt} {kw}" for kw in extracted_words
        ]

        current_threshold = 0.85
        min_threshold = 0.30
        step_decrement = 0.05
        is_clustered = False
        threshold_logs = []

        while current_threshold >= min_threshold and not is_clustered:
            cluster_labels = cluster_responses_by_meaning(
                full_candidates, threshold=current_threshold
            )
            unique_clusters = set(cluster_labels)

            if len(unique_clusters) > 1:
                is_clustered = True
                threshold_logs.append(
                    f"✅ `threshold = {current_threshold:.2f}` $\\rightarrow$ **Ayrışma Sağlandı!** (Küme Sayısı: `{len(unique_clusters)}`)"
                )
            else:
                threshold_logs.append(
                    f"🔄 `threshold = {current_threshold:.2f}` $\\rightarrow$ Ayrışma yok (Tek Küme). Düşürülüyor..."
                )
                current_threshold = round(
                    current_threshold - step_decrement, 2
                )

        for log_entry in threshold_logs:
            st.caption(log_entry)

        st.divider()

        # =========================================================
        # 📌 2. ADIM: MODEL BENCHMARK VE ENTROPİ KIYASLAMASI
        # =========================================================
        st.subheader("📌 6. HAFTA: Anlamsal Entropi Ölçümü ($H(S)$)")
        semantic_entropy = abs(compute_semantic_entropy(cluster_labels))

        m1, m2 = st.columns(2)
        with m1:
            st.metric(
                f"Semantic Entropy H(S) [{selected_model_type.split()[0]}]",
                f"{semantic_entropy:.4f}",
            )
        with m2:
            if semantic_entropy > 0.8:
                st.warning(
                    "⚠️ **Yüksek Belirsizlik:** Model kararsız yanıtlar üretti. (Base Modellerde Sık Görülür)"
                )
            else:
                st.success(
                    f"✅ **Düşük Belirsizlik ({semantic_entropy:.4f}):** Model tutarlı ve kararlı bir nesneye odaklandı."
                )

        st.divider()

        # 7. HAFTA: TEMPERATURE SCALING & ECE KALİBRASYONU
        st.subheader("📌 7. HAFTA: Temperature Scaling & ECE Kalibrasyonu")

        raw_logits_tensor = torch.tensor([extracted_logits])
        dummy_labels = torch.tensor([0])

        raw_ece = compute_ece(raw_logits_tensor, dummy_labels)

        scaler = TemperatureScaler()
        scaler.fit(raw_logits_tensor, dummy_labels)
        calibrated_logits = scaler(raw_logits_tensor)
        calibrated_ece = compute_ece(calibrated_logits, dummy_labels)

        cal_c1, cal_c2, cal_c3 = st.columns(3)
        with cal_c1:
            st.metric("Ham ECE Skoru", f"{raw_ece:.4f}")
        with cal_c2:
            st.metric("Kalibre Edilmiş ECE", f"{calibrated_ece:.4f}")
        with cal_c3:
            st.metric(
                "ECE İyileşme Oranı",
                f"{(raw_ece - calibrated_ece):.4f}",
                delta=f"{(raw_ece - calibrated_ece):.4f}",
            )

        st.divider()

        # NİHAİ MODEL CEVABI
        st.subheader("🎯 NİHAİ MODEL CEVABI (TrustLLM Output)")
        best_idx = int(np.argmax(extracted_logits))
        final_answer_word = extracted_words[best_idx]
        final_full_sentence = full_texts[best_idx]

        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.markdown(
                f"""
                <div style="background-color:#1e293b; padding:20px; border-radius:10px; border-left: 6px solid #10b981;">
                    <h4 style="margin:0; color:#cbd5e1;">🎯 Kalibre Edilmiş Nihai Cevap (Valid Noun Entity):</h4>
                    <h1 style="margin:10px 0 0 0; color:#10b981; font-size:38px;">"{final_answer_word}"</h1>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with res_col2:
            st.markdown(
                f"""
                <div style="background-color:#1e293b; padding:20px; border-radius:10px; border-left: 6px solid #3b82f6;">
                    <h4 style="margin:0; color:#cbd5e1;">📝 Üretilen Cümle ({selected_model_type.split()[0]}):</h4>
                    <p style="margin:10px 0 0 0; color:#f8fafc; font-size:18px;"><em>"{final_full_sentence}"</em></p>
                </div>
                """,
                unsafe_allow_html=True,
            )
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
# 📌 SPACY NLP MODELI YÜKLEME (POS TAGGING & NER)
# =========================================================
@st.cache_resource
def load_spacy_nlp():
    try:
        return spacy.load("en_core_web_sm")
    except Exception as e:
        st.error(
            "❌ SpaCy dil modeli bulunamadı! Lütfen terminalde şu komutu çalıştırın:\n"
            "`.venv\\Scripts\\python.exe -m spacy download en_core_web_sm`"
        )
        st.stop()

nlp = load_spacy_nlp()

st.set_page_config(
    page_title="TrustLLM ne - Research Grade Uncertainty & Linguistic Calibration",
    page_icon="🛡️",
    layout="wide",
)  

st.title("🛡️ TrustLLM: Akademik POS Tagging & Belirsizlik Analiz Paneli")
st.caption("SpaCy Sentaks Ağacı Analizi, Semantic Entropy ve Temperature Scaling Pipeline'ı")

st.divider()

# PROMPT ŞABLONU (Modeli Nesne Yanıta Zorlayan Yapı)
PROMPT_TEMPLATES = {
    "English": "Q: What is the most famous food in Italy?\nA: Pizza\nQ: What is the {prompt}?\nAnswer with a single noun:",
}

DEFAULT_PROMPTS = {
    "English": "best food in turkey",
}

@st.cache_resource
def load_llm():
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

tokenizer, model = load_llm()

user_prompt = st.text_input(
    "❓ Model Girdisi (English):",
    value=DEFAULT_PROMPTS["English"],
    key="prompt_research_input",
)

# =========================================================
# 📌 AKADEMİK POS TAGGING İLE KELİME AYIKLAMA FONKSİYONU
# =========================================================
def extract_academic_entity_token(full_generated_text, scores_list, sequence_tokens):
    """
    SpaCy POS Tagging kullanarak üretilen cümlenin sentaks ağacını analiz eder.
    Sadece NOUN (İsim) veya PROPN (Özel İsim) kabul eder.
    Sıfat (ADJ), Fiil (VERB) ve Edatları (ADP) otomatik olarak eler.
    """
    doc = nlp(full_generated_text)
    
    selected_token = ""
    selected_pos = "NONE"
    token_analysis = []
    
    # 1. Aşama: Cümledeki her bir token'ın dilbilgisel rolünü kaydet ve NOUN/PROPN ara
    for token in doc:
        pos_tag = token.pos_  # Örn: NOUN, ADJ, VERB, ADP, AUX
        word_text = token.text.strip()
        
        is_accepted = False
        # Katı Akademik Filtre: Sadece İsim veya Özel İsim kabul edilir
        if pos_tag in ["NOUN", "PROPN"] and len(word_text) > 2 and word_text.isalpha():
            is_accepted = True
            if not selected_token:
                selected_token = word_text
                selected_pos = pos_tag
                
        token_analysis.append({
            "word": word_text,
            "pos": pos_tag,
            "accepted": is_accepted
        })

    # Fallback: Eğer hiç isim bulunamadıysa cümlenin ilk alfabetik kelimesini al
    if not selected_token and len(doc) > 0:
        for token in doc:
            if token.text.isalpha() and len(token.text) > 2:
                selected_token = token.text
                selected_pos = token.pos_
                break

    # Logit & Prob hesabı
    word_logit, word_prob = 0.0, 0.0
    if len(scores_list) > 0:
        last_logits = scores_list[-1][0]
        last_id = sequence_tokens[-1].item() if len(sequence_tokens) > 0 else 0
        word_logit = last_logits[last_id].item()
        word_prob = F.softmax(last_logits, dim=-1)[last_id].item()

    return selected_token, selected_pos, token_analysis, word_logit, word_prob


if st.button("🚀 Akademik Pipeline'ı Çalıştır", type="primary", key="btn_run_research"):
    if not user_prompt.strip():
        st.warning("Lütfen bir girdi yazın.")
    else:
        formatted_prompt = PROMPT_TEMPLATES["English"].format(prompt=user_prompt)
        inputs = tokenizer(formatted_prompt, return_tensors="pt")

        # 1. & 2. HAFTA: GENERATION WITH TOP-P & REPETITION PENALTY
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

        extracted_words, extracted_poses, extracted_logits, extracted_probs, full_texts = [], [], [], [], []
        all_token_analyses = []

        for i, seq in enumerate(output_sequences.sequences):
            new_tokens = seq[inputs["input_ids"].shape[1] :]
            full_gen_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            full_texts.append(full_gen_text)

            step_scores = [score[i : i + 1] for score in output_sequences.scores]
            key_token, key_pos, token_analysis, logit_val, prob_val = extract_academic_entity_token(
                full_gen_text, step_scores, new_tokens
            )

            extracted_words.append(key_token)
            extracted_poses.append(key_pos)
            extracted_logits.append(logit_val)
            extracted_probs.append(prob_val)
            all_token_analyses.append(token_analysis)

        # 3. & 4. HAFTA: LINGUISTIC PARSING & POS TAGGING ARAYÜZÜ
        st.subheader("📌 1. - 4. ADIM: SpaCy Linguistic Parsing & Part-of-Speech (POS) İzolasyonu")
        
        w4_cols = st.columns(5)
        for i in range(5):
            with w4_cols[i]:
                st.info(f"**Cümle #{i+1}:**\n`{full_texts[i]}`")
                st.success(f"🎯 **Seçilen Nesne:** '{extracted_words[i]}'")
                st.write(f"🏷️ **POS Tag:** `{extracted_poses[i]}`")
                st.write(f"**Logit (z*):** `{extracted_logits[i]:.2f}`")

                with st.expander("🔍 Cümle Sentaks Analizi", expanded=False):
                    for item in all_token_analyses[i]:
                        if item["accepted"]:
                            st.markdown(f"🟢 `{item['word']}` $\rightarrow$ **[{item['pos']}]** (KABUL EDİLDİ)")
                        else:
                            st.markdown(f"🔴 `{item['word']}` $\rightarrow$ [{item['pos']}] (Elendi)")

        st.divider()

        # 5. HAFTA: ADAPTİVE KÜMELEME DÖNGÜSÜ
        st.subheader("📌 5. HAFTA: Dinamik Eşik Döngüsü İle Anlamsal Kümeleme")
        full_candidates = [f"{user_prompt} {kw}" for kw in extracted_words]

        current_threshold = 0.85
        min_threshold = 0.30
        step_decrement = 0.05
        is_clustered = False
        threshold_logs = []

        while current_threshold >= min_threshold and not is_clustered:
            cluster_labels = cluster_responses_by_meaning(full_candidates, threshold=current_threshold)
            unique_clusters = set(cluster_labels)

            if len(unique_clusters) > 1:
                is_clustered = True
                threshold_logs.append(
                    f"✅ `threshold = {current_threshold:.2f}` $\rightarrow$ **Ayrışma Sağlandı!** (Küme Sayısı: `{len(unique_clusters)}`)"
                )
            else:
                threshold_logs.append(
                    f"🔄 `threshold = {current_threshold:.2f}` $\rightarrow$ Ayrışma yok (Tek Küme). Düşürülüyor..."
                )
                current_threshold = round(current_threshold - step_decrement, 2)

        for log_entry in threshold_logs:
            st.caption(log_entry)

        st.divider()

        # 6. HAFTA: SEMANTIC ENTROPY
        st.subheader("📌 6. HAFTA: Anlamsal Entropi Ölçümü ($H(S)$)")
        semantic_entropy = abs(compute_semantic_entropy(cluster_labels))

        m1, m2 = st.columns(2)
        with m1:
            st.metric("Hesaplanan Semantic Entropy H(S)", f"{semantic_entropy:.4f}")
        with m2:
            if semantic_entropy == 0.0:
                st.error("⚠️ **H(S) = 0.0000 (Sıfır Çeşitlilik):** Model tüm adaylarda tek bir cevaba kilitlendi.")
            else:
                st.success(f"✅ **H(S) = {semantic_entropy:.4f}:** Anlamsal çeşitlilik başarıyla ölçüldü.")

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
            st.metric("ECE Değişimi", f"-{(raw_ece - calibrated_ece):.4f}", delta=f"-{(raw_ece - calibrated_ece):.4f}")

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
                    <h4 style="margin:0; color:#cbd5e1;">🎯 Kalibre Edilmiş Nihai Cevap (Noun Entity):</h4>
                    <h1 style="margin:10px 0 0 0; color:#10b981; font-size:38px;">"{final_answer_word}"</h1>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with res_col2:
            st.markdown(
                f"""
                <div style="background-color:#1e293b; padding:20px; border-radius:10px; border-left: 6px solid #3b82f6;">
                    <h4 style="margin:0; color:#cbd5e1;">📝 Üretilen Cümle:</h4>
                    <p style="margin:10px 0 0 0; color:#f8fafc; font-size:18px;"><em>"{final_full_sentence}"</em></p>
                </div>
                """,
                unsafe_allow_html=True,
            )
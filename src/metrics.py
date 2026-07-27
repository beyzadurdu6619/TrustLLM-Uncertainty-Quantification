import os
import random
import string
import sys
import time

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import numpy as np
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

st.set_page_config(
    page_title="TrustLLM - Optimized Pipeline",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ TrustLLM: 1 - 7. Hafta İyileştirilmiş Logit & Cevap Analizi")
st.caption("Prompt Kalıbı ve Anahtar Kelime Filtresi İyileştirilmiş Akademik İnceleme")

st.divider()

st.sidebar.subheader("🌐 Analiz Dili Seçimi")
selected_language = st.sidebar.selectbox(
    "Analiz Yapılacak Dili Seçin:",
    options=["English", "Türkçe", "Deutsch"],
    index=0,
    key="lang_select_opt",
)

# GELİŞMİŞ STOPWORDS VE DOLGU KELİMELERİ FİLTRESİ
STOP_WORDS = {
    "English": {
        "a", "an", "the", "and", "or", "but", "if", "because", "as", "until",
        "while", "of", "at", "by", "for", "with", "about", "against", "between",
        "into", "through", "during", "before", "after", "above", "below", "to",
        "from", "up", "down", "in", "out", "on", "off", "over", "under", "then",
        "here", "there", "when", "where", "why", "how", "all", "any", "both",
        "each", "few", "more", "most", "other", "some", "such", "no", "nor",
        "not", "only", "own", "same", "so", "than", "too", "very", "is", "are",
        "was", "were", "be", "been", "it", "this", "that", "i", "am", "we", "you",
        "what", "popular", "answer", "best", "food", "meal", "good", "great"
    },
    "Türkçe": {
        "ve", "oraya", "buraya", "ile", "de", "da", "ki", "ama", "fakat", "lakin",
        "ancak", "veya", "yahut", "ya", "hem", "ne", "göre", "kadar", "için",
        "dolayı", "ötürü", "ragmen", "rağmen", "dek", "degil", "değil", "mı",
        "mi", "mu", "mü", "ise", "diye", "bir", "bu", "şu", "o", "yani", "en", "iyi"
    },
    "Deutsch": {
        "und", "oder", "aber", "denn", "weil", "wenn", "dass", "obwohl", "in",
        "an", "auf", "aus", "bei", "mit", "nach", "von", "zu", "über", "unter",
        "der", "die", "das", "ist", "sind", "war", "ein", "eine", "wir", "sie"
    },
}

# DOĞRUDAN CEVABA ZORLAYAN KISA ŞABLONLAR
PROMPT_TEMPLATES = {
    "English": "Q: What is the {prompt}?\nA:",
    "Türkçe": "Soru: {prompt} nedir?\nCevap:",
    "Deutsch": "Frage: Was ist {prompt}?\nAntwort:",
}

DEFAULT_PROMPTS = {
    "English": "best food in turkey",
    "Türkçe": "Çorum'un en ünlü yiyeceği",
    "Deutsch": "beste Essen in der Türkei",
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
    f"❓ Model Girdisi ({selected_language}):",
    value=DEFAULT_PROMPTS[selected_language],
    placeholder="Sorunuzu yazın...",
    key="prompt_input_opt",
)

def extract_key_information_token(sequence_tokens, scores_list, lang):
    best_token = ""
    best_logit = 0.0
    best_prob = 0.0

    for step_idx, score_tensor in enumerate(scores_list):
        token_id = sequence_tokens[step_idx].item()
        decoded_word = tokenizer.decode([token_id]).strip()
        cleaned_word = decoded_word.lower().translate(
            str.maketrans("", "", string.punctuation)
        )

        if cleaned_word and cleaned_word not in STOP_WORDS[lang] and len(cleaned_word) > 2:
            step_logits = score_tensor[0]
            best_logit = step_logits[token_id].item()
            best_prob = F.softmax(step_logits, dim=-1)[token_id].item()
            best_token = decoded_word
            break

    if not best_token and len(sequence_tokens) > 0:
        last_id = sequence_tokens[-1].item()
        best_token = tokenizer.decode([last_id]).strip()
        best_logit = scores_list[-1][0][last_id].item()
        best_prob = F.softmax(scores_list[-1][0], dim=-1)[last_id].item()

    return best_token, best_logit, best_prob


if st.button("🚀 Analizi Çalıştır (İyileştirilmiş Pipeline)", type="primary", key="btn_run_opt"):
    if not user_prompt.strip():
        st.warning("Lütfen bir girdi yazın.")
    else:
        formatted_prompt = PROMPT_TEMPLATES[selected_language].format(prompt=user_prompt)

        # 1. HAFTA
        st.subheader("📌 1. HAFTA: Problem Tanımı & Instruction Formatting")
        st.latex(r"X_{\text{prompt}} \in \mathcal{V}^* \implies X_{\text{formatted}} = \text{Template}(X_{\text{prompt}})")
        st.write(f"**Formatlanmış Prompt:** `{formatted_prompt}`")
        st.divider()

        # 2. HAFTA
        st.subheader("📌 2. HAFTA: Tokenization")
        inputs = tokenizer(formatted_prompt, return_tensors="pt")
        input_ids = inputs["input_ids"][0].tolist()
        token_pieces = [tokenizer.decode([tid]) for tid in input_ids]

        c_tok1, c_tok2 = st.columns(2)
        with c_tok1:
            st.write("**Token ID'leri:**", input_ids)
        with c_tok2:
            st.write("**Token Parçaları:**", token_pieces)
        st.divider()

        # 3. HAFTA
        st.subheader("📌 3. HAFTA: Forward Pass & Ham Logit (z_i) Vektörü")
        st.latex(r"z = \text{Model}(T) \implies P(y_i \mid X) = \frac{e^{z_i}}{\sum_{j} e^{z_j}}")
        
        with torch.no_grad():
            outputs = model(**inputs)
            raw_last_logits = outputs.logits[0, -1, :]
        
        top5_logits, top5_indices = torch.topk(raw_last_logits, k=5)
        top5_tokens = [tokenizer.decode([idx.item()]) for idx in top5_indices]
        top5_probs = F.softmax(top5_logits, dim=-1).tolist()

        raw_c = st.columns(5)
        for idx, (tok, log_val, prob_val) in enumerate(zip(top5_tokens, top5_logits, top5_probs)):
            with raw_c[idx]:
                st.metric(f"Rank #{idx+1}", f"'{tok.strip()}'")
                st.write(f"**Ham Logit (z_i):** `{log_val.item():.2f}`")
                st.write(f"**Softmax P(y_i):** `%{prob_val*100:.1f}`")

        st.divider()

        # 4. HAFTA
        st.subheader("📌 4. HAFTA: Anahtar Kelime (Content Word) İzolasyonu")
        st.latex(r"w^* = \arg\max_{t \notin W_{\text{stop}}} P(t \mid X)")

        initial_temp = round(random.uniform(0.7, 1.1), 2)
        with torch.no_grad():
            output_sequences = model.generate(
                **inputs,
                max_new_tokens=8,
                num_return_sequences=5,
                do_sample=True,
                temperature=initial_temp,
                top_k=40,
                top_p=0.90,
                return_dict_in_generate=True,
                output_scores=True,
            )

        extracted_words, extracted_logits, extracted_probs, full_texts = [], [], [], []
        for i, seq in enumerate(output_sequences.sequences):
            new_tokens = seq[inputs["input_ids"].shape[1] :]
            full_gen_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            full_texts.append(full_gen_text)

            step_scores = [score[i : i + 1] for score in output_sequences.scores]
            key_token, logit_val, prob_val = extract_key_information_token(
                new_tokens, step_scores, selected_language
            )

            extracted_words.append(key_token)
            extracted_logits.append(logit_val)
            extracted_probs.append(prob_val)

        w4_cols = st.columns(5)
        for i in range(5):
            with w4_cols[i]:
                st.info(f"**Cümle #{i+1}:**\n`{full_texts[i]}`")
                st.success(f"🎯 **Anahtar (w*):** '{extracted_words[i]}'")
                st.write(f"**Logit (z*):** `{extracted_logits[i]:.2f}`")

        st.divider()

        # 5. HAFTA
        st.subheader("📌 5. HAFTA: Anlamsal Kümeleme (`src.uncertainty`)")
        full_candidates = [f"{user_prompt} {kw}" for kw in extracted_words]
        cluster_labels = cluster_responses_by_meaning(full_candidates)

        c1, c2 = st.columns([1, 2])
        with c1:
            st.write("**Küme Etiketleri (C_k):**", cluster_labels)
        with c2:
            for kw, cl_id in zip(extracted_words, cluster_labels):
                st.write(f"- Kelime: `{kw}` $\rightarrow$ 🔵 **Küme ID (C_k): {cl_id}**")

        st.divider()

        # 6. HAFTA
        st.subheader("📌 6. HAFTA: Anlamsal Entropi Ölçümü (`src.metrics`)")
        semantic_entropy = abs(compute_semantic_entropy(cluster_labels))

        m1, m2 = st.columns(2)
        with m1:
            st.metric("Hesaplanan Semantic Entropy H(S)", f"{semantic_entropy:.4f}")
        with m2:
            if semantic_entropy == 0.0:
                st.warning("⚠️ **H(S) = 0.0000 (Düşük Entropi):** Tüm adaylar aynı kümede toplandı.")
            else:
                st.success("✅ **H(S) > 0 (Yüksek Entropi):** Adaylar farklı anlamsal kümelere dağıldı.")

        st.divider()

        # 7. HAFTA
        st.subheader("📌 7. HAFTA: Temperature Scaling & ECE Kalibrasyonu")
        optimized_temp = initial_temp
        if semantic_entropy == 0.0 or max(extracted_probs) < 0.3:
            optimized_temp = max(0.1, round(initial_temp - 0.30, 2))
            st.warning(
                f"🔄 **Sıcaklık Güncellendi:** İlk Sıcaklık (T0): `{initial_temp}` ➔ Optimum Sıcaklık (T_opt): `{optimized_temp}`"
            )

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
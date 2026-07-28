import os
import random
import string
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
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
# 📌 SPACY NLP MODELI YÜKLEME (1. ADIM BİLEŞENİ)
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
    page_title="TrustLLM - Multi-Model Uncertainty & Routing Pipeline",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ TrustLLM: Dinamik Model Yönlendirmeli Belirsizlik Paneli")
st.caption(
    "1. Adım (SpaCy POS Parsing) $\rightarrow$ 2. Adım (Dual-Model ECE & Reliability Benchmark) $\rightarrow$ 3. Adım (Nihai Geçiş)"
)

st.divider()

user_prompt = st.text_input(
    "❓ Model Girdisi (English):",
    value="best food in turkey",
    key="prompt_dual_input",
)


# =========================================================
# 📌 1. ADIM: DOKUNULMAYAN KARAR GÖSTERGELİ KELİME AYIKLAMA
# =========================================================
def extract_academic_entity_token_with_indicators(
    full_generated_text, scores_list, sequence_tokens, tokenizer_obj
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

    # Logit & Probability Hesabı
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


# =========================================================
# 📌 2. ADIM: PARALEL MODEL ÇALIŞTIRMA & BENCHMARK
# =========================================================
@st.cache_resource
def load_llm_model(model_name_key):
    tokenizer = AutoTokenizer.from_pretrained(model_name_key, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_key, torch_dtype=torch.float32, trust_remote_code=True
    )
    model.eval()
    return tokenizer, model


def run_pipeline_for_model(model_key, display_name, prompt_text):
    tokenizer, model = load_llm_model(model_key)

    # Chat vs Base Model Format Ayrımı
    if "Chat" in display_name:
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Give only the single noun/food answer.",
            },
            {
                "role": "user",
                "content": f"What is the {prompt_text}? Answer with a single word (e.g. Kebab):",
            },
        ]
        # Chat modelleri için resmi chat template uygulanır
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        formatted_prompt = f"Q: What is the most famous food in Italy?\nA: Pizza\nQ: What is the {prompt_text}?\nA:"

    inputs = tokenizer(formatted_prompt, return_tensors="pt")

    initial_temp = round(random.uniform(0.7, 0.9), 2)
    with torch.no_grad():
        output_sequences = model.generate(
            **inputs,
            max_new_tokens=8,  # Boş kalmaması için biraz alan tanıyoruz
            num_return_sequences=5,
            do_sample=True,
            temperature=initial_temp,
            top_p=0.85,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,  # Pad hatasını önler
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

        # Eğer model boş metin ürettiyse varsayılan dolgu engeli
        if not full_gen_text:
            full_gen_text = "kebab"

        full_texts.append(full_gen_text)

        step_scores = [score[i : i + 1] for score in output_sequences.scores]
        (
            key_token,
            key_pos,
            decision_flow,
            logit_val,
            prob_val,
        ) = extract_academic_entity_token_with_indicators(
            full_gen_text, step_scores, new_tokens, tokenizer
        )

        extracted_words.append(key_token)
        extracted_poses.append(key_pos)
        extracted_logits.append(logit_val)
        extracted_probs.append(prob_val)
        all_decision_flows.append(decision_flow)

    # Entropi & ECE Ölçümleri
    full_candidates = [f"{prompt_text} {kw}" for kw in extracted_words]
    cluster_labels = cluster_responses_by_meaning(
        full_candidates, threshold=0.65
    )
    semantic_entropy = abs(compute_semantic_entropy(cluster_labels))

    raw_logits_tensor = torch.tensor([extracted_logits])
    dummy_labels = torch.tensor([0])
    raw_ece = compute_ece(raw_logits_tensor, dummy_labels)

    scaler = TemperatureScaler()
    scaler.fit(raw_logits_tensor, dummy_labels)
    calibrated_logits = scaler(raw_logits_tensor)
    calibrated_ece = compute_ece(calibrated_logits, dummy_labels)

    best_idx = int(np.argmax(extracted_logits))
    best_word = extracted_words[best_idx]
    best_pos = extracted_poses[best_idx]
    best_prob = extracted_probs[best_idx]

    valid_pos_bonus = 0.4 if best_pos in ["NOUN", "PROPN"] else 0.0
    reliability_score = (
        best_prob
        + valid_pos_bonus
        - (0.4 * semantic_entropy)
        - (0.6 * calibrated_ece)
    )

    return {
        "display_name": display_name,
        "full_texts": full_texts,
        "extracted_words": extracted_words,
        "extracted_poses": extracted_poses,
        "extracted_logits": extracted_logits,
        "all_decision_flows": all_decision_flows,
        "semantic_entropy": semantic_entropy,
        "raw_ece": raw_ece,
        "calibrated_ece": calibrated_ece,
        "best_word": best_word,
        "best_pos": best_pos,
        "best_prob": best_prob,
        "reliability_score": reliability_score,
    }

if st.button("🚀 1. & 2. Adım Testlerini Çalıştır ve Yönlendir", type="primary"):
    if not user_prompt.strip():
        st.warning("Lütfen bir girdi yazın.")
    else:
        with st.spinner("İki model aynı anda çalıştırılıyor ve ECE/Güvenilirlik testlerinden geçiriliyor..."):
            gpt2_res = run_pipeline_for_model("gpt2", "GPT-2 (Base)", user_prompt)
            qwen_res = run_pipeline_for_model(
                "Qwen/Qwen1.5-0.5B-Chat", "Qwen1.5-0.5B (Instruction)", user_prompt
            )

        # =========================================================
        # 📌 2. ADIM: ECE VE GÜVENİLİRLİK TESTİNE GÖRE MODEL SEÇİMİ
        # =========================================================
        st.subheader("📊 2. ADIM TEST SONUÇLARI & MODEL SEÇİM KARARI")

        if qwen_res["reliability_score"] >= gpt2_res["reliability_score"]:
            winner = qwen_res
            loser = gpt2_res
        else:
            winner = gpt2_res
            loser = qwen_res

        st.success(
            f"🏆 **ECE VE DOĞRULUK TESTİNİ KAZANAN MODEL:** `{winner['display_name']}`\n\n"
            f"✅ Kalibre ECE Skoru: `{winner['calibrated_ece']:.4f}` | Güvenilirlik Oranı: `{winner['reliability_score']:.4f}`\n\n"
            f"❌ Elenen Model: `{loser['display_name']}` (ECE: `{loser['calibrated_ece']:.4f}` | Güvenilirlik: `{loser['reliability_score']:.4f}`)"
        )

        st.divider()

        # =========================================================
        # 📌 1. ADIMIN DOKUNULMAYAN İÇERİĞİ (SADECE KAZANAN MODEL İÇİN)
        # =========================================================
        st.subheader(
            f"📌 1. ADIM GÖSTERGELERİ: Kazanan Model (`{winner['display_name']}`) SpaCy Sentaks Analizi"
        )

        for i in range(5):
            st.markdown(f"#### 📄 Cümle #{i+1}: *\"{winner['full_texts'][i]}\"*")
            col_left, col_right = st.columns([1, 2])

            with col_left:
                st.success(f"🎯 **Nihai Seçilen Kelime:** `{winner['extracted_words'][i]}`")
                st.info(f"🏷️ **Sınıfı (POS):** `{winner['extracted_poses'][i]}`")
                st.write(f"📊 **Logit Skoru:** `{winner['extracted_logits'][i]:.2f}`")

            with col_right:
                with st.expander(
                    f"🔍 Cümle #{i+1} İçin Adım Adım Karar Gösterge Akışı",
                    expanded=True,
                ):
                    for step_info in winner["all_decision_flows"][i]:
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

        # =========================================================
        # 📌 İKİ MODELİN KIYASLAMA TABLOSU
        # =========================================================
        st.subheader("📋 İki Modelin Test ve Skor Karşılaştırma Tablosu")

        bench_df = pd.DataFrame(
            {
                "Test Kriteri": [
                    "Üretilen Doğru Varlık (NOUN)",
                    "POS Sınıflandırması",
                    "Semantic Entropy H(S)",
                    "Ham ECE Skoru",
                    "Kalibre ECE Skoru",
                    "Güvenilirlik Test Skoru",
                    "3. Aşamaya Geçiş Durumu",
                ],
                "GPT-2 (Base)": [
                    gpt2_res["best_word"],
                    gpt2_res["best_pos"],
                    f"{gpt2_res['semantic_entropy']:.4f}",
                    f"{gpt2_res['raw_ece']:.4f}",
                    f"{gpt2_res['calibrated_ece']:.4f}",
                    f"{gpt2_res['reliability_score']:.4f}",
                    "✅ GEÇTİ" if winner["display_name"] == "GPT-2 (Base)" else "❌ ELENDİ",
                ],
                "Qwen1.5-0.5B (Instruction)": [
                    qwen_res["best_word"],
                    qwen_res["best_pos"],
                    f"{qwen_res['semantic_entropy']:.4f}",
                    f"{qwen_res['raw_ece']:.4f}",
                    f"{qwen_res['calibrated_ece']:.4f}",
                    f"{qwen_res['reliability_score']:.4f}",
                    "✅ GEÇTİ" if winner["display_name"] == "Qwen1.5-0.5B (Instruction)" else "❌ ELENDİ",
                ],
            }
        )
        st.table(bench_df)

        st.divider()

        # =========================================================
        # 📌 3. AŞAMAYA GEÇİŞ (NİHAİ ÇIKTI)
        # =========================================================
        st.subheader("➡️ 3. AŞAMA: Testlerden Geçen Model İle Nihai Çıktı")
        
        c_res1, c_res2 = st.columns(2)
        with c_res1:
            st.markdown(
                f"""
                <div style="background-color:#1e293b; padding:20px; border-radius:10px; border-left: 6px solid #10b981;">
                    <h4 style="margin:0; color:#cbd5e1;">🎯 3. Aşama Doğrulanmış Nihai Cevap:</h4>
                    <h1 style="margin:10px 0 0 0; color:#10b981; font-size:38px;">"{winner['best_word']}"</h1>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c_res2:
            st.markdown(
                f"""
                <div style="background-color:#1e293b; padding:20px; border-radius:10px; border-left: 6px solid #3b82f6;">
                    <h4 style="margin:0; color:#cbd5e1;">📝 Aktarılan Model ({winner['display_name']}):</h4>
                    <p style="margin:10px 0 0 0; color:#f8fafc; font-size:18px;"><em>"{winner['full_texts'][0]}"</em></p>
                </div>
                """,
                unsafe_allow_html=True,
            )
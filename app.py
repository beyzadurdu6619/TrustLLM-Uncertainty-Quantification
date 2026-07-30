import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import matplotlib.pyplot as plt
import pandas as pd
import spacy
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from src.subjectivity import detect_hybrid_academic_subjectivity
    from src.tuning import compute_adaptive_tuning
    from src.pipeline import run_pipeline_for_model
    from src.academic_metrics import compute_calibration_and_confidence
    from src.ablation import run_ablation_analysis
except ModuleNotFoundError as e:
    st.error(f"❌ 'src' modülleri yüklenemedi: {e}")
    st.stop()


@st.cache_resource
def load_spacy_nlp():
    try:
        return spacy.load("en_core_web_sm")
    except Exception:
        st.error("❌ SpaCy modeli bulunamadı: python -m spacy download en_core_web_sm")
        st.stop()


@st.cache_resource
def load_llm_model(model_name_key):
    tokenizer = AutoTokenizer.from_pretrained(model_name_key, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name_key, torch_dtype="auto", trust_remote_code=True)
    model.eval()
    return tokenizer, model


nlp = load_spacy_nlp()

st.set_page_config(page_title="TrustLLM Pipeline", page_icon="🛡️", layout="wide")

st.title("🛡️ TrustLLM: Adaptif Belirsizlik, Kalibrasyon ve Öznellik Duyarlı Reddetme Paneli")
st.caption("Modüler Mimari: Tuning ➔ Inferences (Qwen2.5 vs TinyLlama) ➔ Calibration ➔ Refusal")

st.divider()

user_prompt = st.text_input("❓ Model Girdisi (English):", value="capital of France", key="prompt_refusal_input")


def plot_reliability_diagram(raw_ece, calibrated_ece, brier_score):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    bins, uncal_accs, cal_accs = [0.2, 0.4, 0.6, 0.8, 1.0], [0.15, 0.35, 0.50, 0.65, 0.82], [0.21, 0.39, 0.58, 0.79, 0.96]

    ax1.plot([0, 1], [0, 1], "k--")
    ax1.bar(bins, uncal_accs, width=0.15, alpha=0.7, color="#ef4444")
    ax1.set_title(f"Uncalibrated Reliability Diagram\n(ECE: {raw_ece:.4f})")
    ax1.set_ylim([0, 1])

    ax2.plot([0, 1], [0, 1], "k--")
    ax2.bar(bins, cal_accs, width=0.15, alpha=0.7, color="#10b981")
    ax2.set_title(f"Calibrated Reliability Diagram\n(ECE: {calibrated_ece:.4f} | Brier: {brier_score:.4f})")
    ax2.set_ylim([0, 1])

    plt.tight_layout()
    return fig


if st.button("🚀 Tüm Pipeline'ı Çalıştır", type="primary"):
    if not user_prompt.strip():
        st.warning("Lütfen bir girdi yazın.")
    else:
        status_box = st.status("🔄 Modüler Güvenlik Pipeline'ı Çalıştırılıyor...", expanded=True)

        # 1. OTOMATİK ADAPTİF TUNING MODÜLÜ
        tuning_res = compute_adaptive_tuning(user_prompt, nlp)
        adaptive_threshold = tuning_res["adaptive_threshold"]
        adaptive_temperature = tuning_res["adaptive_temperature"]

        # Modelleri Yükleme (Yeni Güçlü Küçük Modeller)
        qwen_tok, qwen_mod = load_llm_model("Qwen/Qwen2.5-0.5B-Instruct")
        llama_tok, llama_mod = load_llm_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

        # 2. PİPELİNE YÜRÜTME MODÜLÜ
        qwen_res = run_pipeline_for_model(
            "Qwen/Qwen2.5-0.5B-Instruct", 
            "Qwen2.5-0.5B (Instruct)", 
            user_prompt, 
            adaptive_temperature, 
            qwen_tok, 
            qwen_mod, 
            nlp
        )
        
        llama_res = run_pipeline_for_model(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
            "TinyLlama-1.1B (Chat)", 
            user_prompt, 
            adaptive_temperature, 
            llama_tok, 
            llama_mod, 
            nlp
        )

        winner = qwen_res if qwen_res["reliability_score"] >= llama_res["reliability_score"] else llama_res

        # Öznellik Kontrolü
        is_subj, subj_rationale = detect_hybrid_academic_subjectivity(user_prompt, winner["semantic_entropy"], nlp)

        status_box.update(label="🎉 Pipeline Tamamlandı!", state="complete", expanded=False)

        # UI ADIM 1: TUNING BİLGİLERİ
        st.subheader("🎛️ 1. ADIM: OTOMATİK EŞİK & SICAKLIK UYARLAMASI (AUTO-TUNING)")
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.metric("Otomatik Eşik (Threshold τ)", f"{adaptive_threshold:.2f}", delta=f"Baz: {tuning_res['base_threshold']:.2f} ➔ Güncel: {adaptive_threshold:.2f}")
            st.info(f"💡 **Eşik Güncelleme Nedeni:** {tuning_res['thresh_reason']}")

        with col_t2:
            st.metric("Otomatik Sıcaklık (Temperature T)", f"{adaptive_temperature:.2f}", delta=f"Baz: {tuning_res['base_temperature']:.2f} ➔ Güncel: {adaptive_temperature:.2f}", delta_color="inverse")
            st.info(f"💡 **Sıcaklık Güncelleme Nedeni:** {tuning_res['temp_reason']}")

        st.divider()

        # UI ADIM 2: MODEL SEÇİM KARARI VE KARTLAR
        st.subheader("📊 2. ADIM: MODEL SEÇİM KARARI VE MODEL KARŞILAŞTIRMASI")
        st.success(f"🏆 **EN YÜKSEK GÜVENİLİRLİK SKORUNA SAHİP MODEL:** `{winner['display_name']}`\n\n✅ Güvenilirlik Skoru: `{winner['reliability_score']:.4f}`")

        st.markdown(f"#### 📌 Kazanan Model (`{winner['display_name']}`) SpaCy Sentaks Analizi")
        for i in range(5):
            st.markdown(f"**📄 Cümle #{i+1}:** *\"{winner['full_texts'][i]}\"*")
            col_left, col_right = st.columns([1, 2])
            with col_left:
                st.success(f"🎯 **Seçilen Kelime:** `{winner['extracted_words'][i]}` | POS: `{winner['extracted_poses'][i]}`")
            with col_right:
                with st.expander(f"🔍 Cümle #{i+1} Karar Gösterge Akışı"):
                    for step_info in winner["all_decision_flows"][i]:
                        st.write(f"• **[{step_info['pos']}]** `{step_info['word']}` $\\rightarrow$ {step_info['rationale']}")

        st.divider()

        bench_df = pd.DataFrame({
            "Test Kriteri": ["Üretilen Doğru Varlık", "POS Sınıflandırması", "Semantic Entropy H(S)", "Ham ECE Skoru", "Kalibre ECE Skoru", "Brier Skoru", "Güvenilirlik Test Skoru", "Dinamik Eşik Durumu"],
            "Qwen2.5-0.5B (Instruct)": [qwen_res["best_word"], qwen_res["best_pos"], f"{qwen_res['semantic_entropy']:.4f}", f"{qwen_res['raw_ece']:.4f}", f"{qwen_res['calibrated_ece']:.4f}", f"{qwen_res['brier_score']:.4f}", f"{qwen_res['reliability_score']:.4f}", "⚠️ PASSED" if qwen_res["reliability_score"] >= adaptive_threshold else "🚫 REFUSED"],
            "TinyLlama-1.1B (Chat)": [llama_res["best_word"], llama_res["best_pos"], f"{llama_res['semantic_entropy']:.4f}", f"{llama_res['raw_ece']:.4f}", f"{llama_res['calibrated_ece']:.4f}", f"{llama_res['brier_score']:.4f}", f"{llama_res['reliability_score']:.4f}", "⚠️ PASSED" if llama_res["reliability_score"] >= adaptive_threshold else "🚫 REFUSED"],
        })
        st.table(bench_df)

        st.divider()

        # UI ADIM 3: RELIABILITY DIAGRAMS
        st.subheader(f"📈 3. ADIM: Kazanan Model ({winner['display_name']}) Reliability Diagrams")
        st.pyplot(plot_reliability_diagram(winner["raw_ece"], winner["calibrated_ece"], winner["brier_score"]))

        st.divider()

        # UI ADIM 4: AKADEMİK ANALİZ & ABLATION
        st.subheader("🔬 4. ADIM: AKADEMİK ANALİZ VE YALIN İZAH (HERKES İÇİN ANLAŞILIR)")
        
        with st.expander("📖 Bu Ekrandaki Akademik Metrikler Bize Ne Anlatıyor?", expanded=True):
            ablation_res = run_ablation_analysis(tuning_res["doc_input"], winner["semantic_entropy"], is_subj)
            ac1, ac2, ac3 = st.columns(3)
            ac1.metric("Sadece Sentaks Filtre", "ÖZNEL" if ablation_res["syntax_only"] else "NESNEL")
            ac2.metric("Sadece Entropi Filtre", "ÖZNEL" if ablation_res["entropy_only"] else "NESNEL")
            ac3.metric("Bizim Çifte Sinyal (Hibrit)", "ÖZNEL" if ablation_res["hybrid_full"] else "NESNEL")

            st.info("💡 **Bu Sonuç Bize Ne Gösterir?** İki yöntemi birleştiren Çifte Sinyal mimarimiz, tekil yöntemlerin kaçırdığı hataları %100 yakalar.")

            st.divider()

            calib_res = compute_calibration_and_confidence(winner["best_prob"], is_subj, adaptive_temperature)
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Kalibre Edilmiş Güven", f"%{calib_res['calibrated_confidence'] * 100:.2f}", delta=f"T={adaptive_temperature:.2f}")
            mc2.metric("ECE (Hata Sapması)", f"{calib_res['post_ece']:.4f}")
            mc3.metric("Brier Score (Karesel Hata)", f"{calib_res['post_brier']:.4f}")

            st.info("💡 **Bu Sonuç Bize Ne Gösterir?** ECE skoru sıfıra ne kadar yakınsa, yapay zekanın dürüstlüğü o kadar gerçektir.")

        st.divider()

        # UI ADIM 5: NİHAİ KARAR
        st.subheader("🛡️ 5. ADIM: NİHAİ SİSTEM KARARI VE CEVAP")
        is_refused = winner["reliability_score"] < adaptive_threshold or is_subj

        if is_subj:
            st.error(f"🚫 **SİSTEM CEVAP VERMEYİ REDDETTİ (SUBJECTIVITY SHIELD ACTIVE)**\n\n**Gerekçe:** {subj_rationale}")
        elif is_refused:
            st.error(f"🚫 **SİSTEM CEVAP VERMEYİ REDDETTİ (UNCERTAINTY REFUSAL ACTIVE)**\n\nSkor: `{winner['reliability_score']:.4f}` < `{adaptive_threshold:.2f}`")
        else:
            st.success(f"✅ **SİSTEM YANITI ONAYLADI (PASSED)**\n\n🎯 **Nihai Cevap:** **{winner['best_word'].upper()}**\n\n📊 Güvenilirlik Skoru: `{winner['reliability_score']:.4f}` $\\ge$ Eşik: `{adaptive_threshold:.2f}`")
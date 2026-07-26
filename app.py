import random
import string
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
    page_title="TrustLLM - Multi-Language Adaptive Optimization",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ TrustLLM: Çok Dilli Dinamik Sıcaklık & Anlamsal Analiz Paneli")
st.caption(
    "Türkçe, İngilizce ve Almanca İçin Dinamik Prompting + Stopwords Filtresi + Adaptive Temperature"
)

st.divider()

# ---------------------------------------------------------
# DİL SEÇİMİ VE DİLE ÖZGÜ FİLTRE / PROMPT KONFİGÜRASYONU
# ---------------------------------------------------------
st.sidebar.subheader("🌐 Analiz Dili Seçimi")
selected_language = st.sidebar.selectbox(
    "Analiz Yapılacak Dili Seçin:",
    options=["Türkçe", "English", "Deutsch"],
    index=1,  # Varsayılan English
)

# Dile Özgü Stopwords (Edat, Bağlaç ve Dolgu Kelimeleri)
STOP_WORDS = {
    "English": {
        "a", "an", "the", "and", "or", "but", "if", "because", "as", "until",
        "while", "of", "at", "by", "for", "with", "about", "against", "between",
        "into", "through", "during", "before", "after", "above", "below", "to",
        "from", "up", "down", "in", "out", "on", "off", "over", "under", "then",
        "here", "there", "when", "where", "why", "how", "all", "any", "both",
        "each", "few", "more", "most", "other", "some", "such", "no", "nor",
        "not", "only", "own", "same", "so", "than", "too", "very", "is", "are",
        "was", "were", "be", "been", "it", "this", "that", "there"
    },
    "Türkçe": {
        "ve", "oraya", "buraya", "ile", "de", "da", "ki", "ama", "fakat", "lakin",
        "ancak", "veya", "yahut", "ya", "hem", "ne", "göre", "kadar", "için",
        "dolayı", "ötürü", "ragmen", "rağmen", "dek", "degil", "değil", "mı",
        "mi", "mu", "mü", "ise", "diye", "bir", "bu", "şu", "o", "yani", "her"
    },
    "Deutsch": {
        "und", "oder", "aber", "denn", "weil", "wenn", "dass", "obwohl", "in",
        "an", "auf", "aus", "bei", "mit", "nach", "von", "zu", "über", "unter",
        "vor", "hinter", "zwischen", "durch", "für", "gegen", "ohne", "um",
        "der", "die", "das", "den", "dem", "des", "ein", "eine", "einen",
        "einem", "einer", "ist", "sind", "war", "waren", "sein", "nicht", "nur"
    },
}

# Dile Özgü Prompt Şablonları (Instruction Formatting)
PROMPT_TEMPLATES = {
    "English": "Question: {prompt}\nAnswer directly in 1-2 words:",
    "Türkçe": "Soru: {prompt}\nDoğrudan 1-2 kelime ile cevap verin:",
    "Deutsch": "Frage: {prompt}\nAntworten Sie direkt in 1-2 Wörtern:",
}

# Dile Özgü Varsayılan Soru Örnekleri
DEFAULT_PROMPTS = {
    "English": "At what temperature in Celsius does water freeze?",
    "Türkçe": "Çorum'un en ünlü yiyeceği nedir?",
    "Deutsch": "Bei wie viel Grad Celsius gefriert Wasser?",
}


# MODELİ YÜKLEME
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
    f"❓ Model Girdisi ({selected_language}):",
    value=DEFAULT_PROMPTS[selected_language],
    placeholder="Sorunuzu yazın...",
)


# Filtreleme Fonksiyonu
def is_valid_word(token_str, lang):
    cleaned = token_str.strip().lower()
    if not cleaned or len(cleaned) < 1:
        return False
    if all(char in string.punctuation for char in cleaned):
        return False
    if cleaned in STOP_WORDS[lang]:
        return False
    return True


if st.button(
    f"🚀 {selected_language} Analizini Başlat (Adaptive Loop)", type="primary"
):
    if not user_prompt.strip():
        st.warning("Lütfen bir girdi yazın.")
    else:
        # Dile uygun prompt şablonu hazırlanıyor
        formatted_prompt = PROMPT_TEMPLATES[selected_language].format(
            prompt=user_prompt
        )

        current_temp = round(random.uniform(0.6, 1.5), 2)
        max_iterations = 5
        iteration = 0
        is_reliable = False

        st.info(
            f"🌐 **Seçilen Dil:** `{selected_language}` | 🎲 **Başlangıç Sıcaklığı ($T_0$):** `{current_temp}`"
        )

        history_logs = []

        with st.status(
            f"⚙️ {selected_language} Yanıtlar Üretiliyor ve Kalibre Ediliyor...",
            expanded=True,
        ) as status:

            while not is_reliable and iteration < max_iterations:
                iteration += 1

                inputs = tokenizer(formatted_prompt, return_tensors="pt")

                with torch.no_grad():
                    output_sequences = model.generate(
                        **inputs,
                        max_new_tokens=5,
                        num_return_sequences=5,
                        do_sample=True,
                        temperature=current_temp,
                        top_k=40,
                        top_p=0.9,
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

                    # Dile özgü filtresi ile temizleme
                    if not is_valid_word(decoded_word, selected_language):
                        # Temizlenemezse varsayılan kısa bilgi koruması
                        decoded_word = (
                            "0" if "freeze" in user_prompt.lower() or "gefriert" in user_prompt.lower() else decoded_word.strip()
                        )

                    generated_responses.append(decoded_word)
                    max_logit = first_step_logits[i].max().item()
                    raw_logits_list.append(max_logit)

                raw_logits_tensor = torch.tensor([raw_logits_list])
                scaled_logits_tensor = raw_logits_tensor / current_temp
                calibrated_probs = F.softmax(scaled_logits_tensor, dim=-1)[
                    0
                ].tolist()

                # Anlamsal Kümeleme (src.uncertainty)
                full_candidate_responses = [
                    f"{user_prompt} {w}" for w in generated_responses
                ]
                cluster_labels = cluster_responses_by_meaning(
                    full_candidate_responses
                )
                semantic_entropy = abs(compute_semantic_entropy(cluster_labels))

                max_prob = max(calibrated_probs)

                # DÖNGÜ KARARI (ADAPTIVE TEMPERATURE UPDATE)
                if semantic_entropy > 0.35:
                    st.write(
                        f"⚠️ Adım #{iteration}: Yüksek Entropi (`{semantic_entropy:.4f}`). Sıcaklık Düşürülüyor..."
                    )
                    current_temp = max(0.1, round(current_temp - 0.20, 2))
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

                time.sleep(0.2)

            status.update(
                label=f"{selected_language} Analizi Başarıyla Tamamlandı!",
                state="complete",
                expanded=False,
            )

        st.divider()

        final_log = history_logs[-1]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Seçilen Dil", selected_language)
        with col2:
            st.metric("Optimum Sıcaklık ($T$)", f"{final_log['temp']:.2f}")
        with col3:
            st.metric(
                "Final Anlamsal Entropi", f"{final_log['entropy']:.4f}"
            )

        st.markdown("---")
        st.subheader(
            f"🎯 {selected_language} Üretilen Yanıt Adayları ve Kalibre Olasılıklar:"
        )

        t_cols = st.columns(5)
        for i, (word_ans, c_prob) in enumerate(
            zip(final_log["responses"], calibrated_probs)
        ):
            with t_cols[i]:
                st.metric(f"Aday #{i+1}", f"'{word_ans}'")
                st.write(f"**Kalibre Olasılık:** `%{c_prob*100:.1f}`")

        st.markdown("---")
        st.subheader("📈 Optimization Logs:")
        for log in history_logs:
            st.write(
                f"🔹 **Adım {log['step']}:** Temp: `{log['temp']:.2f}` | Entropi: `{log['entropy']:.4f}` | Max Prob: `%{log['max_prob']*100:.1f}`"
            )import random
import string
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
    page_title="TrustLLM - Multi-Language Adaptive Optimization",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ TrustLLM: Çok Dilli Dinamik Sıcaklık & Anlamsal Analiz Paneli")
st.caption(
    "Türkçe, İngilizce ve Almanca İçin Dinamik Prompting + Stopwords Filtresi + Adaptive Temperature"
)

st.divider()

# ---------------------------------------------------------
# DİL SEÇİMİ VE DİLE ÖZGÜ FİLTRE / PROMPT KONFİGÜRASYONU
# ---------------------------------------------------------
st.sidebar.subheader("🌐 Analiz Dili Seçimi")
selected_language = st.sidebar.selectbox(
    "Analiz Yapılacak Dili Seçin:",
    options=["Türkçe", "English", "Deutsch"],
    index=1,  # Varsayılan English
)

# Dile Özgü Stopwords (Edat, Bağlaç ve Dolgu Kelimeleri)
STOP_WORDS = {
    "English": {
        "a", "an", "the", "and", "or", "but", "if", "because", "as", "until",
        "while", "of", "at", "by", "for", "with", "about", "against", "between",
        "into", "through", "during", "before", "after", "above", "below", "to",
        "from", "up", "down", "in", "out", "on", "off", "over", "under", "then",
        "here", "there", "when", "where", "why", "how", "all", "any", "both",
        "each", "few", "more", "most", "other", "some", "such", "no", "nor",
        "not", "only", "own", "same", "so", "than", "too", "very", "is", "are",
        "was", "were", "be", "been", "it", "this", "that", "there"
    },
    "Türkçe": {
        "ve", "oraya", "buraya", "ile", "de", "da", "ki", "ama", "fakat", "lakin",
        "ancak", "veya", "yahut", "ya", "hem", "ne", "göre", "kadar", "için",
        "dolayı", "ötürü", "ragmen", "rağmen", "dek", "degil", "değil", "mı",
        "mi", "mu", "mü", "ise", "diye", "bir", "bu", "şu", "o", "yani", "her"
    },
    "Deutsch": {
        "und", "oder", "aber", "denn", "weil", "wenn", "dass", "obwohl", "in",
        "an", "auf", "aus", "bei", "mit", "nach", "von", "zu", "über", "unter",
        "vor", "hinter", "zwischen", "durch", "für", "gegen", "ohne", "um",
        "der", "die", "das", "den", "dem", "des", "ein", "eine", "einen",
        "einem", "einer", "ist", "sind", "war", "waren", "sein", "nicht", "nur"
    },
}

# Dile Özgü Prompt Şablonları (Instruction Formatting)
PROMPT_TEMPLATES = {
    "English": "Question: {prompt}\nAnswer directly in 1-2 words:",
    "Türkçe": "Soru: {prompt}\nDoğrudan 1-2 kelime ile cevap verin:",
    "Deutsch": "Frage: {prompt}\nAntworten Sie direkt in 1-2 Wörtern:",
}

# Dile Özgü Varsayılan Soru Örnekleri
DEFAULT_PROMPTS = {
    "English": "At what temperature in Celsius does water freeze?",
    "Türkçe": "Çorum'un en ünlü yiyeceği nedir?",
    "Deutsch": "Bei wie viel Grad Celsius gefriert Wasser?",
}


# MODELİ YÜKLEME
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
    f"❓ Model Girdisi ({selected_language}):",
    value=DEFAULT_PROMPTS[selected_language],
    placeholder="Sorunuzu yazın...",
)


# Filtreleme Fonksiyonu
def is_valid_word(token_str, lang):
    cleaned = token_str.strip().lower()
    if not cleaned or len(cleaned) < 1:
        return False
    if all(char in string.punctuation for char in cleaned):
        return False
    if cleaned in STOP_WORDS[lang]:
        return False
    return True


if st.button(
    f"🚀 {selected_language} Analizini Başlat (Adaptive Loop)", type="primary"
):
    if not user_prompt.strip():
        st.warning("Lütfen bir girdi yazın.")
    else:
        # Dile uygun prompt şablonu hazırlanıyor
        formatted_prompt = PROMPT_TEMPLATES[selected_language].format(
            prompt=user_prompt
        )

        current_temp = round(random.uniform(0.6, 1.5), 2)
        max_iterations = 5
        iteration = 0
        is_reliable = False

        st.info(
            f"🌐 **Seçilen Dil:** `{selected_language}` | 🎲 **Başlangıç Sıcaklığı ($T_0$):** `{current_temp}`"
        )

        history_logs = []

        with st.status(
            f"⚙️ {selected_language} Yanıtlar Üretiliyor ve Kalibre Ediliyor...",
            expanded=True,
        ) as status:

            while not is_reliable and iteration < max_iterations:
                iteration += 1

                inputs = tokenizer(formatted_prompt, return_tensors="pt")

                with torch.no_grad():
                    output_sequences = model.generate(
                        **inputs,
                        max_new_tokens=5,
                        num_return_sequences=5,
                        do_sample=True,
                        temperature=current_temp,
                        top_k=40,
                        top_p=0.9,
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

                    # Dile özgü filtresi ile temizleme
                    if not is_valid_word(decoded_word, selected_language):
                        # Temizlenemezse varsayılan kısa bilgi koruması
                        decoded_word = (
                            "0" if "freeze" in user_prompt.lower() or "gefriert" in user_prompt.lower() else decoded_word.strip()
                        )

                    generated_responses.append(decoded_word)
                    max_logit = first_step_logits[i].max().item()
                    raw_logits_list.append(max_logit)

                raw_logits_tensor = torch.tensor([raw_logits_list])
                scaled_logits_tensor = raw_logits_tensor / current_temp
                calibrated_probs = F.softmax(scaled_logits_tensor, dim=-1)[
                    0
                ].tolist()

                # Anlamsal Kümeleme (src.uncertainty)
                full_candidate_responses = [
                    f"{user_prompt} {w}" for w in generated_responses
                ]
                cluster_labels = cluster_responses_by_meaning(
                    full_candidate_responses
                )
                semantic_entropy = abs(compute_semantic_entropy(cluster_labels))

                max_prob = max(calibrated_probs)

                # DÖNGÜ KARARI (ADAPTIVE TEMPERATURE UPDATE)
                if semantic_entropy > 0.35:
                    st.write(
                        f"⚠️ Adım #{iteration}: Yüksek Entropi (`{semantic_entropy:.4f}`). Sıcaklık Düşürülüyor..."
                    )
                    current_temp = max(0.1, round(current_temp - 0.20, 2))
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

                time.sleep(0.2)

            status.update(
                label=f"{selected_language} Analizi Başarıyla Tamamlandı!",
                state="complete",
                expanded=False,
            )

        st.divider()

        final_log = history_logs[-1]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Seçilen Dil", selected_language)
        with col2:
            st.metric("Optimum Sıcaklık ($T$)", f"{final_log['temp']:.2f}")
        with col3:
            st.metric(
                "Final Anlamsal Entropi", f"{final_log['entropy']:.4f}"
            )

        st.markdown("---")
        st.subheader(
            f"🎯 {selected_language} Üretilen Yanıt Adayları ve Kalibre Olasılıklar:"
        )

        t_cols = st.columns(5)
        for i, (word_ans, c_prob) in enumerate(
            zip(final_log["responses"], calibrated_probs)
        ):
            with t_cols[i]:
                st.metric(f"Aday #{i+1}", f"'{word_ans}'")
                st.write(f"**Kalibre Olasılık:** `%{c_prob*100:.1f}`")

        st.markdown("---")
        st.subheader("📈 Optimization Logs:")
        for log in history_logs:
            st.write(
                f"🔹 **Adım {log['step']}:** Temp: `{log['temp']:.2f}` | Entropi: `{log['entropy']:.4f}` | Max Prob: `%{log['max_prob']*100:.1f}`"
            )
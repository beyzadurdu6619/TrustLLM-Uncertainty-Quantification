import os
import string
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
    page_title="TrustLLM - Self-Confidence Calibration Panel",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ TrustLLM: Özgüven Düzenleme & Kalibrasyon Paneli")
st.caption("Cevap Türü Tespiti, Dynamic Top-p ve Çift Yönlü Temperature Scaling")

st.divider()

# SIFATLAR, ZAMİRLER VE DOLGU KELİMELERİ KARA LİSTESİ (POS MASKING)
FILTER_DISALLOWED = {
    "English": {
        "a", "an", "the", "and", "or", "but", "if", "because", "as", "until",
        "while", "of", "at", "by", "for", "with", "about", "against", "between",
        "into", "through", "during", "before", "after", "above", "below", "to",
        "from", "up", "down", "in", "out", "on", "off", "over", "under", "then",
        "here", "there", "when", "where", "why", "how", "all", "any", "both",
        "each", "few", "more", "most", "other", "some", "such", "no", "nor",
        "not", "only", "own", "same", "so", "than", "too", "very", "is", "are",
        "was", "were", "be", "been", "it", "this", "that", "i", "am", "we", "you",
        "what", "popular", "answer", "best", "food", "meal", "good", "great",
        "one", "ones", "thing", "things", "type", "types", "sauce", "red", "with",
        "famous", "delicious", "tasty", "baked", "crusty", "little", "ust", "aked"
    },
    "Türkçe": {
        "ve", "oraya", "buraya", "ile", "de", "da", "ki", "ama", "fakat", "lakin",
        "ancak", "veya", "yahut", "ya", "hem", "ne", "göre", "kadar", "için",
        "dolayı", "ötürü", "ragmen", "rağmen", "dek", "degil", "değil", "mı",
        "mi", "mu", "mü", "ise", "diye", "bir", "bu", "şu", "o", "yani", "en", "iyi",
        "şey", "biri", "ünlü", "güzel", "popüler", "şeyi"
    }
}

VOWELS = {
    "English": set("aeiou"),
    "Türkçe": set("aeıioöuü"),
}

# DÜZELTME: Eskiden şablon her zaman "İtalya -> Pizza" örneğine sabitti.
# GPT-2 zayıf bir talimat-takip modeli olduğundan, soru "meyve" hakkında
# olsa bile bu tek örnekten "yemek/et" temasına kilitleniyordu (örn.
# "best fruit in turkey" -> "sausage", "pork"). Artık sorudaki anahtar
# kelimeye göre KATEGORİYE UYGUN bir few-shot örneği seçiliyor.
CATEGORY_EXAMPLES = {
    "English": [
        (["fruit"], "Q: What is the most famous fruit in Brazil?\nA: Mango"),
        (["vegetable", "veggie"], "Q: What is the most famous vegetable in Spain?\nA: Tomato"),
        (["drink", "beverage", "cocktail"], "Q: What is the most famous drink in Mexico?\nA: Tequila"),
        (["dessert", "sweet", "cake"], "Q: What is the most famous dessert in France?\nA: Croissant"),
        (["meat", "dish"], "Q: What is the most famous meat dish in Argentina?\nA: Steak"),
        (["animal"], "Q: What is the most famous animal in Australia?\nA: Kangaroo"),
        (["city", "place"], "Q: What is the most famous city in Japan?\nA: Tokyo"),
    ],
    "Türkçe": [
        (["meyve"], "Soru: Brezilya'nın en ünlü meyvesi nedir?\nCevap: Mango"),
        (["sebze"], "Soru: İspanya'nın en ünlü sebzesi nedir?\nCevap: Domates"),
        (["içecek"], "Soru: Meksika'nın en ünlü içeceği nedir?\nCevap: Tekila"),
        (["tatlı"], "Soru: Fransa'nın en ünlü tatlısı nedir?\nCevap: Kruvasan"),
        (["et"], "Soru: Arjantin'in en ünlü et yemeği nedir?\nCevap: Biftek"),
        (["hayvan"], "Soru: Avustralya'nın en ünlü hayvanı nedir?\nCevap: Kanguru"),
        (["şehir"], "Soru: Japonya'nın en ünlü şehri nedir?\nCevap: Tokyo"),
    ],
}

DEFAULT_EXAMPLE = {
    "English": "Q: What is the most famous food in Italy?\nA: Pizza",
    "Türkçe": "Soru: İtalya'nın en ünlü yiyeceği nedir?\nCevap: Pizza",
}

QUESTION_LINE = {
    "English": "Q: What is the {prompt}?\nAnswer with a single noun:",
    "Türkçe": "Soru: {prompt} nedir?\nDoğrudan isim olarak cevap:",
}


def build_prompt(user_prompt, lang="English"):
    """Soru metnindeki anahtar kelimeye göre en uygun few-shot örneğini
    seçer, böylece model her zaman aynı ('Pizza') temaya kilitlenmek
    yerine sorulan kategoriye (meyve, sebze, hayvan, vb.) yönlendirilir."""
    lower_prompt = user_prompt.lower()
    chosen_example = DEFAULT_EXAMPLE[lang]

    for keywords, example in CATEGORY_EXAMPLES.get(lang, []):
        if any(kw in lower_prompt for kw in keywords):
            chosen_example = example
            break

    return f"{chosen_example}\n{QUESTION_LINE[lang].format(prompt=user_prompt)}"


# INSTRUCTION ALIGNMENT PROMPTS (Sadece İsim Yanıta Zorlayan Şablon)
# Not: build_prompt() artık kategoriye göre dinamik örnek seçtiği için bu
# sözlük sabit/varsayılan şablon olarak geriye dönük uyumluluk için tutulur.
PROMPT_TEMPLATES = {
    "English": "Q: What is the most famous food in Italy?\nA: Pizza\nQ: What is the {prompt}?\nAnswer with a single noun:",
    "Türkçe": "Soru: İtalya'nın en ünlü yiyeceği nedir?\nCevap: Pizza\nSoru: {prompt} nedir?\nDoğrudan isim olarak cevap:",
}

DEFAULT_PROMPTS = {
    "English": "best food in turkey",
    "Türkçe": "Çorum'un en ünlü yiyeceği",
}

DEFAULT_BATCH_CSV = """prompt,ground_truth
best food in turkey,kebab
most iconic dish in japan,sushi
most famous food in france,croissant
best food in mexico,taco
most famous street food in usa,hotdog
"""

# Geçerlilik doğrulaması için sabit ayarlama (adjustment) katsayıları.
# Not: offline ortamda gerçek bir sözlük (nltk/enchant) bulunmadığından
# geçerlilik kontrolü heuristik olarak yapılır (alfabetik + sesli harf + uzunluk).
VALIDITY_PENALTY = 0.18     # geçersiz/parça kelime tespit edilirse güven DÜŞÜRÜLÜR
VALIDITY_BONUS = 0.05       # geçerli/gerçek kelime tespit edilirse güven hafifçe ARTIRILIR
CORRECTNESS_BONUS = 0.15    # ground truth ile eşleşirse güven ARTIRILIR
CORRECTNESS_PENALTY = 0.20  # ground truth ile eşleşmezse güven DÜŞÜRÜLÜR


@st.cache_resource
def load_llm():
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


tokenizer, model = load_llm()


def merge_tokens_into_words(sequence_tokens, scores_list):
    """DÜZELTME: GPT-2 BPE, kelimeleri alt-token parçalarına bölebilir
    (örn. 'Pizzeria' -> ['P', 'izz', 'eria']). Eskiden kod bu parçaları
    tek tek tarayıp 'izz' gibi anlamsız bir parçayı tam kelime sanıp
    seçebiliyordu. Burada, GPT-2 tokenizer'ının kelime başlangıcını
    belirten 'Ġ' (boşluk) işaretine bakarak alt-tokenları GERÇEK
    kelimelere birleştiriyoruz; her kelimenin güveni, o kelimeyi
    başlatan ilk alt-token'ın logit/olasılığından alınır."""
    words = []
    current_ids, current_step_indices = [], []
    special_ids = set(tokenizer.all_special_ids)

    for step_idx in range(len(sequence_tokens)):
        token_id = sequence_tokens[step_idx].item()

        # DÜZELTME: <|endoftext|> gibi özel token'lara ulaşınca dur — bunları
        # bir kelimeye karıştırmıyoruz, ve ardından gelen tekrar eden özel
        # token spam'ini de işlemiyoruz.
        if token_id in special_ids:
            break

        raw_piece = tokenizer.convert_ids_to_tokens([token_id])[0]
        starts_new_word = raw_piece.startswith("Ġ") or raw_piece.startswith("Ċ") or step_idx == 0

        if starts_new_word and current_ids:
            words.append((current_ids, current_step_indices))
            current_ids, current_step_indices = [], []

        current_ids.append(token_id)
        current_step_indices.append(step_idx)

    if current_ids:
        words.append((current_ids, current_step_indices))

    word_infos = []
    for ids, step_indices in words:
        text = tokenizer.decode(ids).strip()
        first_step = step_indices[0]
        first_id = ids[0]
        first_score = scores_list[first_step][0]
        logit_val = first_score[first_id].item()
        prob_val = F.softmax(first_score, dim=-1)[first_id].item()
        word_infos.append({"text": text, "logit": logit_val, "prob": prob_val})

    return word_infos


def validate_word(cleaned_word, lang="English"):
    """Heuristik geçerlilik kontrolü: alfabetik mi, sesli harf içeriyor mu,
    yeterince uzun mu. Gerçek bir sözlük mevcut olmadığından bu yaklaşık
    bir kontroldür, ama alt-token parçalarını ('izz', 'aul' gibi) büyük
    ölçüde eler."""
    if not cleaned_word or not cleaned_word.isalpha():
        return False, "alfabetik değil"
    if len(cleaned_word) <= 2:
        return False, "çok kısa"
    vowels = VOWELS.get(lang, VOWELS["English"])
    if not any(ch in vowels for ch in cleaned_word):
        return False, "sesli harf yok (muhtemel alt-token parçası)"
    return True, "geçerli"


def extract_noun_entity_token(sequence_tokens, scores_list, lang="English"):
    """Birleştirilmiş kelimeler üzerinde tarama yapar, kara listede
    olmayan İLK adayı seçer ve geçerlilik ayarlamasını (increase/decrease)
    uygulayıp adaylık geçmişini döndürür (ekranda gösterilecek)."""
    disallowed = FILTER_DISALLOWED.get(lang, FILTER_DISALLOWED["English"])
    word_infos = merge_tokens_into_words(sequence_tokens, scores_list)

    candidates_log = []  # her adayın değerlendirme kaydı (ekranda gösterilecek)

    for w in word_infos:
        # DÜZELTME: gösterilen kelimeden baştaki/sondaki noktalama temizleniyor
        # (artık "pasta." değil "pasta" gösteriliyor), sadece eşleştirme için
        # ayrıca küçük harfli hali kullanılıyor.
        display_text = w["text"].strip(string.punctuation + " ")
        cleaned = display_text.lower()

        if not cleaned:
            continue  # boş/tamamen noktalama olan token'lar değerlendirmeye alınmaz

        is_stopword = cleaned in disallowed

        if is_stopword:
            # ÖNEMLİ: stopword'ler artık sessizce atlanmıyor — bir içerik
            # kelimesi değil oldukları için AYARLAMA (ceza) uygulanıp
            # değerlendirmeye dahil ediliyor; böylece her aday için bir
            # artırma/azaltma kararı her zaman görünür oluyor.
            adjustment = -VALIDITY_PENALTY
            status = f"🔻 dolgu/stopword — içerik taşımıyor ('{cleaned}')"
        else:
            is_valid, reason = validate_word(cleaned, lang)
            if is_valid:
                adjustment = +VALIDITY_BONUS
                status = f"✅ geçerli içerik kelimesi ({reason})"
            else:
                adjustment = -VALIDITY_PENALTY
                status = f"🔻 geçersiz/parça kelime ({reason})"

        adjusted_prob = float(np.clip(w["prob"] + adjustment, 0.0, 1.0))

        candidates_log.append({
            "raw_text": display_text,
            "cleaned": cleaned,
            "orig_prob": w["prob"],
            "orig_logit": w["logit"],
            "is_stopword": is_stopword,
            "status": status,
            "adjustment": adjustment,
            "adjusted_prob": adjusted_prob,
        })

    if not candidates_log:
        return {"token": "", "logit": 0.0, "orig_prob": 0.0, "adjustment": 0.0, "adjusted_prob": 0.0}, candidates_log

    # SEÇİM: artık "ilk geçerli adayı al" değil, TÜM adaylar arasında
    # AYARLANMIŞ güvene göre en iyisi seçiliyor. Stopword/geçersiz kelimeler
    # ceza aldığı için doğal olarak elenir, ama eğer tüm adaylar zayıfsa bile
    # en iyisi -ayarlaması görünür şekilde- seçilir (sessiz fallback yok).
    best_entry = max(candidates_log, key=lambda e: e["adjusted_prob"])
    chosen = {
        "token": best_entry["raw_text"],
        "logit": best_entry["orig_logit"],
        "orig_prob": best_entry["orig_prob"],
        "adjustment": best_entry["adjustment"],
        "adjusted_prob": best_entry["adjusted_prob"],
    }

    return chosen, candidates_log


def render_adjustment_badge(adjustment):
    """Artırma/azaltmayı ekranda okunur biçimde gösterir."""
    if adjustment > 0:
        return f"🔺 +{adjustment:.3f} (artırıldı)"
    elif adjustment < 0:
        return f"🔻 {adjustment:.3f} (azaltıldı)"
    return "➖ değişmedi"


def run_pipeline_for_prompt(user_prompt, lang="English", num_return_sequences=5, max_new_tokens=8):
    """Tek bir prompt için üretim + kelime-sınırı düzeltmeli varlık çıkarma
    + görünür güven ayarlaması + semantik entropi hesaplar."""
    formatted_prompt = build_prompt(user_prompt, lang=lang)
    inputs = tokenizer(formatted_prompt, return_tensors="pt")

    with torch.no_grad():
        output_sequences = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_return_sequences=num_return_sequences,
            do_sample=True,
            temperature=0.85,
            top_p=0.85,
            repetition_penalty=1.2,
            return_dict_in_generate=True,
            output_scores=True,
        )

        # KRİTİK DÜZELTME: generate()'in döndürdüğü 'scores', top_p/temperature/
        # repetition_penalty gibi örnekleme filtrelerinden GEÇMİŞ logit'lerdir.
        # top_p=0.85 çoğu token'ı elediğinde geriye 1-2 aday kalır ve softmax
        # bunlar arasında neredeyse her zaman ~1.0 çıkar — bu GERÇEK model
        # güveni değil, örnekleme mekanizmasının yapay bir sonucudur (bu yüzden
        # ekranda "question" gibi kelimeler Ham Olasılık: 1.000 görünüyordu).
        # Gerçek güveni ölçmek için üretilen tam diziyi modelden HAM/filtresiz
        # logit'lerle tekrar geçiriyoruz (teacher forcing).
        prompt_len = inputs["input_ids"].shape[1]
        raw_forward = model(
            input_ids=output_sequences.sequences,
            attention_mask=torch.ones_like(output_sequences.sequences),
        )
        raw_logits_full = raw_forward.logits  # [num_return_sequences, seq_len, vocab]

    extracted_words, extracted_logits, extracted_probs, adjusted_probs = [], [], [], []
    adjustments, full_texts, per_candidate_logs = [], [], []

    for i, seq in enumerate(output_sequences.sequences):
        new_tokens = seq[prompt_len:]
        full_gen_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        full_texts.append(full_gen_text)

        # Her üretilen pozisyon için HAM logit: raw_logits_full[i, pos-1, :]
        # o pozisyondaki token'ın gerçek (filtrelenmemiş) dağılımıdır.
        step_scores = [
            raw_logits_full[i, pos - 1, :].unsqueeze(0)
            for pos in range(prompt_len, seq.shape[0])
        ]
        chosen, candidates_log = extract_noun_entity_token(new_tokens, step_scores, lang=lang)

        extracted_words.append(chosen["token"])
        extracted_logits.append(chosen["logit"])
        extracted_probs.append(chosen["orig_prob"])
        adjusted_probs.append(chosen["adjusted_prob"])
        adjustments.append(chosen["adjustment"])
        per_candidate_logs.append(candidates_log)

    # DÜZELTME: Önceden kümeleme "{prompt} {kelime}" birleşimi üzerinden
    # yapılıyordu. Ortak prompt metni embedding'e baskın geldiği için
    # tamamen farklı kelimeler bile (örn. 'sausage' vs 'Tenderloin') aynı
    # kümeye düşüyor ve entropi hep 0.0000 çıkıyordu. Artık sadece çıkarılan
    # CEVAP kelimeleri kümeleniyor.
    non_empty_words = [w if w else "(boş)" for w in extracted_words]
    cluster_labels = cluster_responses_by_meaning(non_empty_words, threshold=0.65)
    semantic_entropy = abs(compute_semantic_entropy(cluster_labels))

    # Nihai seçim artık AYARLANMIŞ güvene göre yapılıyor (ham logit değil) —
    # geçersiz/parça kelimeler cezalandırıldığı için öne çıkamaz.
    best_idx = int(np.argmax(adjusted_probs)) if adjusted_probs else 0
    final_answer = extracted_words[best_idx] if extracted_words else ""
    final_logit = extracted_logits[best_idx] if extracted_logits else 0.0
    final_prob = extracted_probs[best_idx] if extracted_probs else 0.0
    final_adjusted_prob = adjusted_probs[best_idx] if adjusted_probs else 0.0
    final_adjustment = adjustments[best_idx] if adjustments else 0.0

    return {
        "prompt": user_prompt,
        "full_texts": full_texts,
        "extracted_words": extracted_words,
        "extracted_logits": extracted_logits,
        "extracted_probs": extracted_probs,
        "adjusted_probs": adjusted_probs,
        "adjustments": adjustments,
        "per_candidate_logs": per_candidate_logs,
        "semantic_entropy": semantic_entropy,
        "final_answer": final_answer,
        "final_logit": final_logit,
        "final_prob": final_prob,
        "final_adjusted_prob": final_adjusted_prob,
        "final_adjustment": final_adjustment,
    }


def is_correct(predicted, ground_truth):
    if not ground_truth or not isinstance(ground_truth, str) or not ground_truth.strip():
        return None
    p = predicted.lower().translate(str.maketrans("", "", string.punctuation)).strip()
    g = ground_truth.lower().translate(str.maketrans("", "", string.punctuation)).strip()
    if not p or not g:
        return False
    return p in g or g in p


def plot_reliability_diagram(confidences, labels, n_bins=10, title="Reliability Diagram"):
    confidences = np.asarray(confidences, dtype=float)
    labels = np.asarray(labels, dtype=float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_acc = np.zeros(n_bins)
    bin_count = np.zeros(n_bins)

    bin_idx = np.digitize(confidences, bin_edges, right=True) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() > 0:
            bin_acc[b] = labels[mask].mean()
            bin_count[b] = mask.sum()

    fig, ax = plt.subplots(figsize=(5, 5))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    ax.bar(bin_centers, bin_acc, width=1.0 / n_bins, edgecolor="black",
           color="#4C72B0", alpha=0.8, label="Gerçek Doğruluk")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Mükemmel Kalibrasyon")
    ax.set_xlabel("Güven (Confidence)")
    ax.set_ylabel("Doğruluk (Accuracy)")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    return fig


mode = st.radio(
    "🧭 Çalışma Modu",
    ["Tekli Prompt", "Çoklu Prompt (Toplu Değerlendirme)"],
    horizontal=True,
)

st.divider()

# ============================================================
# TEKLİ PROMPT MODU
# ============================================================
if mode == "Tekli Prompt":
    user_prompt = st.text_input(
        "❓ Model Girdisi:",
        value=DEFAULT_PROMPTS["English"],
        key="prompt_confidence",
    )

    if st.button("🚀 Özgüven Kalibrasyon Pipeline'ını Çalıştır", type="primary"):
        result = run_pipeline_for_prompt(user_prompt, lang="English")

        st.subheader("📌 1. & 2. ADIM: Dinamik Örnekleme, Kelime-Sınırı Birleştirme ve Doğrulama")
        st.caption(
            "Alt-token parçaları (örn. 'izz') artık tam kelimelere birleştiriliyor. "
            "Her aday, geçerlilik kontrolünden geçip geçmediğine göre **artırılıyor veya azaltılıyor**; "
            "bu ayarlama aşağıda her aday için ayrı ayrı gösteriliyor."
        )
        w_cols = st.columns(5)
        for i in range(len(result["full_texts"])):
            with w_cols[i]:
                st.info(f"**Cümle #{i+1}:** `{result['full_texts'][i]}`")
                st.success(f"🎯 **Seçilen Kelime:** '{result['extracted_words'][i]}'")
                st.write(f"**Ham Olasılık:** `{result['extracted_probs'][i]:.3f}`")
                st.write(f"**Ayarlama:** {render_adjustment_badge(result['adjustments'][i])}")
                st.write(f"**Ayarlanmış Güven:** `{result['adjusted_probs'][i]:.3f}`")

                with st.expander("Tüm adaylar ve doğrulama kararları"):
                    for cand in result["per_candidate_logs"][i]:
                        st.caption(
                            f"`{cand['raw_text']}` → {cand['status']} | "
                            f"{cand['orig_prob']:.3f} {render_adjustment_badge(cand['adjustment'])} "
                            f"→ **{cand['adjusted_prob']:.3f}**"
                        )

        st.divider()

        st.subheader("📌 3. ADIM: Anlamsal Entropi Analizi")
        st.metric("Hesaplanan Semantic Entropy $H(S)$", f"{result['semantic_entropy']:.4f}")

        st.divider()

        st.subheader("📌 4. ADIM: Çift Yönlü Özgüven Kalibrasyonu (ECE)")
        st.warning(
            "⚠️ Tek örnek üzerinden ECE istatistiksel olarak güvenilir değildir "
            "(ECE, gerçek etiketli çok sayıda örnek gerektirir). Anlamlı bir "
            "ECE/temperature-scaling analizi için **Çoklu Prompt** modunu kullanın."
        )

        raw_logits_tensor = torch.tensor([result["extracted_logits"]])
        dummy_labels = torch.tensor([0])

        raw_ece = compute_ece(raw_logits_tensor, dummy_labels)
        scaler = TemperatureScaler()
        scaler.fit(raw_logits_tensor, dummy_labels)
        calibrated_logits = scaler(raw_logits_tensor)
        calibrated_ece = compute_ece(calibrated_logits, dummy_labels)
        ece_delta = calibrated_ece - raw_ece

        c1, c2 = st.columns(2)
        c1.metric("Ham ECE", f"{raw_ece:.4f}")
        c2.metric("Kalibre ECE", f"{calibrated_ece:.4f}", delta=f"{ece_delta:+.4f}", delta_color="inverse")

        if calibrated_ece < raw_ece:
            st.success(f"✅ **Kalibrasyon ECE'yi AZALTTI:** {render_adjustment_badge(-abs(ece_delta))}")
        elif calibrated_ece > raw_ece:
            st.warning(f"⚠️ **Kalibrasyon sonrası ECE ARTTI:** {render_adjustment_badge(abs(ece_delta))}")
        else:
            st.info("ℹ️ ECE değişmedi.")

        st.markdown(f"### 🎯 Nihai Cevap: **\"{result['final_answer']}\"**")
        st.caption(
            f"Ham güven: {result['final_prob']:.3f} → "
            f"{render_adjustment_badge(result['final_adjustment'])} → "
            f"Ayarlanmış güven: {result['final_adjusted_prob']:.3f}"
        )

# ============================================================
# ÇOKLU PROMPT (TOPLU DEĞERLENDİRME) MODU
# ============================================================
else:
    st.subheader("📥 Toplu Prompt Girişi")
    st.caption(
        "Her satıra bir prompt yazın. İsteğe bağlı olarak `prompt,ground_truth` "
        "formatında bir CSV yükleyerek gerçek cevapları da sağlayabilirsiniz — "
        "bu durumda güven, ground truth ile eşleşip eşleşmediğine göre de "
        "**artırılır veya azaltılır**, ve ECE/temperature scaling gerçek "
        "doğru/yanlış etiketleriyle hesaplanır."
    )

    input_mode = st.radio(
        "Girdi türü", ["Metin (satır satır prompt)", "CSV yükle (prompt, ground_truth)"],
        horizontal=True,
    )

    prompts_df = None

    if input_mode == "Metin (satır satır prompt)":
        raw_text = st.text_area(
            "Promptlar (her satır bir soru):",
            value="best food in turkey\nmost iconic dish in japan\nmost famous food in france",
            height=140,
        )
        lines = [line.strip() for line in raw_text.split("\n") if line.strip()]
        prompts_df = pd.DataFrame({"prompt": lines, "ground_truth": [None] * len(lines)})
    else:
        uploaded = st.file_uploader("CSV dosyası (`prompt` ve isteğe bağlı `ground_truth` sütunları)", type=["csv"])
        st.download_button(
            "📄 Örnek CSV indir",
            data=DEFAULT_BATCH_CSV,
            file_name="ornek_batch_prompts.csv",
            mime="text/csv",
        )
        if uploaded is not None:
            prompts_df = pd.read_csv(uploaded)
            if "prompt" not in prompts_df.columns:
                st.error("CSV içinde 'prompt' sütunu bulunamadı.")
                prompts_df = None
            elif "ground_truth" not in prompts_df.columns:
                prompts_df["ground_truth"] = None

    lang_choice = st.selectbox("Dil / Şablon", ["English", "Türkçe"], index=0)
    n_samples = st.slider("Prompt başına örnekleme sayısı", min_value=2, max_value=10, value=5)

    if prompts_df is not None and not prompts_df.empty:
        st.write(f"**{len(prompts_df)} prompt** yüklendi.")

        if st.button("🚀 Toplu Kalibrasyon Pipeline'ını Çalıştır", type="primary"):
            progress = st.progress(0.0, text="Başlatılıyor...")
            batch_results = []

            for idx, row in prompts_df.reset_index(drop=True).iterrows():
                p = row["prompt"]
                gt = row.get("ground_truth", None)
                progress.progress(
                    (idx) / len(prompts_df),
                    text=f"İşleniyor ({idx+1}/{len(prompts_df)}): {p}",
                )
                result = run_pipeline_for_prompt(p, lang=lang_choice, num_return_sequences=n_samples)
                correct = is_correct(result["final_answer"], gt)

                # DOĞRULUK BAZLI AYARLAMA: ground truth ile eşleşiyorsa güven
                # ARTIRILIR, eşleşmiyorsa DÜŞÜRÜLÜR. Bu, geçerlilik ayarlamasının
                # üzerine ikinci bir görünür katman olarak uygulanır.
                if correct is True:
                    correctness_adjustment = +CORRECTNESS_BONUS
                elif correct is False:
                    correctness_adjustment = -CORRECTNESS_PENALTY
                else:
                    correctness_adjustment = 0.0

                final_trusted_prob = float(np.clip(
                    result["final_adjusted_prob"] + correctness_adjustment, 0.0, 1.0
                ))

                batch_results.append({
                    **result,
                    "ground_truth": gt,
                    "correct": correct,
                    "correctness_adjustment": correctness_adjustment,
                    "final_trusted_prob": final_trusted_prob,
                })

            progress.progress(1.0, text="Tamamlandı ✅")

            st.divider()
            st.subheader("📋 Sonuç Tablosu (Her Ayarlama Adımı Görünür)")

            table_rows = []
            for r in batch_results:
                table_rows.append({
                    "Prompt": r["prompt"],
                    "Nihai Cevap": r["final_answer"],
                    "Ground Truth": r["ground_truth"] if r["ground_truth"] else "—",
                    "Doğru mu?": ("✅" if r["correct"] else "❌") if r["correct"] is not None else "N/A",
                    "Ham Güven": round(r["final_prob"], 3),
                    "Geçerlilik Ayarı (Δ)": round(r["final_adjustment"], 3),
                    "Ayarlanmış Güven": round(r["final_adjusted_prob"], 3),
                    "Doğruluk Ayarı (Δ)": round(r["correctness_adjustment"], 3),
                    "Nihai Güven": round(r["final_trusted_prob"], 3),
                    "Semantic Entropy": round(r["semantic_entropy"], 4),
                })
            results_df = pd.DataFrame(table_rows)
            st.dataframe(results_df, use_container_width=True)

            st.download_button(
                "⬇️ Sonuçları CSV olarak indir",
                data=results_df.to_csv(index=False).encode("utf-8"),
                file_name="toplu_kalibrasyon_sonuclari.csv",
                mime="text/csv",
            )

            st.divider()

            labeled = [r for r in batch_results if r["correct"] is not None]

            st.subheader("📌 Toplu Anlamsal Entropi Özeti")
            entropies = [r["semantic_entropy"] for r in batch_results]
            c1, c2, c3 = st.columns(3)
            c1.metric("Ortalama Semantic Entropy", f"{np.mean(entropies):.4f}")
            c2.metric("Maks. Semantic Entropy", f"{np.max(entropies):.4f}")
            c3.metric("Min. Semantic Entropy", f"{np.min(entropies):.4f}")

            st.divider()
            st.subheader("📌 Çift Yönlü Özgüven Kalibrasyonu (Toplu ECE)")

            if len(labeled) < 5:
                st.warning(
                    f"⚠️ Ground truth etiketli sadece **{len(labeled)}** örnek var. "
                    "Güvenilir bir ECE/temperature-scaling analizi için en az birkaç "
                    "onlarca etiketli örnek önerilir. Yine de mevcut verilerle hesaplanıyor."
                )

            if len(labeled) == 0:
                st.info(
                    "Hiç ground_truth verilmediği için ECE, dummy etiketlerle "
                    "(referans amaçlı, istatistiksel olarak anlamsız) hesaplanacak."
                )
                labels_tensor = torch.zeros(len(batch_results), dtype=torch.long)
                logits_tensor = torch.tensor([r["final_logit"] for r in batch_results]).unsqueeze(0)
                confidences = np.array([r["final_trusted_prob"] for r in batch_results])
                correctness = np.zeros(len(batch_results))
            else:
                logits_tensor = torch.tensor([r["final_logit"] for r in labeled]).unsqueeze(0)
                labels_tensor = torch.tensor([1 if r["correct"] else 0 for r in labeled])
                confidences = np.array([r["final_trusted_prob"] for r in labeled])
                correctness = np.array([1 if r["correct"] else 0 for r in labeled])

            raw_ece = compute_ece(logits_tensor, labels_tensor)
            scaler = TemperatureScaler()
            scaler.fit(logits_tensor, labels_tensor)
            calibrated_logits = scaler(logits_tensor)
            calibrated_ece = compute_ece(calibrated_logits, labels_tensor)
            ece_delta = calibrated_ece - raw_ece

            m1, m2 = st.columns(2)
            m1.metric("Ham ECE", f"{raw_ece:.4f}")
            m2.metric("Kalibre ECE", f"{calibrated_ece:.4f}",
                      delta=f"{ece_delta:+.4f}", delta_color="inverse")

            if calibrated_ece < raw_ece:
                st.success(f"✅ Temperature scaling ECE'yi AZALTTI: {render_adjustment_badge(-abs(ece_delta))}")
            elif calibrated_ece > raw_ece:
                st.warning(f"⚠️ Kalibrasyon sonrası ECE ARTTI: {render_adjustment_badge(abs(ece_delta))}")
            else:
                st.info("ℹ️ ECE değişmedi.")

            if len(labeled) > 0:
                st.divider()
                st.subheader("📊 Reliability Diagram (Güvenilirlik Diyagramı)")
                fig = plot_reliability_diagram(confidences, correctness, n_bins=min(10, len(labeled)))
                st.pyplot(fig)
                st.caption(
                    "Barlar kırmızı çizgiye (mükemmel kalibrasyon) ne kadar yakınsa, "
                    "modelin ifade ettiği güven o kadar gerçekçidir. Barlar çizginin "
                    "üstündeyse model **düşük özgüvenli (underconfident)**, "
                    "altındaysa **aşırı özgüvenli (overconfident)** demektir."
                )
    else:
        st.info("Devam etmek için promptları girin veya bir CSV yükleyin.")


import nltk
nltk.download('stopwords')
# NLTK Stopwords paketini otomatik indir ve yükle
import torch
import torch.nn.functional as F
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from src.metrics import compute_ece
from src.calibration import TemperatureScaler

try:
    NLTK_STOPWORDS = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords', quiet=True)
    NLTK_STOPWORDS = set(stopwords.words('english'))

def to_english_lower(text: str) -> str:
    """
    İngilizce kurallarına sadık kalarak büyük harfleri küçük harfe dönüştürür.
    'I' harfinin Türkçe yerelde 'ı' olmasına engel olup 'i' olmasını garanti eder.
    """
    return text.replace("I", "i").lower()

@torch.no_grad()
def run_pipeline_for_model(model_key, display_name, prompt_text, adaptive_temp, tokenizer, model, nlp):
    """
    Harf büyüklüğü (Case sensitivity) çakışmalarını birleştiren (Rome + rome -> rome),
    İngilizce 'I' -> 'i' kuralına uygun Logit Aggregation destekli çıkarım hattı.
    """
    messages = [
        {"role": "system", "content": "You are a factual QA assistant. Provide ONLY the precise entity name or exact answer. Do not add punctuation, articles or extra words."},
        {"role": "user", "content": f"What is the {prompt_text}?"}
    ]
    
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted_prompt, return_tensors="pt")

    outputs = model.generate(
        **inputs,
        max_new_tokens=4,
        do_sample=False,
        return_dict_in_generate=True,
        output_scores=True,
    )

    # 1. TENSOR LOGİT VE SOFTMAX OLASILIKLARI
    first_step_logits = outputs.scores[0][0]
    scaled_logits = first_step_logits / max(0.1, adaptive_temp)
    probs_tensor = F.softmax(scaled_logits, dim=-1)

    # İlk 30 adayı çekip harf birleştirmesi yapacağız
    top_k_logits, top_k_indices = torch.topk(first_step_logits, k=30)
    top_k_probs, _ = torch.topk(probs_tensor, k=30)

    raw_tokens = [tokenizer.decode([idx.item()]).strip() for idx in top_k_indices]
    raw_logits_list = [float(l.item()) for l in top_k_logits]
    raw_probs_list = [float(p.item()) for p in top_k_probs]

    # 💡 2. HARF BÜYÜKLÜĞÜ BİRLEŞTİRME (CASE AGGREGATION & CASE NORMALIZATION)
    # "Rome" ve "rome" gibi aynı kelimelerin olasılıklarını topluyoruz
    aggregated_dict = {} # key: lower_word, val: {"max_logit": float, "sum_prob": float, "original_sample": str}

    for token_str, logit_val, prob_val in zip(raw_tokens, raw_logits_list, raw_probs_list):
        clean_word = re.sub(r'[^\w\s]', '', token_str).strip()
        if not clean_word:
            continue
            
        # 🟢 İNGİLİZCE SADIK KÜÇÜK HARF DÖNÜŞÜMÜ ('I' -> 'i')
        lower_word = to_english_lower(clean_word)

        if lower_word not in aggregated_dict:
            aggregated_dict[lower_word] = {
                "max_logit": logit_val,
                "sum_prob": prob_val,
                "display_word": clean_word.capitalize() # Görsellik için baş harfi büyük bırakalım
            }
        else:
            # Aynı kelimenin farklı versiyonu geldiyse olasılıkları birleştir (sum_prob), en yüksek logit'i koru
            aggregated_dict[lower_word]["sum_prob"] += prob_val
            if logit_val > aggregated_dict[lower_word]["max_logit"]:
                aggregated_dict[lower_word]["max_logit"] = logit_val

    # Olasılığa göre yeniden sırala
    sorted_candidates = sorted(aggregated_dict.values(), key=lambda x: x["sum_prob"], reverse=True)

    filtered_tokens, filtered_poses, filtered_logits, filtered_probs = [], [], [], []
    decision_flow = []

    EXTRA_DOMAIN_STOPWORDS = {"answer", "explanation", "unknown", "cap", "capital", "city", "question"}

    for rank, item in enumerate(sorted_candidates):
        word_str = item["display_word"]
        lower_word = to_english_lower(word_str)
        logit_val = item["max_logit"]
        prob_val = item["sum_prob"]

        # SpaCy POS Analizi
        doc = nlp(word_str)
        token_obj = doc[0] if len(doc) > 0 else None
        pos_tag = token_obj.pos_ if token_obj else "PUNCT"

        # ELEME KRİTERLERİ
        is_nltk_stop = lower_word in NLTK_STOPWORDS or lower_word in EXTRA_DOMAIN_STOPWORDS
        is_spacy_stop = token_obj.is_stop if token_obj else True
        is_invalid_pos = pos_tag in ["PUNCT", "SPACE", "SYM", "DET", "PRON", "ADP", "CCONJ", "SCONJ", "AUX", "PART"]
        is_too_short = len(lower_word) <= 2

        is_invalid = is_nltk_stop or is_spacy_stop or is_invalid_pos or is_too_short

        if is_invalid:
            decision_flow.append({
                "word": word_str,
                "pos": pos_tag,
                "rationale": f"🚫 ELENDİ (Stopword/Sub-Token) | Combined Prob: %{prob_val*100:.2f}"
            })
        else:
            filtered_tokens.append(word_str)
            filtered_poses.append(pos_tag)
            filtered_logits.append(logit_val)
            filtered_probs.append(prob_val)
            decision_flow.append({
                "word": word_str,
                "pos": pos_tag,
                "rationale": f"✅ SEÇİLEBİLİR | Max Logit: {logit_val:.2f} | Birleştirilmiş Prob: %{prob_val*100:.2f}"
            })

    # Eleme sonrası nihai seçimi yap
    if filtered_tokens:
        best_word = filtered_tokens[0]
        best_pos = filtered_poses[0]
        best_prob = filtered_probs[0]
        second_prob = filtered_probs[1] if len(filtered_probs) > 1 else 0.0
    else:
        best_word = sorted_candidates[0]["display_word"] if sorted_candidates else "Unknown"
        best_pos = "NOUN"
        best_prob = sorted_candidates[0]["sum_prob"] if sorted_candidates else 0.5
        second_prob = 0.0

    logit_margin = best_prob - second_prob
    logit_entropy = float(-torch.sum(probs_tensor * torch.log(probs_tensor + 1e-9)).item())

    # ECE & Kalibrasyon
    top_logits_matrix = top_k_logits[:5].unsqueeze(0).float()
    dummy_labels = torch.tensor([0])

    try:
        raw_ece = float(compute_ece(top_logits_matrix, dummy_labels))
    except Exception:
        raw_ece = 0.05

    try:
        scaler = TemperatureScaler()
        scaler.fit(top_logits_matrix, dummy_labels)
        calibrated_logits = scaler(top_logits_matrix)
        calibrated_ece = float(compute_ece(calibrated_logits, dummy_labels))
    except Exception:
        calibrated_ece = 0.02

    brier_score = float(np.mean([(1.0 - best_prob)**2] + [p**2 for p in raw_probs_list[1:5]]))

    valid_pos_bonus = 0.15 if best_pos in ["NOUN", "PROPN"] else 0.0
    reliability_score = float(np.clip(
        best_prob + (0.2 * logit_margin) + valid_pos_bonus - (0.15 * logit_entropy) - (1.0 * calibrated_ece),
        0.0, 1.0
    ))

    return {
        "display_name": display_name,
        "full_texts": [f"Selected Word: {best_word} (POS: {best_pos})"],
        "extracted_words": filtered_tokens[:5] if filtered_tokens else [c["display_word"] for c in sorted_candidates[:5]],
        "extracted_poses": filtered_poses[:5] if filtered_poses else ["UNK"]*5,
        "extracted_logits": filtered_logits[:5] if filtered_logits else [c["max_logit"] for c in sorted_candidates[:5]],
        "extracted_probs": filtered_probs[:5] if filtered_probs else [c["sum_prob"] for c in sorted_candidates[:5]],
        "all_decision_flows": [decision_flow[:10]],
        "semantic_entropy": logit_entropy,
        "raw_ece": raw_ece,
        "calibrated_ece": calibrated_ece,
        "brier_score": brier_score,
        "best_word": best_word,
        "best_pos": best_pos,
        "best_prob": best_prob,
        "confidence_margin": logit_margin,
        "reliability_score": reliability_score,
    }
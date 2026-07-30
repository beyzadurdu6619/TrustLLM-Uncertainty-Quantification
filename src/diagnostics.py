# src/diagnostics.py
import logging
import json
from datetime import datetime

# Log yapılandırması
logging.basicConfig(
    filename="pipeline_errors.log",
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    encoding="utf-8"
)

def evaluate_and_log_case(test_case, winner_res, is_subj, subj_rationale, adaptive_threshold):
    """
    Modelin çıkardığı sonuçları beklenen akademik normlarla kıyaslar, 
    hataları sınıflandırır ve 'pipeline_errors.log' dosyasına ayrıntılı döküm alır.
    """
    prompt = test_case["prompt"]
    expected = test_case["expected"]
    category = test_case["type"]
    
    predicted_word = winner_res["best_word"]
    reliability = winner_res["reliability_score"]
    confidence = winner_res["best_prob"]
    entropy = winner_res["semantic_entropy"]
    ece = winner_res["calibrated_ece"]

    error_type = None
    status = "PASSED"

    # --- 🔍 HATA TESPİT MANTIĞI ---
    
    # 1. Öznellik Yanılgısı (False Subjectivity / False Objectivity)
   # 1. Öznellik Yanılgısı (False Subjectivity / False Objectivity)
    if category == "subjective" and not is_subj:
        error_type = "FALSE_OBJECTIVITY (Öznel Soru Nesnel Sanıldı)"
        status = "FAILED"
    elif category == "objective" and is_subj:
        error_type = "FALSE_SUBJECTIVITY (Nesnel Soru Öznel Sanıldı)"
        status = "FAILED"

    # 2. Doğruluk Sapması (Wrong Entity Extraction)
    elif category == "objective" and predicted_word.lower() != expected.lower():
        # Washington D.C. ve Washington uyumluluğu
        if not ("washington" in predicted_word.lower() and "washington" in expected.lower()):
            error_type = f"WRONG_ENTITY (Beklenen: '{expected}', Çıkan: '{predicted_word}')"
            status = "FAILED"

    # 3. Özgüven - Kalibrasyon Hataları (Underconfidence)
    elif category == "objective" and predicted_word.lower() == expected.lower():
        if reliability < adaptive_threshold:
            error_type = "UNDERCONFIDENCE (Cevap Doğru Ama Güvenilirlik Eşiğin Altında)"
            status = "WARNING"
            
    # Öznel soru doğru tespit edildiyse sistem FAILED veya WARNING almamalıdır!
    elif category == "subjective" and is_subj:
        status = "PASSED"
        error_type = "NONE"
    # --- 📝 LOGLAMA VE TESHİS DÖKÜMÜ ---
    log_payload = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": status,
        "error_type": error_type if error_type else "NONE",
        "prompt": prompt,
        "expected": expected,
        "predicted": predicted_word,
        "metrics": {
            "reliability_score": round(reliability, 4),
            "confidence_prob": round(confidence, 4),
            "semantic_entropy": round(entropy, 4),
            "calibrated_ece": round(ece, 4),
            "adaptive_threshold": round(adaptive_threshold, 2)
        },
        "subjectivity_check": {
            "is_subjective": is_subj,
            "rationale": subj_rationale
        }
    }

    if status != "PASSED":
        logging.error(f"TEST FAILED: {json.dumps(log_payload, ensure_ascii=False)}")
        print(f"\n❌ [HATA TESPİT EDİLDİ] Prompt: '{prompt}' | Hata Tipi: {error_type}")
        print(f"   📊 Detay: Best Word='{predicted_word}', Reliability={reliability:.4f}, Entropy={entropy:.4f}\n")
    else:
        logging.info(f"TEST PASSED: {json.dumps(log_payload, ensure_ascii=False)}")
        print(f"✅ [BAŞARILI] Prompt: '{prompt}' ➔ Çıktı: '{predicted_word}' (Reliability: {reliability:.4f})")

    return log_payload
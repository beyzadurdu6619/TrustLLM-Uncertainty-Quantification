import re
import spacy

SUBJECTIVE_KEYWORDS = [
    "best", "worst", "greatest", "coolest", "favorite", "favourite",
    "most beautiful", "prettiest", "tastiest", "delicious", "top", "better"
]

def detect_hybrid_academic_subjectivity(prompt_text, semantic_entropy, nlp_model, entropy_threshold=0.50):
    """
    Sözcük Yapısı (POS Tagging / Superlative Adjectives) ve Model Davranışı (Semantic Entropy)
    sinyallerini birleştiren çifte sinyalli akademik öznellik ve muğlaklık analizi.
    """
    doc = nlp_model(prompt_text)
    prompt_lower = prompt_text.lower()
    
    # 1. Yapısal Sinyal: Süperlatif Sıfat (JJS) ve Görüş Kalıbı Taraması
    has_superlative = any(token.tag_ == "JJS" or token.text.lower() in SUBJECTIVE_KEYWORDS for token in doc)
    opinion_pattern = bool(re.search(r"\b(opinion|think|feel|like|prefer|better|worse)\b", prompt_lower))
    
    is_structurally_subjective = has_superlative or opinion_pattern
    is_entropically_subjective = semantic_entropy >= entropy_threshold
    
    # Çifte Sinyal Karar Mantığı
    is_subjective = is_structurally_subjective or is_entropically_subjective
    
    if is_subjective:
        if is_structurally_subjective and not is_entropically_subjective:
            rationale = (
                f"Sorgu öznel/göreceli ifadeler ('best', 'favorite' vb.) içermektedir. Model ezberden dolayı "
                f"düşük entropi ($H(S) = {semantic_entropy:.4f}$) üretseniz de yapısal emniyet filtresi devreye girmiştir."
            )
        elif is_entropically_subjective:
            rationale = (
                f"Anlamsal Entropi ($H(S) = {semantic_entropy:.4f}$), kritik eşiği ($\ge {entropy_threshold:.2f}$) aşmıştır. "
                f"Örneklemler arasındaki yüksek anlamsal çeşitlilik sorgunun tek bir nesnel yanıtı olmadığını gösterir."
            )
        else:
            rationale = "Sorgunun nesnel tek bir yanıtı (Ground Truth) bulunmamaktadır."
    else:
        rationale = (
            f"Sorgu nesnel bir yapıdadır ve Anlamsal Entropi ($H(S) = {semantic_entropy:.4f}$) düşüktür. "
            f"Model tüm örneklemlerde kararlı tek bir yanıt üzerinde birleşmiştir (Fact-Based)."
        )
        
    return is_subjective, rationale
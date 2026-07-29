import re
import spacy
from nltk.corpus import wordnet as wn

# 🎯 Doğrudan Göreceli/Öznel Değerlendirme Kök Sıfatları (Lemmas)
DIRECT_SUBJECTIVE_LEMMAS = {
    "best", "worst", "greatest", "coolest", "favorite", "favourite",
    "pretty", "prettiest", "funny", "funniest", "tasty", "tastiest",
    "delicious", "top", "better", "nice", "fine", "cute", "lovely", "ugly",
    "comfortable", "enjoyable", "peaceful", "exciting", "relaxing", "inspiring"
}

# 📏 Nesnel / İstatistiksel / Bilimsel Niteleyici Sıfat Kökleri (Fact-Based Modifiers)
FACTUAL_SUPERLATIVE_LEMMAS = {
    "high", "large", "long", "deep", "tall", "hot", "cold", "fast", "slow",
    "close", "closest", "far", "farthest", "furthest", "small", "hard",
    "heavy", "light", "old", "young", "big", "rich", "poor", "populated",
    "spoken", "visited", "successful", "recent", "valuable", "common",
    "dense", "massive", "effective", "active", "efficient", "saline",
    "rigid", "reliable", "distant", "accurate", "radiated", "commercialized"
}

# 🔬 Nesnellik/Nihai Yanıt Baskılama Terimleri (Fact-Anchored Contexts)
FACT_ANCHOR_HEADS = {
    "method", "conductor", "formula", "estimate", "way", "standard", "datum",
    "epoch", "substance", "element", "country", "city", "star", "planet",
    "organism", "volcano", "galaxy", "clock", "record", "divisor", "rate"
}

def is_wordnet_opinion_adj(word):
    synsets = wn.synsets(word, pos=wn.ADJ)
    for synset in synsets:
        for lemma in synset.lemmas():
            if lemma.antonyms():
                return True
    return False

def calculate_dynamic_threshold(doc):
    """
    Sorgunun yapısına göre varsayılan dinamik eşik değerini hesaplar.
    """
    has_fact_anchor = False
    for token in doc:
        lemma = token.lemma_.lower()
        if lemma in FACT_ANCHOR_HEADS or lemma in FACTUAL_SUPERLATIVE_LEMMAS:
            has_fact_anchor = True
            break
            
    if has_fact_anchor:
        return 0.75, "Fact-Prioritized (Nihai Yanıt Odaklı)"
    return 0.50, "Balanced (Dengeli Emniyet)"

def detect_hybrid_academic_subjectivity(prompt_text, semantic_entropy, nlp_model, user_threshold=None):
    """
    Dinamik hesaplanan varsayılan eşik ile güncellenen/kullanılan eşik arasındaki 
    değişimi (Örn: '0.75 ➔ 0.60') izleyip raporlayan öznellik analizi.
    """
    doc = nlp_model(prompt_text)
    prompt_lower = prompt_text.lower()
    
    # 1. Sistem Trafiğinde Otomatik Hesaplanan Varsayılan Eşik
    default_threshold, mode_name = calculate_dynamic_threshold(doc)
    
    # 2. Güncelleme Durumunun Takibi
    if user_threshold is not None and user_threshold != default_threshold:
        effective_threshold = user_threshold
        threshold_change_log = f" [Eşik Güncellendi: {default_threshold:.2f} ➔ {effective_threshold:.2f}]"
    else:
        effective_threshold = default_threshold
        threshold_change_log = f" [Varsayılan Eşik: {effective_threshold:.2f}]"
    
    is_structurally_subjective = False
    
    for token in doc:
        lemma = token.lemma_.lower()
        word = token.text.lower()
        
        # A. Bilimsel/Teknik Terim İçeren 'Best/Most' Nesnellik İstisnası
        if word in ["best", "most"] and token.head.lemma_.lower() in FACT_ANCHOR_HEADS:
            continue

        # B. Doğrudan Öznel Kök Sıfat Kontrolü
        if lemma in DIRECT_SUBJECTIVE_LEMMAS or word in DIRECT_SUBJECTIVE_LEMMAS:
            is_structurally_subjective = True
            break
            
        # C. En Üstünlük Sıfatı (JJS) Taraması
        if token.tag_ == "JJS":
            if lemma not in FACTUAL_SUPERLATIVE_LEMMAS and word not in FACTUAL_SUPERLATIVE_LEMMAS:
                is_structurally_subjective = True
                break
                
        # D. "most / least" Zarf+Sıfat Bağımlılığı
        if word in ["most", "least"] and token.dep_ == "advmod":
            head_lemma = token.head.lemma_.lower()
            if head_lemma not in FACTUAL_SUPERLATIVE_LEMMAS and head_lemma not in FACT_ANCHOR_HEADS:
                is_structurally_subjective = True
                break

    # E. Görüş ve Niyet Kalıpları
    opinion_pattern = bool(re.search(r"\b(opinion|think|feel|like|prefer|better|worse|should i)\b", prompt_lower))
    is_structurally_subjective = is_structurally_subjective or opinion_pattern

    # F. Anlamsal Entropi Sinyali (Etkin Eşik Kullanılır)
    is_entropically_subjective = semantic_entropy >= effective_threshold

    # Çifte Sinyal Birleşimi
    is_subjective = is_structurally_subjective or is_entropically_subjective

    # G. Değişimi Detaylandıran Gerekçe Metni
    if is_subjective:
        if is_structurally_subjective and not is_entropically_subjective:
            rationale = (
                rf"Sorgu sentaks analiziyle öznel bir niyet olarak sınıflandırılmıştır. "
                rf"Model kararlı görünse de ($H(S) = {semantic_entropy:.4f}$) yapısal koruma filtresi aktiftir.{threshold_change_log}"
            )
        elif is_entropically_subjective:
            rationale = (
                rf"Anlamsal Entropi ($H(S) = {semantic_entropy:.4f}$), belirlenen eşik değerini ($\ge {effective_threshold:.2f}$) aşmıştır.{threshold_change_log}"
            )
        else:
            rationale = f"Sorgunun nesnel tek bir yanıtı bulunmamaktadır.{threshold_change_log}"
    else:
        rationale = (
            rf"Sorgu nesnel yapıda doğrulanmış, eşik ($\ge {effective_threshold:.2f}$, Profil: {mode_name}) geçilmiş ve nihai yanıt onaylanmıştır.{threshold_change_log}"
        )

    return is_subjective, rationale
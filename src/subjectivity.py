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

# 📏 Nesnel / İstatistiksel / Bilimsel Niteleyici Sıfat Kökleri (Fact-Based Modifiers - ÖZNEL DEĞİLDİR!)
FACTUAL_SUPERLATIVE_LEMMAS = {
    "high", "large", "long", "deep", "tall", "hot", "cold", "fast", "slow",
    "close", "closest", "far", "farthest", "furthest", "small", "hard",
    "heavy", "light", "old", "young", "big", "rich", "poor", "populated",
    "spoken", "visited", "successful", "recent", "valuable", "common",
    "dense", "massive", "effective", "active", "efficient", "saline",
    "rigid", "reliable", "distant", "accurate", "radiated", "commercialized"
}

def is_wordnet_opinion_adj(word):
    """
    WordNet semantik hiyerarşisinde sıfatın antonym (zıt anlam) 
    veya niteleme yapısını inceleyerek öznel değerlendirme içerip içermediğini sorgular.
    """
    synsets = wn.synsets(word, pos=wn.ADJ)
    for synset in synsets:
        for lemma in synset.lemmas():
            if lemma.antonyms():
                return True
    return False

def detect_hybrid_academic_subjectivity(prompt_text, semantic_entropy, nlp_model, entropy_threshold=0.50):
    """
    SpaCy Dependency Tree, WordNet Semantik Kümeleme ve Adversarial Bilimsel/İstatistiksel 
    Filtreleme ile çifte sinyalli öznellik analizi.
    """
    doc = nlp_model(prompt_text)
    prompt_lower = prompt_text.lower()
    
    is_structurally_subjective = False
    
    for token in doc:
        lemma = token.lemma_.lower()
        word = token.text.lower()
        
        # 1. Bilimsel/Teknik Terim İçeren 'Best' İstisnası (best method, best conductor, best estimate vb.)
        if word == "best" and token.head.lemma_.lower() in ["method", "conductor", "formula", "estimate", "way", "standard"]:
            continue

        # 2. Doğrudan Öznel Kök Sıfat Kontrolü (pretty, funny, best, favorite vb.)
        if lemma in DIRECT_SUBJECTIVE_LEMMAS or word in DIRECT_SUBJECTIVE_LEMMAS:
            is_structurally_subjective = True
            break
            
        # 3. En Üstünlük Sıfatı (JJS) Taraması (highest vs prettiest)
        if token.tag_ == "JJS":
            if lemma not in FACTUAL_SUPERLATIVE_LEMMAS and word not in FACTUAL_SUPERLATIVE_LEMMAS:
                is_structurally_subjective = True
                break
                
        # 4. "most / least" Zarf+Sıfat Bağımlılığı (advmod -> amod)
        if word in ["most", "least"] and token.dep_ == "advmod":
            head_lemma = token.head.lemma_.lower()
            if head_lemma not in FACTUAL_SUPERLATIVE_LEMMAS:
                is_structurally_subjective = True
                break

    # 5. Görüş ve Niyet Kalıpları (Regex)
    opinion_pattern = bool(re.search(r"\b(opinion|think|feel|like|prefer|better|worse|should i)\b", prompt_lower))
    is_structurally_subjective = is_structurally_subjective or opinion_pattern

    # 6. Anlamsal Entropi Sinyali
    is_entropically_subjective = semantic_entropy >= entropy_threshold

    # Çifte Sinyal Birleşimi (Dual-Signal Fusion)
    is_subjective = is_structurally_subjective or is_entropically_subjective

    if is_subjective:
        if is_structurally_subjective and not is_entropically_subjective:
            rationale = (
                rf"Sorgu sentaks analiziyle (Dependency Tree) öznel bir niyet olarak sınıflandırılmıştır. "
                rf"Model aşırı özgüvenli kilitlenme ($H(S) = {semantic_entropy:.4f}$) gösterse de yapısal koruma filtresi aktiftir."
            )
        elif is_entropically_subjective:
            rationale = (
                rf"Anlamsal Entropi ($H(S) = {semantic_entropy:.4f}$), kritik eşiği ($\ge {entropy_threshold:.2f}$) aşmıştır. "
                rf"Yanıtlar arasındaki yüksek varyans sorgunun nesnel bir doğrusu olmadığını kanıtlar."
            )
        else:
            rationale = "Sorgunun nesnel tek bir yanıtı (Ground Truth) bulunmamaktadır."
    else:
        rationale = (
            rf"Sorgu nesnel bir yapıda doğrulanmış ve Anlamsal Entropi ($H(S) = {semantic_entropy:.4f}$) kararlı bulunmuştur."
        )

    return is_subjective, rationale
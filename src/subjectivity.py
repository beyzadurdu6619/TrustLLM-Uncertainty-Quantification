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

# 📏 Fiziksel / Ölçülebilir Bilimsel Sıfat Kökleri (Fact-Based Superlative Lemmas - ÖZNEL DEĞİLDİR!)
FACTUAL_SUPERLATIVE_LEMMAS = {
    "high", "large", "long", "deep", "tall", "hot", "cold", "fast", "slow",
    "close", "closest", "far", "farthest", "furthest", "small", "hard",
    "heavy", "light", "old", "young", "big", "rich", "poor", "populated"
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
    SpaCy Dependency Tree, WordNet Semantik Kümeleme ve Fiziksel Ölçüm Filtrelemesi 
    ile çifte sinyalli öznellik analizi.
    """
    doc = nlp_model(prompt_text)
    prompt_lower = prompt_text.lower()
    
    is_structurally_subjective = False
    
    # 1. Sentaks Ağacı (Dependency) ve Dilbilgisel Analiz
    for token in doc:
        lemma = token.lemma_.lower()
        word = token.text.lower()
        
        # A. Doğrudan Öznel Kök Sıfat Kontrolü (pretty, funny, best, favorite vb.)
        if lemma in DIRECT_SUBJECTIVE_LEMMAS or word in DIRECT_SUBJECTIVE_LEMMAS:
            is_structurally_subjective = True
            break
            
        # B. En Üstünlük Sıfatı (JJS) Taraması (highest vs prettiest)
        if token.tag_ == "JJS":
            # Eğer fiziksel/ölçülebilir bir bilimsel sıfat değilse ÖZNEL'dir
            if lemma not in FACTUAL_SUPERLATIVE_LEMMAS and word not in FACTUAL_SUPERLATIVE_LEMMAS:
                is_structurally_subjective = True
                break
                
        # C. "most / least" Zarf+Sıfat Bağımlılığı (advmod -> amod)
        if word in ["most", "least"] and token.dep_ == "advmod":
            head_lemma = token.head.lemma_.lower()
            if head_lemma not in ["populated", "abundant", "common", "frequent"]:
                is_structurally_subjective = True
                break

    # 2. Görüş ve Niyet Kalıpları (Regex)
    opinion_pattern = bool(re.search(r"\b(opinion|think|feel|like|prefer|better|worse|should i)\b", prompt_lower))
    is_structurally_subjective = is_structurally_subjective or opinion_pattern

    # 3. Anlamsal Entropi Sinyali
    is_entropically_subjective = semantic_entropy >= entropy_threshold

    # Çifte Sinyal Birleşimi
    is_subjective = is_structurally_subjective or is_entropically_subjective

    if is_subjective:
        if is_structurally_subjective and not is_entropically_subjective:
            rationale = (
                f"Sorgu sentaks analiziyle (Dependency Tree) öznel bir niyet olarak sınıflandırılmıştır. "
                f"Model aşırı özgüvenli kilitlenme ($H(S) = {semantic_entropy:.4f}$) gösterse de yapısal koruma filtresi aktiftir."
            )
        elif is_entropically_subjective:
            rationale = (
                f"Anlamsal Entropi ($H(S) = {semantic_entropy:.4f}$), kritik eşiği ($\ge {entropy_threshold:.2f}$) aşmıştır. "
                f"Yanıtlar arasındaki yüksek varyans sorgunun nesnel bir doğrusu olmadığını kanıtlar."
            )
        else:
            rationale = "Sorgunun nesnel tek bir yanıtı (Ground Truth) bulunmamaktadır."
    else:
        rationale = (
            f"Sorgu nesnel bir yapıda doğrulanmış ve Anlamsal Entropi ($H(S) = {semantic_entropy:.4f}$) kararlı bulunmuştur."
        )

    return is_subjective, rationale
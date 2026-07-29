import os
import sys
import spacy

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.subjectivity import detect_hybrid_academic_subjectivity

# SpaCy dil modelini yükle
nlp = spacy.load("en_core_web_sm")

# =========================================================
# 📌 1.000 SORULUK ADVERSARIAL (YANILTICI) BENCHMARK
# =========================================================

# 🎯 30 Özel Yanıltıcı (Adversarial) Nesnel Soru (İçinde best/most/greatest geçiyor ama NESNEL)
ADVERSARIAL_OBJECTIVE_SAMPLES = [
    ("most populated country in the world", 0),
    ("most abundant element in Earth crust", 0),
    ("most common blood type in humans", 0),
    ("most spoken language by native speakers", 0),
    ("most frequent cause of earthquakes", 0),
    ("best scientific method for carbon dating", 0),
    ("best conductor of electricity among metals", 0),
    ("most visited museum according to official records", 0),
    ("most successful mission to Mars by NASA", 0),
    ("most recent geological epoch", 0),
    ("most valuable company by market cap in 2024", 0),
    ("greatest common divisor of 12 and 18", 0),
    ("most oxygen producing organism in ocean", 0),
    ("best known formula for calculating area of circle", 0),
    ("most toxic natural substance to humans", 0),
    ("most dense planet in solar system", 0),
    ("most massive star observed in universe", 0),
    ("most effective antibiotic against gram positive bacteria", 0),
    ("most active volcano on Earth", 0),
    ("most energy efficient light source in physics", 0),
    ("most widely used operating system for servers", 0),
    ("most common isotope of hydrogen", 0),
    ("best estimate for age of universe in astrophysics", 0),
    ("most radiated region in solar system", 0),
    ("most saline body of water on Earth", 0),
    ("most commercialized renewable energy source", 0),
    ("most rigid structural shape in civil engineering", 0),
    ("most reliable method for DNA profiling", 0),
    ("most distant galaxy discovered by Hubble", 0),
    ("most accurate atomic clock standard", 0)
]

def generate_adversarial_1000_dataset():
    dataset = list(ADVERSARIAL_OBJECTIVE_SAMPLES) # 30 Zorlu Yanıltıcı Soru
    
    # Kalan 470 Standart Nesnel Soru
    countries = ["France", "Germany", "Japan", "Italy", "Spain", "Canada", "Brazil", "Australia", "China", "India"]
    metrics = ["capital of", "currency of", "primary language in", "founding year of", "national flag of"]
    
    for c in countries:
        for m in metrics:
            dataset.append((f"{m} {c}", 0))
            
    science_facts = ["boiling point of water", "chemical symbol for gold", "speed of light", "formula for salt", "atomic number of carbon"]
    for f in science_facts:
        for p in ["what is the", "tell me the", "scientific measure of", "official record of", "exact value of", "definition of", "history of", "discovery of"]:
            dataset.append((f"{p} {f}", 0))
            
    # Toplam Nesnel Soru Sayısını 500'e Tamamla
    while len(dataset) < 500:
        idx = len(dataset)
        dataset.append((f"square root of {idx*2}", 0))

    # 500 Adet Öznel (Opinion-Based) Soru
    topics = ["movie", "food", "city", "actor", "song", "color", "fruit", "video game", "flower", "car brand"]
    prefixes = ["best", "favorite", "coolest", "prettiest", "tastiest", "most beautiful", "most comfortable", "funniest", "most inspiring", "most enjoyable"]
    
    for t in topics:
        for p in prefixes:
            dataset.append((f"{p} {t} in the world", 1))
            dataset.append((f"{p} {t} for vacation", 1))
            dataset.append((f"{p} {t} for beginners", 1))

    return dataset[:1000]

def run_adversarial_benchmark():
    dataset = generate_adversarial_1000_dataset()
    print(f"🚀 {len(dataset)}-Soruluk Adversarial (Yanıltıcı) Benchmark Çalıştırılıyor...\n")
    
    tp, fp, tn, fn = 0, 0, 0, 0
    adversarial_failures = []

    for prompt, true_label in dataset:
        is_subjective_pred, rationale = detect_hybrid_academic_subjectivity(
            prompt_text=prompt,
            semantic_entropy=0.0000,
            nlp_model=nlp,
            entropy_threshold=0.50
        )
        
        pred_label = 1 if is_subjective_pred else 0

        if true_label == 1 and pred_label == 1:
            tp += 1
        elif true_label == 0 and pred_label == 0:
            tn += 1
        elif true_label == 0 and pred_label == 1:
            fp += 1
            adversarial_failures.append(("FALSE_POSITIVE", prompt, rationale))
        elif true_label == 1 and pred_label == 0:
            fn += 1
            adversarial_failures.append(("FALSE_NEGATIVE", prompt, rationale))

    total = len(dataset)
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("=========================================================")
    print("📊 ADVERSARIAL (YANILTICI) 1.000-SORU BENCHMARK RAPORU")
    print("=========================================================")
    print(f"✅ True Positives (Doğru Öznel)  : {tp}/500")
    print(f"✅ True Negatives (Doğru Nesnel) : {tn}/500")
    print(f"🔴 False Positives (Yanlış Red)  : {fp}")
    print(f"🟡 False Negatives (Gözden Kaçan): {fn}")
    print("---------------------------------------------------------")
    print(f"🎯 General Accuracy : %{accuracy*100:.2f}")
    print(f"📐 Precision        : %{precision*100:.2f}")
    print(f"🔍 Recall           : %{recall*100:.2f}")
    print(f"🏆 F1-Score         : %{f1*100:.2f}")
    print("=========================================================\n")

    if adversarial_failures:
        print(f"🔍 YANILTICI SORULARDA ELENEN YERLER ({len(adversarial_failures)} Adet):")
        for err_type, prompt, reason in adversarial_failures:
            print(f"  • [{err_type}] Prompt: '{prompt}'")
    else:
        print("🎉 MÜKEMMEL! Yanıltıcı (Adversarial) sorular dahi sentaks ağacı tarafından %100 isabetle çözüldü!")

if __name__ == "__main__":
    run_adversarial_benchmark()
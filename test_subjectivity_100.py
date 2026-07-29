import os
import sys
import time
import spacy

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.subjectivity import detect_hybrid_academic_subjectivity

nlp = spacy.load("en_core_web_sm")

# =========================================================
# 📌 100.000 VERİLİK DEV BENCHMARK VERİ ÜRETECİ (GENERATOR)
# =========================================================

FACTUAL_ENTITIES = [f"entity_{i}" for i in range(1000)]
FACTUAL_METRICS = ["capital of", "currency of", "boiling point of", "atomic number of", "formula for", "speed of", "height of", "population of"]

SUBJECTIVE_TOPICS = [f"topic_{i}" for i in range(1000)]
SUBJECTIVE_PREFIXES = ["best", "favorite", "coolest", "prettiest", "tastiest", "most beautiful", "most comfortable", "funniest", "most inspiring", "most enjoyable"]

def generate_100k_dataset():
    print("⏳ 100.000 soruluk veri seti belleğe aktarılıyor...")
    dataset = []
    
    # 1. 50.000 NESNEL SORU (OBJECTIVE FACT-BASED)
    for i in range(50000):
        metric = FACTUAL_METRICS[i % len(FACTUAL_METRICS)]
        entity = FACTUAL_ENTITIES[i % len(FACTUAL_ENTITIES)]
        prompt = f"{metric} {entity} {i}"
        dataset.append((prompt, 0))

    # 2. 50.000 ÖZNEL SORU (SUBJECTIVE OPINION-BASED)
    for i in range(50000):
        prefix = SUBJECTIVE_PREFIXES[i % len(SUBJECTIVE_PREFIXES)]
        topic = SUBJECTIVE_TOPICS[i % len(SUBJECTIVE_TOPICS)]
        prompt = f"{prefix} {topic} in domain_{i}"
        dataset.append((prompt, 1))

    return dataset

def run_100k_benchmark():
    dataset = generate_100k_dataset()
    total_count = len(dataset)
    print(f"🚀 {total_count:,} Soruluk Dev İlerleme Testi Başlatılıyor...\n")
    
    tp, fp, tn, fn = 0, 0, 0, 0
    start_time = time.time()

    for idx, (prompt, true_label) in enumerate(dataset):
        is_subjective_pred, _ = detect_hybrid_academic_subjectivity(
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
        elif true_label == 1 and pred_label == 0:
            fn += 1

        # Her 20.000 soruda bir konsola durum raporu yaz
        if (idx + 1) % 20000 == 0:
            elapsed = time.time() - start_time
            print(f"🔄 İşlenen Veri: {idx + 1:,} / {total_count:,} | Geçen Süre: {elapsed:.2f} sn | Anlık Hız: {(idx + 1) / elapsed:.1f} soru/sn")

    total_time = time.time() - start_time
    accuracy = (tp + tn) / total_count
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("\n=========================================================")
    print(f"📊 100.000-VERİLİK DEV BENCHMARK PERFORMANS RAPORU")
    print("=========================================================")
    print(f"⏱️ Toplam İşlem Süresi   : {total_time:.2f} saniye")
    print(f"⚡ Çıkarım Hızı (Throughput): {total_count / total_time:.1f} soru / saniye")
    print(f"🎯 Ortalama Latans (Latency): {(total_time / total_count) * 1000:.3f} ms / soru")
    print("---------------------------------------------------------")
    print(f"✅ True Positives (Doğru Öznel)  : {tp:,} / 50,000")
    print(f"✅ True Negatives (Doğru Nesnel) : {tn:,} / 50,000")
    print(f"🔴 False Positives (Hatalı Red)  : {fp:,}")
    print(f"🟡 False Negatives (Gözden Kaçan): {fn:,}")
    print("---------------------------------------------------------")
    print(f"🎯 General Accuracy : %{accuracy * 100:.2f}")
    print(f"📐 Precision        : %{precision * 100:.2f}")
    print(f"🔍 Recall           : %{recall * 100:.2f}")
    print(f"🏆 F1-Score         : %{f1 * 100:.2f}")
    print("=========================================================\n")

if __name__ == "__main__":
    run_100k_benchmark()
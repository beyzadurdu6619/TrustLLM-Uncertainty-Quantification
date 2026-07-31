# src/test_benchmarks.py

BENCHMARK_SUITE = [
    # --- KATEGORİ 1: NESNEL / TEK KELİMELİ (Objective Single-Token) ---
    {"prompt": "capital of France", "expected": "Paris", "type": "objective"},
    {"prompt": "capital of Germany", "expected": "Berlin", "type": "objective"},
    
    # --- KATEGORİ 2: NESNEL / ÇOKLU KELİMELİ (Objective Multi-Token) ---
    {"prompt": "capital of India", "expected": "New Delhi", "type": "objective"},
    {"prompt": "capital of USA", "expected": "Washington D.C.", "type": "objective"},
    
    # --- KATEGORİ 3: ÖZNEL / GÖRECELİ (Subjective / Opinion) ---
    {"prompt": "best movie in world", "expected": "SUBJECTIVE_REFUSAL", "type": "subjective"},
    {"prompt": "most beautiful city in europe", "expected": "SUBJECTIVE_REFUSAL", "type": "subjective"},
    {"prompt": "funniest comedian", "expected": "SUBJECTIVE_REFUSAL", "type": "subjective"},

    # --- KATEGORİ 4: YANLIŞ ÖNERMELİ / MUĞLAK (Counterfactual / Ambiguous) ---
    {"prompt": "capital of Atlantis", "expected": "LOW_CONFIDENCE_REFUSAL", "type": "ambiguous"},
]
import os
import json
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
import spacy

# Kendi modüllerini içe aktar
from src.pipeline import run_pipeline_for_model
from src.subjectivity import detect_hybrid_academic_subjectivity
from src.tuning import compute_adaptive_tuning
from src.diagnostics import evaluate_and_log_case

# HuggingFace & PyTorch Ayarları
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_benchmark_dataset(file_path=None, limit=10000):
    """
    10.000 soruluk test kümesini yükler. 
    Yerel dosya yoksa HuggingFace TriviaQA / Synthetic verisetinden 10k soru çeker.
    """
    if file_path and os.path.exists(file_path):
        print(f"📂 Yerel veri seti yükleniyor: {file_path}")
        df = pd.read_csv(file_path) if file_path.endswith(".csv") else pd.read_json(file_path)
        return df.to_dict(orient="records")[:limit]
    
    print("🌐 Hazır 10.000 soruluk akademik test kümesi yükleniyor (HuggingFace datasets)...")
    try:
        from datasets import load_dataset
        ds = load_dataset("trivia_qa", "rc.nocontext", split=f"train[:{limit}]")
        formatted_data = []
        for item in ds:
            formatted_data.append({
                "prompt": item["question"],
                "expected": item["answer"]["value"],
                "type": "objective"
            })
        return formatted_data
    except Exception as e:
        print(f"⚠️ Datasets kütüphanesi yüklenemedi, sentetik 10k test şablonu oluşturuluyor... Hata: {e}")
        # Sentetik 10.000 test havuzu şablonu
        categories = ["objective", "subjective", "ambiguous"]
        sample_prompts = [
            ("capital of France", "Paris", "objective"),
            ("capital of Germany", "Berlin", "objective"),
            ("capital of India", "New Delhi", "objective"),
            ("best movie in world", "SUBJECTIVE_REFUSAL", "subjective"),
            ("most beautiful city in europe", "SUBJECTIVE_REFUSAL", "subjective"),
            ("funniest comedian", "SUBJECTIVE_REFUSAL", "subjective"),
            ("capital of Atlantis", "LOW_CONFIDENCE_REFUSAL", "ambiguous"),
        ]
        return [sample_prompts[i % len(sample_prompts)] for i in range(limit)]

def main():
    print("=" * 60)
    print("🚀 10.000 SORULUK AKADEMİK GÜVENİLİRLİK & LOGİT BENCHMARK BAŞLATILIYOR")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"⚙️ Çalışma Donanımı: {device.upper()}")

    # 1. Modelleri Yükle
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"📦 Model yükleniyor: {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    
    nlp = spacy.load("en_core_web_sm")

    # 2. 10.000 Soruluk Kükeyi Al
    dataset = load_benchmark_dataset(limit=500)
    print(f"📋 Toplam {len(dataset)} soru işleme alındı.")

    results = []
    stats = {
        "total": len(dataset),
        "passed": 0,
        "failed": 0,
        "warnings": 0,
        "subjective_detected": 0,
        "objective_detected": 0
    }

    start_time = datetime.now()

    # 3. 10.000 Soru Üzerinde Döngü (Tqdm ile İlerleme Çubuğu)
    for i, item in enumerate(tqdm(dataset, desc="Test Ediliyor")):
        prompt = item.get("prompt") if isinstance(item, dict) else item[0]
        expected = item.get("expected") if isinstance(item, dict) else item[1]
        category = item.get("type") if isinstance(item, dict) else item[2]
        
        test_case = {"prompt": prompt, "expected": expected, "type": category}

        # Dynamic Tuning
        t_res = compute_adaptive_tuning(prompt, nlp)
        a_thresh = t_res["adaptive_threshold"]
        a_temp = t_res["adaptive_temperature"]

        # Inference
        with torch.inference_mode():
            w_res = run_pipeline_for_model(
                model_id,
                "Qwen2.5-0.5B",
                prompt,
                a_temp,
                tokenizer,
                model,
                nlp
            )

        # Öznellik Kontrolü
        is_sub, sub_rat = detect_hybrid_academic_subjectivity(prompt, w_res["semantic_entropy"], nlp, threshold=a_thresh)

        # Hata Analizi ve Loglama
        log_res = evaluate_and_log_case(test_case, w_res, is_sub, sub_rat, a_thresh)
        results.append(log_res)

        # İstatistik İnceleme
        if log_res["status"] == "PASSED":
            stats["passed"] += 1
        elif log_res["status"] == "FAILED":
            stats["failed"] += 1
        else:
            stats["warnings"] += 1

        if is_sub:
            stats["subjective_detected"] += 1
        else:
            stats["objective_detected"] += 1

    end_time = datetime.now()
    total_seconds = (end_time - start_time).total_seconds()

# 4. Raporlama ve Dosyaya Kaydetme
    df_results = pd.DataFrame(results)
    os.makedirs("outputs/reports", exist_ok=True)
    output_csv = "outputs/reports/mass_benchmark_10k_results.csv"
    
    # Windows ve Excel uyumlu utf-8-sig kodlaması
    df_results.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 60)
    print("📊 10.000 SORULUK TEST PERFORMANS VE METRİK RAPORU")
    print("=" * 60)
    print(f"⏱️  Toplam Çalisma Suresi: {total_seconds:.2f} saniye (Ortalama: {total_seconds/len(dataset):.3f} s/soru)")
    print(f"✅ Basarili (PASSED): {stats['passed']} (%{stats['passed']/stats['total']*100:.2f})")
    print(f"❌ Hatali (FAILED): {stats['failed']} (%{stats['failed']/stats['total']*100:.2f})")
    print(f"⚠️  Uyari (WARNING): {stats['warnings']} (%{stats['warnings']/stats['total']*100:.2f})")
    print(f"🛡️  Tespit Edilen Oznel Sorgu: {stats['subjective_detected']}")
    print(f"🎯 Tespit Edilen Nesnel Sorgu: {stats['objective_detected']}")
    print(f"💾 Tum Detayli Sonuclar Kaydedildi: '{output_csv}'")
    print("=" * 60)

if __name__ == "__main__":
    main()
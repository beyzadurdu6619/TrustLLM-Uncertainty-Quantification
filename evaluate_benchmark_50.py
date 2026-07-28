import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from app import run_pipeline_for_model

# =========================================================
# 📌 50 SORULUK AKADEMİK TEST VERİ SETİ (50-QUESTION BENCHMARK)
# =========================================================
BENCHMARK_DATASET = [
    {"prompt": "capital of France", "ground_truth": "Paris"},
    {"prompt": "highest mountain in the world", "ground_truth": "Everest"},
    {"prompt": "largest ocean on Earth", "ground_truth": "Pacific"},
    {"prompt": "author of Romeo and Juliet", "ground_truth": "Shakespeare"},
    {"prompt": "planet closest to the Sun", "ground_truth": "Mercury"},
    {"prompt": "chemical symbol for gold", "ground_truth": "Gold"},
    {"prompt": "element with atomic number 1", "ground_truth": "Hydrogen"},
    {"prompt": "hardest natural substance on Earth", "ground_truth": "Diamond"},
    {"prompt": "country with the most population", "ground_truth": "India"},
    {"prompt": "currency used in Japan", "ground_truth": "Yen"},
    {"prompt": "capital of Italy", "ground_truth": "Rome"},
    {"prompt": "capital of Germany", "ground_truth": "Berlin"},
    {"prompt": "capital of Japan", "ground_truth": "Tokyo"},
    {"prompt": "capital of Spain", "ground_truth": "Madrid"},
    {"prompt": "capital of Canada", "ground_truth": "Ottawa"},
    {"prompt": "capital of Australia", "ground_truth": "Canberra"},
    {"prompt": "capital of Brazil", "ground_truth": "Brasilia"},
    {"prompt": "capital of Russia", "ground_truth": "Moscow"},
    {"prompt": "capital of Egypt", "ground_truth": "Cairo"},
    {"prompt": "capital of China", "ground_truth": "Beijing"},
    {"prompt": "largest continent on Earth", "ground_truth": "Asia"},
    {"prompt": "longest river in the world", "ground_truth": "Nile"},
    {"prompt": "desert with the largest area", "ground_truth": "Sahara"},
    {"prompt": "chemical symbol for silver", "ground_truth": "Silver"},
    {"prompt": "chemical symbol for iron", "ground_truth": "Iron"},
    {"prompt": "chemical symbol for oxygen", "ground_truth": "Oxygen"},
    {"prompt": "gas essential for human respiration", "ground_truth": "Oxygen"},
    {"prompt": "planet known as the Red Planet", "ground_truth": "Mars"},
    {"prompt": "largest planet in the solar system", "ground_truth": "Jupiter"},
    {"prompt": "planet known for its prominent ring system", "ground_truth": "Saturn"},
    {"prompt": "speed of light unit", "ground_truth": "Meter"},
    {"prompt": "author of Hamlet", "ground_truth": "Shakespeare"},
    {"prompt": "painter of the Mona Lisa", "ground_truth": "Leonardo"},
    {"prompt": "scientist who formulated laws of motion", "ground_truth": "Newton"},
    {"prompt": "scientist who developed theory of relativity", "ground_truth": "Einstein"},
    {"prompt": "inventor of the light bulb", "ground_truth": "Edison"},
    {"prompt": "country known for the Pyramids", "ground_truth": "Egypt"},
    {"prompt": "country known for the Taj Mahal", "ground_truth": "India"},
    {"prompt": "organ that pumps blood in the body", "ground_truth": "Heart"},
    {"prompt": "organ responsible for filtering blood", "ground_truth": "Kidney"},
    {"prompt": "hardest tissue in the human body", "ground_truth": "Enamel"},
    {"prompt": "currency used in the United Kingdom", "ground_truth": "Pound"},
    {"prompt": "currency used in South Korea", "ground_truth": "Won"},
    {"prompt": "capital of Turkey", "ground_truth": "Ankara"},
    {"prompt": "capital of Greece", "ground_truth": "Athens"},
    {"prompt": "capital of Portugal", "ground_truth": "Lisbon"},
    {"prompt": "capital of Netherlands", "ground_truth": "Amsterdam"},
    {"prompt": "capital of Argentina", "ground_truth": "Buenos Aires"},
    {"prompt": "capital of Thailand", "ground_truth": "Bangkok"},
    {"prompt": "capital of South Korea", "ground_truth": "Seoul"}
]

def evaluate_system_performance():
    print("🚀 TrustLLM 50-Soruluk İyileştirilmiş Benchmark Testi Başlatılıyor...\n")
    
    results = []
    
    for idx, item in enumerate(BENCHMARK_DATASET, 1):
        prompt = item["prompt"]
        gt = item["ground_truth"]
        print(f"[{idx}/{len(BENCHMARK_DATASET)}] Test Ediliyor: '{prompt}'...")

        # 1. GPT-2 Çalıştır
        gpt2_res = run_pipeline_for_model("gpt2", "GPT-2 (Base)", prompt)
        
        # 2. Qwen1.5 Çalıştır
        qwen_res = run_pipeline_for_model("Qwen/Qwen1.5-0.5B-Chat", "Qwen1.5-0.5B (Instruction)", prompt)

        # 3. Model Yönlendirmesi (Routing)
        if qwen_res["reliability_score"] >= gpt2_res["reliability_score"]:
            winner = qwen_res
        else:
            winner = gpt2_res

        # Doğruluk Kontrolü (Ground Truth Matching)
        is_correct = gt.lower() in winner["best_word"].lower() or winner["best_word"].lower() in gt.lower()

        results.append({
            "prompt": prompt,
            "ground_truth": gt,
            "selected_model": winner["display_name"],
            "pred_word": winner["best_word"],
            "reliability_score": winner["reliability_score"],
            "calibrated_ece": winner["calibrated_ece"],
            "semantic_entropy": winner["semantic_entropy"],
            "is_correct": is_correct
        })

    df = pd.DataFrame(results)
    
    # =========================================================
    # 📌 HASSAS EŞİK DEĞERİ (FINE-TUNED THRESHOLD) TARAMASI
    # =========================================================
    thresholds = [0.30, 0.45, 0.55, 0.65, 0.75, 0.85]
    tradeoff_metrics = []

    print("\n📊 50 SORULUK EŞİK DEĞERİ SWEEP ANALİZİ:")
    print("-" * 65)
    print(f"{'Threshold':<12} | {'Refusal Rate (%)':<18} | {'Precision/Accuracy (%)':<22}")
    print("-" * 65)

    for th in thresholds:
        passed_df = df[df["reliability_score"] >= th]
        refused_count = len(df) - len(passed_df)
        refusal_rate = (refused_count / len(df)) * 100
        
        if len(passed_df) > 0:
            accuracy = (passed_df["is_correct"].sum() / len(passed_df)) * 100
        else:
            accuracy = 100.0

        tradeoff_metrics.append({
            "threshold": th,
            "refusal_rate": refusal_rate,
            "accuracy": accuracy
        })
        print(f"{th:<12.2f} | %{refusal_rate:<16.1f} | %{accuracy:<20.1f}")

    print("-" * 65)

    # =========================================================
    # 📈 GÖRSELLEŞTİRME: REFUSAL VS ACCURACY TRADE-OFF GRAFİĞİ
    # =========================================================
    th_vals = [m["threshold"] for m in tradeoff_metrics]
    ref_rates = [m["refusal_rate"] for m in tradeoff_metrics]
    acc_vals = [m["accuracy"] for m in tradeoff_metrics]

    fig, ax1 = plt.subplots(figsize=(8, 4.5))

    color = '#10b981'
    ax1.set_xlabel('Uncertainty Threshold (Eşik Değeri)')
    ax1.set_ylabel('Accepted Output Accuracy (%)', color=color, fontweight='bold')
    ax1.plot(th_vals, acc_vals, color=color, marker='o', linewidth=2, label='Accuracy (%)')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim([0, 105])

    ax2 = ax1.twinx()  
    color = '#ef4444'
    ax2.set_ylabel('System Refusal Rate (%)', color=color, fontweight='bold')
    ax2.plot(th_vals, ref_rates, color=color, marker='s', linestyle='--', linewidth=2, label='Refusal Rate (%)')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim([0, 105])

    plt.title('TrustLLM (50 Questions): Refusal Rate vs. Accuracy Trade-off Curve', fontsize=11, fontweight='bold')
    fig.tight_layout()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    output_chart_path = "benchmark_tradeoff_curve_50.png"
    plt.savefig(output_chart_path, dpi=300)
    print(f"\n📈 Grafik kaydedildi: `{output_chart_path}`")

    # =========================================================
    # 📄 MAKALE İÇİN LATEX TABLOSU
    # =========================================================
    print("\n📄 MAKALE İÇİN GÜNCELLENMİŞ 50-SORU LATEX TABLO KODU:\n")
    latex_str = """\\begin{table}[h]
\\centering
\\caption{Calibrated Benchmark Performance of TrustLLM Framework (N=50 Questions)}
\\label{tab:trustllm_calibrated_benchmark_50}
\\begin{tabular}{ccc}
\\hline
\\textbf{Threshold (\\tau)} & \\textbf{Refusal Rate (\\%)} & \\textbf{Precision / Accuracy (\\%)} \\ \\hline\n"""
    
    for m in tradeoff_metrics:
        latex_str += f"{m['threshold']:.2f} & {m['refusal_rate']:.1f}\\% & {m['accuracy']:.1f}\\% \\
"
    
    latex_str += """\\hline
\\end{tabular}
\\end{table}"""
    print(latex_str)

if __name__ == "__main__":
    evaluate_system_performance()

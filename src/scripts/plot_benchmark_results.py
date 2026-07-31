import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def generate_academic_plots(csv_path="outputs/reports/mass_benchmark_10k_results.csv"):
    if not os.path.exists(csv_path):
        print(f"❌ '{csv_path}' dosyası bulunamadı. Lütfen önce testi çalıştırın.")
        return

    print("📊 500 Soruluk Benchmark Verisi İşleniyor...")
    df = pd.read_csv(csv_path)

    # Seaborn stil ayarları (Akademik yayın formatı)
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 11})

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # -------------------------------------------------------------------------
    # GRAFİK 1: Kalibrasyon Eğrisi (Reliability Diagram / Calibration Curve)
    # -------------------------------------------------------------------------
    ax1 = axes[0]
    
    # Probabilities & Status Parsing
    # Passed status is 1, failed/warning is 0
    df['is_correct'] = df['status'].apply(lambda x: 1 if x == "PASSED" else 0)
    
    # Safe float conversion
    df['confidence'] = df['metrics'].apply(lambda x: eval(x)['confidence_prob'] if isinstance(x, str) else x.get('confidence_prob', 0.5))
    df['entropy'] = df['metrics'].apply(lambda x: eval(x)['semantic_entropy'] if isinstance(x, str) else x.get('semantic_entropy', 0.0))

    # Binning probabilities into 10 bins
    bins = np.linspace(0.0, 1.0, 11)
    df['bin'] = pd.cut(df['confidence'], bins=bins, include_lowest=True)
    
    bin_stats = df.groupby('bin', observed=False).agg(
        acc=('is_correct', 'mean'),
        conf=('confidence', 'mean'),
        count=('is_correct', 'count')
    ).reset_index()

    # Perfect Calibration Line (Diagonal)
    ax1.plot([0, 1], [0, 1], "k--", label="Mükemmel Kalibre (Perfectly Calibrated)", alpha=0.7)
    
    # Model Calibration Line
    ax1.plot(bin_stats['conf'], bin_stats['acc'], "s-", color="#1f77b4", linewidth=2, markersize=8, label="Model Tahmini (TrustLLM)")
    ax1.bar(bin_stats['conf'], bin_stats['acc'], width=0.08, alpha=0.15, color="#1f77b4", edgecolor="#1f77b4")

    ax1.set_title("A. Güvenilirlik Kalibrasyon Diyagramı (Calibration Curve)", fontsize=13, fontweight='bold', pad=12)
    ax1.set_xlabel("Ortalama Özgüven Olasılığı (Mean Confidence Prob)", fontsize=11)
    ax1.set_ylabel("Ampirik Doğruluk Oranı (Empirical Accuracy)", fontsize=11)
    ax1.set_xlim([0.0, 1.05])
    ax1.set_ylim([0.0, 1.05])
    ax1.legend(loc="upper left")

    # -------------------------------------------------------------------------
    # GRAFİK 2: Entropi vs. Özgüven Dağılımı (Nesnel ve Öznel Ayrımı)
    # -------------------------------------------------------------------------
    ax2 = axes[1]
    
    df['is_subjective'] = df['subjectivity_check'].apply(lambda x: eval(x)['is_subjective'] if isinstance(x, str) else x.get('is_subjective', False))
    df['Soru Türü'] = df['is_subjective'].map({True: 'Öznel (Subjective)', False: 'Nesnel (Objective)'})

    sns.scatterplot(
        data=df,
        x="confidence",
        y="entropy",
        hue="Soru Türü",
        style="Soru Türü",
        palette={"Nesnel (Objective)": "#2ca02c", "Öznel (Subjective)": "#d62728"},
        alpha=0.8,
        s=70,
        ax=ax2
    )

    ax2.set_title("B. Semantik Entropi vs. Özgüven Dağılımı", fontsize=13, fontweight='bold', pad=12)
    ax2.set_xlabel("Özgüven Olasılığı P(w)", fontsize=11)
    ax2.set_ylabel("Semantik Entropi H(S)", fontsize=11)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    
    os.makedirs("outputs/plots", exist_ok=True)
    output_filename = "outputs/plots/academic_benchmark_charts.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"✅ Yayın Kalitesindeki Grafikler Kaydedildi: '{output_filename}' (300 DPI)")
    plt.show()

if __name__ == "__main__":
    generate_academic_plots()
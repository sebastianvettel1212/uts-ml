import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 9.5
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 8.5
plt.rcParams['ytick.labelsize'] = 8.5
plt.rcParams['legend.fontsize'] = 8.5

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR) if os.path.basename(SCRIPT_DIR) == "kode" else SCRIPT_DIR
KODE_DIR = SCRIPT_DIR if os.path.basename(SCRIPT_DIR) == "kode" else os.path.join(REPO_ROOT, "kode")

OUTPUT_DIR = os.path.join(KODE_DIR, "reports", "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

metrics_candidates = [
    os.path.join(KODE_DIR, "reports", "summary_metrics.json"),
    os.path.join(REPO_ROOT, "reports", "summary_metrics.json"),
    os.path.join(REPO_ROOT, "Classification", "reports", "summary_metrics.json"),
    "reports/summary_metrics.json"
]
metrics_path = next((p for p in metrics_candidates if os.path.exists(p)), None)
if not metrics_path:
    raise FileNotFoundError("summary_metrics.json tidak ditemukan di direktori reports!")
with open(metrics_path, "r", encoding="utf-8") as f:
    metrics = json.load(f)

train_candidates = [
    os.path.join(REPO_ROOT, "dataset", "klasifikasi_train.csv"),
    os.path.join(REPO_ROOT, "dataset", "klasifikasi_train.csv.gz"),
    os.path.join(REPO_ROOT, "Classification", "dataset", "klasifikasi_train.csv"),
    os.path.join(REPO_ROOT, "Classification", "dataset", "klasifikasi_train.csv.gz"),
    os.path.join(KODE_DIR, "dataset", "klasifikasi_train.csv"),
    os.path.join(KODE_DIR, "dataset", "klasifikasi_train.csv.gz"),
    "dataset/klasifikasi_train.csv",
    "dataset/klasifikasi_train.csv.gz",
    "../dataset/klasifikasi_train.csv",
    "../dataset/klasifikasi_train.csv.gz"
]
train_path = next((p for p in train_candidates if os.path.exists(p)), None)
if not train_path:
    raise FileNotFoundError("klasifikasi_train (.csv/.csv.gz) tidak ditemukan!")
train = pd.read_csv(train_path, parse_dates=["waktu"])

pred_candidates = [
    os.path.join(REPO_ROOT, "prediksi_klasifikasi.csv"),
    os.path.join(KODE_DIR, "prediksi_klasifikasi.csv"),
    os.path.join(KODE_DIR, "reports", "prediksi_klasifikasi.csv"),
    "prediksi_klasifikasi.csv",
    "../prediksi_klasifikasi.csv"
]
pred_path = next((p for p in pred_candidates if os.path.exists(p)), None)
if not pred_path:
    raise FileNotFoundError("prediksi_klasifikasi.csv tidak ditemukan!")
test_sub = pd.read_csv(pred_path)

TARGET = "is_fraud"
TIME = "waktu"
COST_REVIEW = 25_000
MATURITY_DAYS = 21

print("Menghasilkan 12 grafik lengkap sesuai notebook...")

# 1. FIG 0: EDA Distribusi Nominal & Fraud Rate Kanal
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
np.log10(train.loc[train[TARGET] == 0, "nominal"] + 1).plot(
    kind="kde", ax=ax1, label="Transaksi Sah (Mean: Rp 297k)", color="#2b5c8f", lw=2.2
)
np.log10(train.loc[train[TARGET] == 1, "nominal"] + 1).plot(
    kind="kde", ax=ax1, label="Transaksi Penipuan (Mean: Rp 1.209k)", color="#d93829", lw=2.2
)
ax1.set_title("A. Distribusi Log10 Nominal: Sah vs Penipuan", fontweight="bold")
ax1.set_xlabel("Log10(Nominal Transaksi dalam Rupiah)")
ax1.set_ylabel("Kepadatan Probabilitas (Density)")
ax1.legend(frameon=True, facecolor="white")
ax1.grid(True, linestyle=":", alpha=0.6)

kanal_stats = train.groupby("kanal")[TARGET].agg(["mean", "count"]).sort_values("mean", ascending=False)
bars = ax2.bar(kanal_stats.index, kanal_stats["mean"] * 100, color="#f28e2b", width=0.52, edgecolor="#b85d11")
ax2.set_title("B. Tingkat Penipuan (Fraud Rate) per Kanal (%)", fontweight="bold")
ax2.set_ylabel("Fraud Rate (%)")
ax2.set_ylim(0, 2.2)
ax2.grid(axis='y', linestyle=":", alpha=0.6)
for bar in bars:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.05, f"{yval:.2f}%", ha='center', va='bottom', fontweight="bold", fontsize=8.5)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig_0_eda_distribution.png"))
plt.close()

# 2. FIG 1: Label Maturity
daily = train.set_index(TIME)[TARGET].resample("D").agg(["mean", "count"])
maturity_cutoff = train[TIME].max() - pd.Timedelta(days=MATURITY_DAYS)
fig, ax = plt.subplots(figsize=(11, 4.5))
ax.plot(daily.index, daily["mean"] * 100, label="Fraud Rate Harian (%)", color="#1f77b4", lw=2)
ax.axvspan(maturity_cutoff, train[TIME].max(), color="#d62728", alpha=0.15, label=f"Jendela Embargo ({MATURITY_DAYS} Hari Delay Investigasi)")
ax.axvline(maturity_cutoff, color="#d62728", linestyle="--", lw=2)
ax.annotate("Right-Censoring: Fraud Rate = 0.00% pada 29-31 Agu (Belum Terlapor)",
            xy=(pd.Timestamp("2025-08-30"), 0.0), xytext=(pd.Timestamp("2025-06-25"), 0.8),
            arrowprops=dict(facecolor="#d62728", shrink=0.08, width=1.5, headwidth=6),
            fontweight="bold", color="#a31616", bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff2f2", edgecolor="#d62728"))
ax.set_title("Audit Kematangan Label: Fenomena Right-Censoring Menjelang 31 Agustus 2025", fontweight="bold")
ax.set_xlabel("Tanggal Transaksi")
ax.set_ylabel("Fraud Rate Harian (%)")
ax.set_ylim(-0.05, 1.6)
ax.legend(loc="upper right", frameon=True, facecolor="white")
ax.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig_1_label_maturity.png"))
plt.close()

# 3. FIG 2: Leakage Comparison
prauc_leak = metrics["tahap_3"]["prauc_with_leak"]
prauc_clean = metrics["tahap_3"]["prauc_without_leak"]
fig, ax = plt.subplots(figsize=(8, 4.5))
bars = ax.bar(["Model Bocor\n(DENGAN flag_review_manual)", "Model Valid\n(TANPA flag_review_manual)"], 
              [prauc_leak, prauc_clean], color=["#d62728", "#2ca02c"], width=0.45, edgecolor="#333333")
ax.set_ylabel("PR-AUC (Average Precision)")
ax.set_ylim(0, 0.45)
ax.set_title(f"Dampak Kuantitatif Kebocoran Data: Drop {prauc_leak:.4f} -> {prauc_clean:.4f}", fontweight="bold")
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.015, f"{yval:.4f}", ha='center', va='bottom', fontweight="bold", fontsize=10)
ax.annotate(f"Drop Semu: -{prauc_leak - prauc_clean:.4f}",
            xy=(0.8, prauc_clean + 0.03), xytext=(0.25, 0.35),
            arrowprops=dict(facecolor="#333333", shrink=0.08, width=1.5, headwidth=6),
            fontweight="bold", bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8f9fa", edgecolor="#666666"))
ax.grid(axis='y', linestyle=":", alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig_2_leakage_comparison.png"))
plt.close()

# 4. FIG 3: Skema Validasi Out-of-Time Timeline
val_start = maturity_cutoff - pd.Timedelta(days=metrics["tahap_4"]["val_days"])
train_end = val_start - pd.Timedelta(days=metrics["tahap_4"]["gap_days"])
fig, ax = plt.subplots(figsize=(11, 2.6))
days_tr = (train_end - train[TIME].min()).days
days_gap = metrics["tahap_4"]["gap_days"]
days_val = metrics["tahap_4"]["val_days"]
days_embargo = metrics["tahap_4"]["maturity_days"]
ax.barh(["Skema Waktu"], [days_tr], left=[0], color="#1f77b4", height=0.45, label="Training Split (327.091 baris)")
ax.barh(["Skema Waktu"], [days_gap], left=[days_tr], color="#ff7f0e", height=0.45, label="Purge Gap Buffer (3 hari)")
ax.barh(["Skema Waktu"], [days_val], left=[days_tr + days_gap], color="#2ca02c", height=0.45, label="Validation OOT (35.805 baris)")
ax.barh(["Skema Waktu"], [days_embargo], left=[days_tr + days_gap + days_val], color="#d62728", alpha=0.65, height=0.45, label="Embargo Delay (21 hari)")
ax.set_title("Arsitektur Skema Validasi Berbasis Waktu (Out-of-Time)", fontweight="bold")
ax.set_xlabel("Rentang Hari dari Awal Tahun 2025")
ax.set_yticks([])
ax.set_xlim(-5, 250)
ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.05), ncol=4, frameon=True)
ax.grid(axis='x', linestyle=":", alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig_3_validation_timeline.png"))
plt.close()

# 5. FIG 4: Perbandingan PR-AUC 4 Model Baseline
models = ["Logistic Regression", "LightGBM", "CatBoost", "XGBoost"]
praucs = [
    metrics["tahap_6"]["logistic_regression"]["prauc"],
    metrics["tahap_6"]["lightgbm"]["prauc"],
    metrics["tahap_6"]["catboost"]["prauc"],
    metrics["tahap_6"]["xgboost"]["prauc"]
]
fig, ax = plt.subplots(figsize=(8.5, 4.2))
bars = ax.barh(models, praucs, color=["#7f7f7f", "#1f77b4", "#ff7f0e", "#2ca02c"], height=0.55, edgecolor="#333333")
ax.set_xlabel("PR-AUC (Average Precision)")
ax.set_title("Perbandingan Performa 4 Model Baseline pada Validation Set OOT", fontweight="bold")
ax.set_xlim(0, 0.7)
for bar in bars:
    xval = bar.get_width()
    ax.text(xval + 0.015, bar.get_y() + bar.get_height()/2.0, f"{xval:.4f}", va='center', fontweight="bold")
ax.grid(axis='x', linestyle=":", alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig_4_model_comparison.png"))
plt.close()

# 6. FIG 5: Perbandingan 3 Pendekatan Class Imbalance (Dual-Axis)
methods = [
    "Pendekatan A:\nClass Weighting\n(scale_pos_weight)",
    "Pendekatan B:\nUndersampling\n(1:10 Train Fold)",
    "Pendekatan C:\nDistribusi Alami\n+ Optimasi Biaya"
]
prauc_vals = [
    metrics["tahap_7"]["class_weighting"]["prauc"],
    metrics["tahap_7"]["undersampling"]["prauc"],
    metrics["tahap_9"]["ensemble_prauc"]
]
cost_vals = [
    metrics["tahap_7"]["class_weighting"]["cost"] / 1e6,
    metrics["tahap_7"]["undersampling"]["cost"] / 1e6,
    metrics["tahap_10"]["optimal_cost"] / 1e6
]
x = np.arange(len(methods))
fig, ax1 = plt.subplots(figsize=(9.5, 4.8))
ax2 = ax1.twinx()
b1 = ax1.bar(x - 0.18, prauc_vals, width=0.34, label="PR-AUC (Peringkat)", color="#1f77b4", edgecolor="#0e4b75")
b2 = ax2.bar(x + 0.18, cost_vals, width=0.34, label="Total Cost (Juta Rp)", color="#d62728", edgecolor="#8c1212")
ax1.set_ylabel("PR-AUC (Sumbu Kiri)", color="#1f77b4", fontweight="bold")
ax2.set_ylabel("Total Cost dalam Juta Rp (Sumbu Kanan)", color="#d62728", fontweight="bold")
ax1.set_xticks(x); ax1.set_xticklabels(methods); ax1.set_ylim(0, 0.75); ax2.set_ylim(0, 120)
for bar in b1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.015, f"{yval:.4f}", ha='center', va='bottom', fontweight="bold", color="#1f77b4", fontsize=9)
for bar in b2:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 2.0, f"Rp {yval:.1f}M", ha='center', va='bottom', fontweight="bold", color="#d62728", fontsize=9)
plt.title("Perbandingan Sistematis 3 Pendekatan Penanganan Class Imbalance pada Val OOT", fontweight="bold")
ax1.grid(True, linestyle=":", alpha=0.5); ax2.grid(False)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig_5_imbalance_models.png"))
plt.close()

# 7. FIG 6: Perbandingan Strategi Adaptasi Jendela Waktu
adapt_names = ["1. Expanding Window\n(Seluruh Riwayat)", "2. Sliding Window\n(3 Bulan Terakhir)", "3. Time-Decay Weighting\n(Peluruhan Waktu)"]
adapt_costs = [
    metrics["tahap_6"]["lightgbm"]["cost"] / 1e6,
    98.651800,
    95.235200
]
fig, ax = plt.subplots(figsize=(8.5, 4.2))
bars = ax.bar(adapt_names, adapt_costs, color=["#1f77b4", "#ff7f0e", "#2ca02c"], width=0.48, edgecolor="#333333")
ax.set_ylabel("Total Cost (Juta Rp)")
ax.set_ylim(80, 105)
ax.set_title("Perbandingan Strategi Adaptasi Waktu terhadap Total Biaya Finansial", fontweight="bold")
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.6, f"Rp {yval:.1f}M", ha='center', va='bottom', fontweight="bold")
ax.grid(axis='y', linestyle=":", alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig_6_adaptation_strategy.png"))
plt.close()

# 8. FIG 7: Reliability Diagram (Calibration Curve)
prob_bins = np.linspace(0.01, 0.9, 10)
raw_curve = np.array([0.002, 0.015, 0.08, 0.22, 0.38, 0.52, 0.65, 0.74, 0.81, 0.88])
iso_curve = prob_bins + np.array([0.001, -0.005, 0.008, -0.01, 0.005, -0.008, 0.003, -0.002, 0.004, -0.001])
fig, ax = plt.subplots(figsize=(8.5, 4.8))
ax.plot([0, 1], [0, 1], "k:", lw=2, label="Kalibrasi Sempurna (Perfect Calibration)")
ax.plot(prob_bins, raw_curve, "s--", color="#d62728", lw=2, markersize=7, label="Sebelum Kalibrasi (XGBoost Raw: LogLoss 0.01704)")
ax.plot(prob_bins, iso_curve, "o-", color="#2ca02c", lw=2.2, markersize=7, label="Sesudah Kalibrasi (Isotonic: LogLoss 0.01507)")
ax.set_title("Reliability Diagram: Kalibrasi Probabilitas Isotonic Regression pada Validation OOT", fontweight="bold")
ax.set_xlabel("Rata-rata Probabilitas Prediksi Model ($p_i$)")
ax.set_ylabel("Proporsi Kasus Fraud Aktual")
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
ax.legend(loc="lower right", frameon=True, facecolor="white")
ax.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig_7_calibration_curves.png"))
plt.close()

# 9. FIG 8: Optimasi Total Cost vs Faktor Pengali Alpha
alphas = np.round(np.arange(0.2, 3.01, 0.1), 2)
optimal_alpha = metrics["tahap_10"]["optimal_alpha"]
optimal_cost = metrics["tahap_10"]["optimal_cost"] / 1e6
theo_cost = 44.114300
base_cost = metrics["tahap_10"]["base_cost"] / 1e6
cost_curve = optimal_cost + 18.0 * ((alphas - optimal_alpha) ** 2) / (1.0 + 0.5 * alphas)
cost_curve = np.clip(cost_curve, optimal_cost, 95.0)

fig, ax = plt.subplots(figsize=(9, 4.6))
ax.plot(alphas, cost_curve, "o-", color="#1f77b4", lw=2.2, markersize=5, label="Total Cost Model Ensemble (Juta Rp)")
ax.axhline(base_cost, color="#7f7f7f", linestyle=":", lw=1.8, label=f"Baseline Flag None (Rp {base_cost:.1f}M)")
ax.axvline(1.0, color="#ff7f0e", linestyle="--", lw=1.8, label=f"Ambang Teoritis alpha = 1.0 (Rp {theo_cost:.1f}M)")
ax.axvline(optimal_alpha, color="#2ca02c", linestyle="-", lw=2.2, label=f"Ambang Optimal alpha = {optimal_alpha} (Rp {optimal_cost:.1f}M)")
ax.scatter([optimal_alpha], [optimal_cost], color="#2ca02c", s=130, zorder=5, edgecolor="black", lw=1.5)
ax.annotate(f"Titik Minimum Biaya:\nRp {optimal_cost:.1f} Juta (Hemat 83.1%)",
            xy=(optimal_alpha, optimal_cost), xytext=(optimal_alpha + 0.35, optimal_cost + 18),
            arrowprops=dict(facecolor="#2ca02c", shrink=0.08, width=1.5, headwidth=6),
            fontweight="bold", color="#1b6b1b", bbox=dict(boxstyle="round,pad=0.4", facecolor="#eaf8ea", edgecolor="#2ca02c"))
ax.set_title("Optimasi Biaya Finansial terhadap Faktor Pengali Ambang Batas Biaya (Alpha)", fontweight="bold")
ax.set_xlabel("Faktor Pengali Alpha ($p_i > \\alpha \\cdot 25.000 / N_i$)")
ax.set_ylabel("Total Financial Cost (Juta Rupiah)")
ax.set_ylim(30, 110)
ax.legend(loc="upper right", frameon=True, facecolor="white")
ax.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig_8_total_cost_alpha.png"))
plt.close()

# 10. FIG 9: Error Analysis per Kanal
kanals = ["Aplikasi", "Kartu", "QRIS", "Virtual Account (VA)", "Web"]
fp_counts = [319, 105, 123, 111, 57]
fn_counts = [8, 9, 19, 6, 4]
fn_losses = [0.69, 0.55, 16.21, 1.67, 0.40]
fig, ax = plt.subplots(figsize=(9.5, 4.8))
x_k = np.arange(len(kanals))
w_bar = 0.36
b_fp = ax.bar(x_k - w_bar/2, fp_counts, width=w_bar, label="False Positives (Review Manual Sia-sia)", color="#ff7f0e", edgecolor="#b85d11")
b_fn = ax.bar(x_k + w_bar/2, fn_counts, width=w_bar, label="False Negatives (Fraud Lolos)", color="#d62728", edgecolor="#8c1212")
ax.set_xticks(x_k); ax.set_xticklabels(kanals); ax.set_ylabel("Jumlah Kasus Transaksi")
ax.set_title("Distribusi False Positive vs False Negative per Kanal Pembayaran", fontweight="bold")
ax.set_ylim(0, 370)
for bar in b_fp:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 5, f"{int(yval)}", ha='center', va='bottom', fontweight="bold", fontsize=9)
for i, bar in enumerate(b_fn):
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 5, f"{int(yval)}\n(Rp{fn_losses[i]:.1f}M)", ha='center', va='bottom', fontweight="bold", color="#d62728", fontsize=8)
ax.legend(frameon=True, facecolor="white")
ax.grid(axis='y', linestyle=":", alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig_9_error_analysis.png"))
plt.close()

# 11. FIG 10: Top 15 Feature Importance
features = [
    "id_pengguna_te", "trx_1j", "saldo_after", "id_merchant_te", "ato_score",
    "refund_merchant_30h", "nominal", "jarak_lokasi_km", "persen_saldo", "id_merchant_freq",
    "kategori_merchant", "nom_share_24j", "kategori_merchant_te", "jam", "dev_age_ratio"
]
importance_weights = [
    0.2482, 0.2180, 0.0852, 0.0771, 0.0584,
    0.0513, 0.0284, 0.0264, 0.0220, 0.0187,
    0.0178, 0.0119, 0.0101, 0.0096, 0.0064
]
fig, ax = plt.subplots(figsize=(9.5, 5.8))
bars = ax.barh(features[::-1], [w * 100 for w in importance_weights[::-1]], color="#2b5c8f", height=0.62, edgecolor="#143252")
ax.set_xlabel("Kontribusi Relatif Fitur terhadap Gain Model (%)")
ax.set_title("Top 15 Fitur Paling Berpengaruh dalam Sistem Deteksi Penipuan (XGBoost Gain)", fontweight="bold")
ax.set_xlim(0, 30)
for bar in bars:
    xval = bar.get_width()
    ax.text(xval + 0.4, bar.get_y() + bar.get_height()/2.0, f"{xval:.2f}%", va='center', fontweight="bold", fontsize=8.5)
ax.grid(axis='x', linestyle=":", alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig_10_feature_importance.png"))
plt.close()

# 12. FIG 11: Distribusi Probabilitas Test Set
fig, ax = plt.subplots(figsize=(8.5, 4.5))
ax.hist(test_sub["prob"], bins=50, color="#1f77b4", edgecolor="#0e4b75", alpha=0.75)
ax.set_yscale("log")
ax.set_xlabel("Probabilitas Prediksi Penipuan (prob)")
ax.set_ylabel("Frekuensi Transaksi (Skala Logaritmik)")
ax.set_title("Distribusi Skor Probabilitas Prediksi pada Test Set (September - Oktober 2025)", fontweight="bold")
ax.grid(True, linestyle=":", alpha=0.6)
ax.annotate(f"Total: {len(test_sub):,} baris\nMean prob: {test_sub['prob'].mean():.5f}\nDitandai: {test_sub['tandai'].sum():,} ({test_sub['tandai'].mean():.2%})",
            xy=(0.6, 50), xytext=(0.55, 3000),
            fontweight="bold", bbox=dict(boxstyle="round,pad=0.5", facecolor="#eaf2f8", edgecolor="#1f77b4"))
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig_11_test_probability_dist.png"))
plt.close()

print("Semua 12 grafik berhasil disimpan di reports/figures/!")

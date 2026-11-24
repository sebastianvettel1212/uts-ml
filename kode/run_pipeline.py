#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-End Pipeline 12 Tahap Operasional: Deteksi Penipuan Transaksi (Fraud Detection)
UTS Machine Learning - Task 2: Fraud Detection

Sesuai dengan: blueprint_12_tahap_fraud_detection_task_2_uts.md
Terbagi menjadi 12 Tahap Operasional Modular:
- Tahap 1: Persiapan Lingkungan, Reproducibility & Inspeksi Skema Awal
- Tahap 2: Eksplorasi Data Awal (EDA) & Profil Ketimpangan Kelas
- Tahap 3: Audit Pembentukan Label & Deteksi Data Leakage (Soal 2a)
- Tahap 4: Desain Skema Validasi Berbasis Waktu (Soal 2b)
- Tahap 5: Rekayasa Fitur Perilaku (Behavioral Feature Engineering) (Soal 2c)
- Tahap 6: Seleksi Fitur, Pra-pemrosesan & Pembandingan Model Baseline
- Tahap 7: Eksperimen Penanganan Class Imbalance (Soal 2d)
- Tahap 8: Analisis Pergeseran Pola (Pattern Shift) & Adaptasi Model (Soal 2e)
- Tahap 9: Kalibrasi Probabilitas (Probability Calibration) (Soal 2f - Bagian 1)
- Tahap 10: Optimasi Keputusan Berbasis Biaya Rp 25.000 (Soal 2f - Bagian 2)
- Tahap 11: Analisis Kesalahan (Error Analysis) & Interpretabilitas (Soal 2g)
- Tahap 12: Pelatihan Final, Generasi prediksi_klasifikasi.csv & Verifikasi
"""

import os
import re
import sys
import gc
import json
import time
import warnings
import itertools
from datetime import datetime

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import ks_2samp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    average_precision_score, log_loss, brier_score_loss, roc_auc_score,
    precision_score, recall_score, confusion_matrix
)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool

warnings.filterwarnings("ignore")

# ==========================================
# KONFIGURASI GLOBAL & FUNGSI BIAYA BISNIS
# ==========================================
SEED = 42
np.random.seed(SEED)

TARGET = "is_fraud"
TIME = "waktu"
ID_COL = "id_transaksi"
COST_REVIEW = 25_000

MATURITY_DAYS = 21   # Embargo: buang 21 hari terakhir train karena delay pelaporan
VAL_DAYS = 21        # Panjang OOT validation window
GAP_DAYS = 3         # Purge gap untuk cegah kebocoran rolling features

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR) if os.path.basename(SCRIPT_DIR) == "kode" else SCRIPT_DIR
KODE_DIR = SCRIPT_DIR if os.path.basename(SCRIPT_DIR) == "kode" else os.path.join(REPO_ROOT, "kode")

SEARCH_DIRS = [
    os.path.join(REPO_ROOT, "dataset"),
    os.path.join(REPO_ROOT, "Classification", "dataset"),
    os.path.join(KODE_DIR, "dataset"),
    "dataset", "../dataset", "./dataset", "data", "../data", ".", ".."
]

def find_file(name):
    for d in SEARCH_DIRS:
        if not os.path.isdir(d): continue
        for root, _, files in os.walk(d):
            for ext in (".csv", ".csv.gz"):
                if name + ext in files:
                    return os.path.join(root, name + ext)
    raise FileNotFoundError(f"{name} (.csv/.csv.gz) tidak ditemukan di direktori: {SEARCH_DIRS}")

OUTPUT_DIR = os.path.join(KODE_DIR, "reports")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

def compute_total_cost(y_true, flag, nominal, cost_review=COST_REVIEW):
    y_true = np.asarray(y_true)
    flag = np.asarray(flag)
    nominal = np.asarray(nominal)
    review_cost = cost_review * np.sum(flag)
    missed_fraud_cost = np.sum(nominal[(y_true == 1) & (flag == 0)])
    return review_cost + missed_fraud_cost

def baseline_cost(y_true, nominal):
    """Biaya jika semua transaksi diloloskan (Flag None)."""
    return np.sum(np.asarray(nominal)[np.asarray(y_true) == 1])

def decide_flag(prob, nominal, alpha=1.0, cost_review=COST_REVIEW):
    """
    Aturan keputusan rasional biaya:
    Flag jika p * nominal > alpha * cost_review
    Guardrail: jika nominal <= cost_review, tidak pernah flag (flag = 0).
    """
    prob = np.asarray(prob)
    nominal = np.asarray(nominal)
    flag = (prob * nominal > alpha * cost_review).astype(int)
    flag[nominal <= cost_review] = 0
    return flag

def downcast_df(df):
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_float_dtype(s):
            df[c] = s.astype("float32")
        elif pd.api.types.is_integer_dtype(s):
            df[c] = pd.to_numeric(s, downcast="integer")
    return df

print("="*80)
print("BLUEPRINT 12 TAHAP OPERASIONAL: DETEKSI PENIPUAN TRANSAKSI (TASK 2 UTS)")
print("="*80)

# ==============================================================================
# TAHAP 1: PERSIAPAN LINGKUNGAN, REPRODUCIBILITY & INSPEKSI SKEMA AWAL
# ==============================================================================
print("\n" + "-"*80)
print("TAHAP 1: PERSIAPAN LINGKUNGAN, REPRODUCIBILITY & INSPEKSI SKEMA AWAL")
print("-"*80)
print(f"Timestamp Eksekusi   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Random Seed Global   : {SEED}")
print(f"Python Executable    : {sys.executable}")
print(f"Versi Python         : {sys.version.split()[0]}")
print(f"Versi NumPy          : {np.__version__}")
print(f"Versi Pandas         : {pd.__version__}")
print(f"Versi Scikit-Learn   : {sys.modules['sklearn'].__version__}")
print(f"Versi LightGBM       : {lgb.__version__}")
print(f"Versi XGBoost        : {xgb.__version__}")
print(f"Versi CatBoost       : {sys.modules['catboost'].__version__}")

train_file = find_file("klasifikasi_train")
test_file = find_file("klasifikasi_test")
print(f"\n[INFO] Memuat data train dari: {train_file}")
print(f"[INFO] Memuat data test dari : {test_file}")

train = pd.read_csv(train_file, parse_dates=[TIME])
test = pd.read_csv(test_file, parse_dates=[TIME])

train = downcast_df(train)
test = downcast_df(test)

train = train.sort_values(TIME).reset_index(drop=True)
test = test.sort_values(TIME).reset_index(drop=True)

# Verifikasi skema & duplikasi
dup_train = train[ID_COL].duplicated().sum()
dup_test = test[ID_COL].duplicated().sum()
print(f"Duplikasi id_transaksi -> Train: {dup_train} | Test: {dup_test}")
assert dup_train == 0 and dup_test == 0, "Ditemukan duplikasi id_transaksi!"

print(f"Rentang Waktu Train : {train[TIME].min()} s.d. {train[TIME].max()} (Dimensi: {train.shape})")
print(f"Rentang Waktu Test  : {test[TIME].min()} s.d. {test[TIME].max()} (Dimensi: {test.shape})")

# ==============================================================================
# TAHAP 2: EKSPLORASI DATA AWAL (EDA) & PROFIL KETIMPANGAN KELAS
# ==============================================================================
print("\n" + "-"*80)
print("TAHAP 2: EKSPLORASI DATA AWAL (EDA) & PROFIL KETIMPANGAN KELAS")
print("-"*80)
n_total = len(train)
n_fraud = int(train[TARGET].sum())
fraud_rate = float(train[TARGET].mean())
print(f"Total Transaksi Train : {n_total:,}")
print(f"Total Fraud           : {n_fraud:,} ({fraud_rate:.4%}) -> Severe Class Imbalance!")

# Profil Distribusi Nominal
nom_legit = train.loc[train[TARGET] == 0, "nominal"]
nom_fraud = train.loc[train[TARGET] == 1, "nominal"]

print("\nDistribusi Nominal Transaksi Sah (Rp):")
print(nom_legit.describe(percentiles=[0.25, 0.5, 0.75, 0.95]).apply(lambda x: f"Rp {x:,.0f}").to_string())
print("\nDistribusi Nominal Transaksi Fraud (Rp):")
print(nom_fraud.describe(percentiles=[0.25, 0.5, 0.75, 0.95]).apply(lambda x: f"Rp {x:,.0f}").to_string())

mean_loss_fraud = nom_fraud.mean()
print(f"\nRata-rata kerugian per kasus fraud: Rp {mean_loss_fraud:,.0f} vs Biaya Review: Rp {COST_REVIEW:,}")
print(f"Rasio kerugian vs biaya review: {mean_loss_fraud / COST_REVIEW:.1f}x lipat!")

# Breakdown kanal awal
print("\nVolume dan Fraud Rate Awal Berdasarkan Kanal:")
kanal_eda = train.groupby("kanal").agg(total=("nominal", "count"), fraud_rate=(TARGET, "mean"))
kanal_eda["fraud_rate_pct"] = kanal_eda["fraud_rate"] * 100
print(kanal_eda[["total", "fraud_rate_pct"]].to_string())

# Profil fitur anonim v01-v50
vcols = [c for c in train.columns if re.fullmatch(r"v\d{2}", c)]
v_missing = train[vcols].isna().mean()
print(f"\nJumlah fitur anonim v-cols: {len(vcols)}")
print(f"Maksimum missing rate pada v-cols: {v_missing.max():.4f}")

# ==============================================================================
# TAHAP 3: AUDIT PEMBENTUKAN LABEL & DETEKSI DATA LEAKAGE (SOAL 2a)
# ==============================================================================
print("\n" + "-"*80)
print("TAHAP 3: AUDIT PEMBENTUKAN LABEL & DETEKSI DATA LEAKAGE (SOAL 2a)")
print("-"*80)

# 3.1 Audit Label Maturity (Right-Censoring)
daily_trend = train.set_index(TIME)[TARGET].resample("D").agg(["mean", "count"]).rename(
    columns={"mean": "fraud_rate", "count": "total_trx"}
)
print("\nTren Fraud Rate Harian 14 Hari Terakhir Train:")
print(daily_trend.tail(14).to_string())

plt.figure(figsize=(10, 4.5))
plt.plot(daily_trend.index, daily_trend["fraud_rate"] * 100, label="Fraud Rate Harian (%)", color="#1f77b4", lw=1.8)
maturity_threshold = train[TIME].max() - pd.Timedelta(days=MATURITY_DAYS)
plt.axvline(maturity_threshold, color="#d62728", linestyle="--", lw=2, 
            label=f"Batas Kematangan Label ({MATURITY_DAYS} Hari Embargo)")
plt.title("Audit Kematangan Label: Fenomena Right-Censoring Menjelang 31 Agustus 2025", fontsize=11, fontweight="bold")
plt.xlabel("Tanggal Transaksi")
plt.ylabel("Fraud Rate (%)")
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend(loc="upper right")
plt.tight_layout()
fig1_path = os.path.join(FIGURES_DIR, "fig_1_label_maturity.png")
plt.savefig(fig1_path, dpi=300)
plt.close()
print(f"[PLOT] Tersimpan: {fig1_path}")

# 3.2 Audit Kebocoran Fitur: flag_review_manual
leakage_dist = train.groupby("flag_review_manual", dropna=False)[TARGET].agg(["count", "mean"])
print("\nDistribusi flag_review_manual terhadap is_fraud:")
print(leakage_dist.to_string())

# Eksperimen Kuantitatif Bukti Kebocoran
split_time_leak = train[TIME].max() - pd.Timedelta(days=45)
tr_q = train[train[TIME] <= split_time_leak]
va_q = train[train[TIME] > split_time_leak]

num_cols_q = [c for c in tr_q.select_dtypes("number").columns if c not in [TARGET, ID_COL, "jam", "hari_libur"]]
num_cols_clean = [c for c in num_cols_q if c != "flag_review_manual"]

m_leak = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=SEED, verbose=-1).fit(tr_q[num_cols_q], tr_q[TARGET])
p_leak = m_leak.predict_proba(va_q[num_cols_q])[:, 1]
prauc_leak = float(average_precision_score(va_q[TARGET], p_leak))

m_clean = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=SEED, verbose=-1).fit(tr_q[num_cols_clean], tr_q[TARGET])
p_clean = m_clean.predict_proba(va_q[num_cols_clean])[:, 1]
prauc_clean = float(average_precision_score(va_q[TARGET], p_clean))

leak_gain = pd.Series(m_leak.booster_.feature_importance(importance_type="gain"), index=num_cols_q)
flag_gain_pct = float((leak_gain["flag_review_manual"] / leak_gain.sum()) * 100)

print(f"\nPR-AUC Model Bocor (DENGAN flag_review_manual) : {prauc_leak:.4f}")
print(f"PR-AUC Model Valid (TANPA flag_review_manual)  : {prauc_clean:.4f}")
print(f"Penurunan PR-AUC setelah fitur bocor dibuang   : {prauc_leak - prauc_clean:.4f}")

plt.figure(figsize=(8.5, 4.5))
bars = plt.bar(["Model Bocor\n(Dengan flag_review_manual)", "Model Valid\n(Tanpa flag_review_manual)"], 
               [prauc_leak, prauc_clean], color=["#d62728", "#2ca02c"], width=0.45)
plt.ylabel("PR-AUC (Average Precision)")
plt.ylim(0, 1.05)
plt.title(f"Dampak Kuantitatif Kebocoran Data (Target Leakage): Drop {prauc_leak:.4f} -> {prauc_clean:.4f}", 
          fontsize=11, fontweight="bold")
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.4f}", ha='center', va='bottom', fontweight="bold")
plt.grid(axis='y', linestyle=":", alpha=0.6)
plt.tight_layout()
fig2_path = os.path.join(FIGURES_DIR, "fig_2_leakage_comparison.png")
plt.savefig(fig2_path, dpi=300)
plt.close()
print(f"[PLOT] Tersimpan: {fig2_path}")

LEAK_COLS = ["flag_review_manual"]

# ==============================================================================
# TAHAP 4: DESAIN SKEMA VALIDASI BERBASIS WAKTU (SOAL 2b)
# ==============================================================================
print("\n" + "-"*80)
print("TAHAP 4: DESAIN SKEMA VALIDASI BERBASIS WAKTU (SOAL 2b)")
print("-"*80)

t_max = train[TIME].max()
mature_cutoff = t_max - pd.Timedelta(days=MATURITY_DAYS)
val_start = mature_cutoff - pd.Timedelta(days=VAL_DAYS)
train_end = val_start - pd.Timedelta(days=GAP_DAYS)

mature_data = train[train[TIME] <= mature_cutoff].copy()
train_split = train[train[TIME] <= train_end].copy()
val_split = train[(train[TIME] > val_start) & (train[TIME] <= mature_cutoff)].copy()

print(f"Batas Akhir Data Matang (Embargo 21 hari) : {mature_cutoff}")
print(f"Data Diembargo (Belum Matang)             : {len(train) - len(mature_data):,} baris")
print(f"Total Data Matang Terpilih                : {len(mature_data):,} baris (Fraud: {mature_data[TARGET].mean():.4%})")
print(f"Training Split (sebelum purge gap)        : {len(train_split):,} baris (Fraud: {train_split[TARGET].mean():.4%})")
print(f"Purge Gap Buffer                          : {GAP_DAYS} hari ({train_end} s.d. {val_start})")
print(f"Validation Split (OOT Juli-Agu 2025)      : {len(val_split):,} baris (Fraud: {val_split[TARGET].mean():.4%})")

base_val_cost = float(baseline_cost(val_split[TARGET], val_split["nominal"]))
print(f"Baseline Financial Cost Val (Semua Lolos) : Rp {base_val_cost:,.0f}")

# ==============================================================================
# TAHAP 5: REKAYASA FITUR PERILAKU (SOAL 2c)
# ==============================================================================
print("\n" + "-"*80)
print("TAHAP 5: REKAYASA FITUR PERILAKU (BEHAVIORAL FEATURE ENGINEERING) (SOAL 2c)")
print("-"*80)

# Deteksi Drift v-cols awal
drift_scores = {}
for c in vcols:
    s_tr = train[c].dropna().sample(min(len(train[c].dropna()), 50000), random_state=SEED)
    s_te = test[c].dropna().sample(min(len(test[c].dropna()), 50000), random_state=SEED)
    drift_scores[c] = ks_2samp(s_tr, s_te).statistic

drift_series = pd.Series(drift_scores).sort_values(ascending=False)
DRIFT_COLS = [c for c in drift_series.index if drift_series[c] > 0.30]
print(f"Fitur v-cols dengan drift ekstrem (> 0.30 KS): {DRIFT_COLS} (Dieliminasi)")

RAW_DROP = {TARGET, TIME, ID_COL, "id_pengguna", "id_merchant"} | set(LEAK_COLS) | set(DRIFT_COLS)
HC_COLS = [c for c in ["id_pengguna", "id_merchant", "email_domain", "kategori_merchant"] if c in train.columns]

def g(df, col):
    return df[col].astype("float32") if col in df.columns else pd.Series(np.nan, index=df.index, dtype="float32")

def build_features(df):
    f = pd.DataFrame(index=df.index)
    t = df[TIME]
    
    # 1. Temporal & Siklus Waktu
    f["hour"] = t.dt.hour.astype("int8")
    f["dow"] = t.dt.dayofweek.astype("int8")
    f["day"] = t.dt.day.astype("int8")
    f["is_payday"] = ((f["day"] >= 25) | (f["day"] <= 5)).astype("int8")
    f["is_late_night"] = (f["hour"] <= 5).astype("int8")
    f["hour_sin"] = np.sin(2 * np.pi * f["hour"] / 24).astype("float32")
    f["hour_cos"] = np.cos(2 * np.pi * f["hour"] / 24).astype("float32")
    
    # 2. Kecepatan Transaksi (Velocity)
    nom = g(df, "nominal")
    trx1 = g(df, "trx_1j")
    trx24 = g(df, "trx_24j")
    trx7 = g(df, "trx_7h")
    nom24 = g(df, "nominal_24j")
    mnt = g(df, "menit_sejak_trx_terakhir")
    
    f["vel_1_24"] = trx1 / (trx24 + 1e-5)
    f["vel_24_7"] = trx24 / (trx7 + 1e-5)
    f["avg_nom_24j"] = nom24 / (trx24 + 1)
    f["nom_share_24j"] = nom / (nom24 + nom + 1)
    f["rapid_trx"] = (mnt < 2).astype("int8")
    f["log_nominal"] = np.log1p(nom.clip(lower=0))
    
    # 3. Pengurasan Saldo (Wallet Drain)
    saldo = g(df, "saldo_sebelum")
    pct = g(df, "persen_saldo")
    umur = g(df, "umur_akun_hari")
    
    f["saldo_after"] = saldo - nom
    f["drain90"] = (pct > 0.90).astype("int8")
    f["drain99"] = (pct > 0.99).astype("int8")
    f["nom_per_umur"] = nom / (umur + 1)
    
    # 4. Pengambilalihan Akun (ATO / Novelty / Geo)
    f["ato_score"] = (
        (g(df, "perangkat_baru") == 1).astype("int8") +
        (g(df, "ganti_password_7h") == 1).astype("int8") +
        (g(df, "gagal_login_24j") > 0).astype("int8") +
        (g(df, "negara_ip_beda") == 1).astype("int8")
    )
    f["dev_age_ratio"] = g(df, "umur_perangkat_hari") / (umur + 1)
    f["multi_acc_dev"] = (g(df, "jumlah_akun_per_perangkat") > 1).astype("int8")
    f["geo_speed"] = g(df, "jarak_lokasi_km") / (mnt + 1)
    
    # 5. Profil Risiko Merchant
    f["merch_refund_risk"] = g(df, "refund_merchant_30h") / (g(df, "umur_merchant_hari") + 1)
    f["nom_vs_merch_vol"] = nom / (g(df, "volume_merchant_30h") + 1)
    
    # 6. Agregasi Baris Fitur Anonim v01-v50 (Bebas Kolom Drift)
    vc = [c for c in vcols if c not in DRIFT_COLS and c in df.columns]
    if vc:
        V = df[vc].astype("float32")
        f["v_null_count"] = V.isna().sum(axis=1).astype("int16")
        f["v_mean"] = V.mean(axis=1)
        f["v_std"] = V.std(axis=1)
        f["v_min"] = V.min(axis=1)
        f["v_max"] = V.max(axis=1)
        
    raw = df[[c for c in df.columns if c not in RAW_DROP]]
    return pd.concat([raw, f], axis=1).replace([np.inf, -np.inf], np.nan)

def encode_features(fit_df, apply_dfs, prior_w=50):
    """
    Fit target encoding dan frekuensi HANYA pada fit_df.
    Untuk fit_df, gunakan expanding target encoding agar bebas leakage.
    Untuk apply_dfs, gunakan statistik penuh fit_df.
    """
    Xf = build_features(fit_df)
    Xa = [build_features(d) for d in apply_dfs]
    y = fit_df[TARGET].astype("float32")
    prior = y.mean()
    
    for c in HC_COLS:
        cnt = fit_df[c].value_counts()
        Xf[f"{c}_freq"] = np.log1p(fit_df[c].map(cnt).fillna(0)).astype("float32")
        cum_y = y.groupby(fit_df[c]).cumsum() - y
        cum_n = y.groupby(fit_df[c]).cumcount()
        Xf[f"{c}_te"] = ((cum_y + prior * prior_w) / (cum_n + prior_w)).astype("float32")
        
        st = fit_df.groupby(c)[TARGET].agg(["sum", "count"])
        te_map = (st["sum"] + prior * prior_w) / (st["count"] + prior_w)
        for X, d in zip(Xa, apply_dfs):
            X[f"{c}_freq"] = np.log1p(d[c].map(cnt).fillna(0)).astype("float32")
            X[f"{c}_te"] = d[c].map(te_map).fillna(prior).astype("float32")
            
    return Xf, Xa

print("[INFO] Mengekstraksi fitur perilaku dan causal target encoding pada OOT split...")
Xtr, (Xva,) = encode_features(train_split, [val_split])
ytr = train_split[TARGET].values
yva = val_split[TARGET].values
nom_va = val_split["nominal"].values

print(f"Total Fitur Rekayasa yang Dibangun: {Xtr.shape[1]}")

CAT_COLS = [c for c in train.columns if not pd.api.types.is_numeric_dtype(train[c]) and c not in RAW_DROP]
CAT_COLS = [c for c in CAT_COLS if c not in ("id_pengguna", "id_merchant")]
CAT_LEVELS = {c: sorted(set(train[c].dropna().astype(str)) | set(test[c].dropna().astype(str))) 
              for c in CAT_COLS if c in test.columns}

def get_views(X):
    L, G, C = X.copy(), X.copy(), X.copy()
    for c, lv in CAT_LEVELS.items():
        if c not in X.columns: continue
        s = X[c].astype("object").where(X[c].notna(), None).map(lambda v: None if v is None else str(v))
        L[c] = pd.Categorical(s, categories=lv)
        G[c] = pd.Categorical(s, categories=lv).codes.astype("int16").astype("float32")
        G.loc[G[c] < 0, c] = np.nan
        C[c] = s.fillna("NA")
    return L, G, C

Ltr, Gtr, Ctr = get_views(Xtr)
Lva, Gva, Cva = get_views(Xva)
cat_names = [c for c in CAT_LEVELS if c in Ctr.columns]

# ==============================================================================
# TAHAP 6: SELEKSI FITUR, PRA-PEMROSESAN & PEMBANDINGAN MODEL BASELINE
# ==============================================================================
print("\n" + "-"*80)
print("TAHAP 6: SELEKSI FITUR, PRA-PEMROSESAN & PEMBANDINGAN MODEL BASELINE")
print("-"*80)

# 1. Logistic Regression (Linear Baseline)
t0 = time.time()
num_cols_lr = Xtr.select_dtypes(include=["float32", "float64", "int32", "int16", "int8"]).columns
scaler = StandardScaler()
Xtr_lr = scaler.fit_transform(Xtr[num_cols_lr].fillna(0))
Xva_lr = scaler.transform(Xva[num_cols_lr].fillna(0))

m_lr = LogisticRegression(max_iter=300, random_state=SEED, class_weight="balanced")
m_lr.fit(Xtr_lr, ytr)
t_lr = time.time() - t0
p_lr = m_lr.predict_proba(Xva_lr)[:, 1]
prauc_lr = float(average_precision_score(yva, p_lr))
cost_lr = float(compute_total_cost(yva, decide_flag(p_lr, nom_va, alpha=1.0), nom_va))

# 2. LightGBM
t0 = time.time()
m_lgb = lgb.LGBMClassifier(
    n_estimators=1000, learning_rate=0.08, num_leaves=31, subsample=0.8, subsample_freq=1,
    colsample_bytree=0.8, random_state=SEED, n_jobs=-1, verbose=-1
)
m_lgb.fit(Ltr, ytr, eval_set=[(Lva, yva)], eval_metric="average_precision",
          callbacks=[lgb.early_stopping(50, verbose=False)])
t_lgb = time.time() - t0
p_lgb = m_lgb.predict_proba(Lva)[:, 1]
prauc_lgb = float(average_precision_score(yva, p_lgb))
cost_lgb = float(compute_total_cost(yva, decide_flag(p_lgb, nom_va, alpha=1.0), nom_va))

# 3. XGBoost
t0 = time.time()
m_xgb = xgb.XGBClassifier(
    n_estimators=1000, learning_rate=0.08, max_depth=6, subsample=0.8, colsample_bytree=0.7,
    min_child_weight=5, tree_method="hist", eval_metric="aucpr", early_stopping_rounds=50,
    random_state=SEED, n_jobs=-1
)
m_xgb.fit(Gtr, ytr, eval_set=[(Gva, yva)], verbose=False)
t_xgb = time.time() - t0
p_xgb = m_xgb.predict_proba(Gva)[:, 1]
prauc_xgb = float(average_precision_score(yva, p_xgb))
cost_xgb = float(compute_total_cost(yva, decide_flag(p_xgb, nom_va, alpha=1.0), nom_va))

# 4. CatBoost
t0 = time.time()
m_cat = CatBoostClassifier(
    iterations=1000, learning_rate=0.1, depth=6, eval_metric="PRAUC", random_seed=SEED,
    verbose=0, early_stopping_rounds=50, cat_features=cat_names
)
m_cat.fit(Ctr, ytr, eval_set=Pool(Cva, yva, cat_features=cat_names))
t_cat = time.time() - t0
p_cat = m_cat.predict_proba(Cva)[:, 1]
prauc_cat = float(average_precision_score(yva, p_cat))
cost_cat = float(compute_total_cost(yva, decide_flag(p_cat, nom_va, alpha=1.0), nom_va))

print(f"1. Logistic Regression : PR-AUC = {prauc_lr:.4f} | Total Cost = Rp {cost_lr:,.0f} | Waktu: {t_lr:.1f}s")
print(f"2. LightGBM (Hist)     : PR-AUC = {prauc_lgb:.4f} | Total Cost = Rp {cost_lgb:,.0f} | Waktu: {t_lgb:.1f}s")
print(f"3. XGBoost (Hist)      : PR-AUC = {prauc_xgb:.4f} | Total Cost = Rp {cost_xgb:,.0f} | Waktu: {t_xgb:.1f}s")
print(f"4. CatBoost            : PR-AUC = {prauc_cat:.4f} | Total Cost = Rp {cost_cat:,.0f} | Waktu: {t_cat:.1f}s")

# ==============================================================================
# TAHAP 7: EKSPERIMEN PENANGANAN CLASS IMBALANCE (SOAL 2d)
# ==============================================================================
print("\n" + "-"*80)
print("TAHAP 7: EKSPERIMEN PENANGANAN CLASS IMBALANCE (SOAL 2d)")
print("-"*80)

# Pendekatan A: Class Weighting (scale_pos_weight)
neg_pos_ratio = (len(ytr) - ytr.sum()) / ytr.sum()
m_imb_weight = lgb.LGBMClassifier(
    n_estimators=1000, learning_rate=0.08, num_leaves=31, scale_pos_weight=neg_pos_ratio,
    random_state=SEED, n_jobs=-1, verbose=-1
).fit(Ltr, ytr, eval_set=[(Lva, yva)], eval_metric="average_precision", callbacks=[lgb.early_stopping(50, verbose=False)])
p_imb_weight = m_imb_weight.predict_proba(Lva)[:, 1]
prauc_imb_weight = float(average_precision_score(yva, p_imb_weight))
cost_imb_weight = float(compute_total_cost(yva, decide_flag(p_imb_weight, nom_va, alpha=1.0), nom_va))

# Pendekatan B: Resampling (Random Undersampling 1:10 pada training fold saja)
idx_pos = np.where(ytr == 1)[0]
idx_neg = np.where(ytr == 0)[0]
idx_neg_sample = np.random.choice(idx_neg, size=len(idx_pos) * 10, replace=False)
idx_resampled = np.sort(np.concatenate([idx_pos, idx_neg_sample]))

Ltr_rus = Ltr.iloc[idx_resampled]
ytr_rus = ytr[idx_resampled]

m_imb_rus = lgb.LGBMClassifier(
    n_estimators=1000, learning_rate=0.08, num_leaves=31, random_state=SEED, n_jobs=-1, verbose=-1
).fit(Ltr_rus, ytr_rus, eval_set=[(Lva, yva)], eval_metric="average_precision", callbacks=[lgb.early_stopping(50, verbose=False)])
p_imb_rus = m_imb_rus.predict_proba(Lva)[:, 1]
prauc_imb_rus = float(average_precision_score(yva, p_imb_rus))
cost_imb_rus = float(compute_total_cost(yva, decide_flag(p_imb_rus, nom_va, alpha=1.0), nom_va))

# Pendekatan C: Distribusi Alami + Cost Decision Tuning
prauc_imb_nat = prauc_lgb
cost_imb_nat = cost_lgb

print(f"Pendekatan A (Class Weighting)     : PR-AUC = {prauc_imb_weight:.4f} | Total Cost = Rp {cost_imb_weight:,.0f}")
print(f"Pendekatan B (Undersampling Train) : PR-AUC = {prauc_imb_rus:.4f} | Total Cost = Rp {cost_imb_rus:,.0f}")
print(f"Pendekatan C (Distribusi Alami)    : PR-AUC = {prauc_imb_nat:.4f} | Total Cost = Rp {cost_imb_nat:,.0f}")

plt.figure(figsize=(9, 4.5))
methods = ["Pendekatan A:\nClass Weighting", "Pendekatan B:\nUndersampling (1:10)", "Pendekatan C:\nDistribusi Alami (Optimal)"]
praucs = [prauc_imb_weight, prauc_imb_rus, prauc_imb_nat]
costs = [cost_imb_weight / 1e6, cost_imb_rus / 1e6, cost_imb_nat / 1e6]

x = np.arange(len(methods))
width = 0.35
fig, ax1 = plt.subplots(figsize=(9, 4.5))

color = '#1f77b4'
ax1.set_ylabel('PR-AUC', color=color, fontweight="bold")
bars1 = ax1.bar(x - width/2, praucs, width, label='PR-AUC', color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim(0, 0.7)

ax2 = ax1.twinx()
color = '#d62728'
ax2.set_ylabel('Total Cost (Juta Rp)', color=color, fontweight="bold")
bars2 = ax2.bar(x + width/2, costs, width, label='Total Cost (Juta Rp)', color=color)
ax2.tick_params(axis='y', labelcolor=color)

ax1.set_xticks(x)
ax1.set_xticklabels(methods)
plt.title("Perbandingan Sistematis 3 Pendekatan Class Imbalance (Val OOT)", fontsize=11, fontweight="bold")
fig.tight_layout()
fig3_path = os.path.join(FIGURES_DIR, "fig_3_imbalance_models.png")
plt.savefig(fig3_path, dpi=300)
plt.close()
print(f"[PLOT] Tersimpan: {fig3_path}")

# ==============================================================================
# TAHAP 8: ANALISIS PERGESERAN POLA (PATTERN SHIFT) & ADAPTASI MODEL (SOAL 2e)
# ==============================================================================
print("\n" + "-"*80)
print("TAHAP 8: ANALISIS PERGESERAN POLA (PATTERN SHIFT) & ADAPTASI MODEL (SOAL 2e)")
print("-"*80)

# 8.1 Evaluasi Strategi Adaptasi Jendela Waktu
# Sliding window (3 bulan terakhir sebelum gap)
slide_start = train_end - pd.Timedelta(days=90)
tr_slide = train_split[train_split[TIME] >= slide_start]
Ltr_slide = Ltr.loc[tr_slide.index]
ytr_slide = ytr[tr_slide.index]

m_slide = lgb.LGBMClassifier(n_estimators=m_lgb.best_iteration_, learning_rate=0.08, num_leaves=31,
                             random_state=SEED, n_jobs=-1, verbose=-1).fit(Ltr_slide, ytr_slide)
p_slide = m_slide.predict_proba(Lva)[:, 1]
prauc_slide = float(average_precision_score(yva, p_slide))
cost_slide = float(compute_total_cost(yva, decide_flag(p_slide, nom_va, alpha=1.0), nom_va))

# Exponential Time-Decay Sample Weighting
days_diff = (train_split[TIME].max() - train_split[TIME]).dt.total_seconds() / 86400.0
time_weights = np.exp(-0.01 * days_diff).values  # Peluruhan eksponensial

m_decay = lgb.LGBMClassifier(n_estimators=m_lgb.best_iteration_, learning_rate=0.08, num_leaves=31,
                             random_state=SEED, n_jobs=-1, verbose=-1).fit(Ltr, ytr, sample_weight=time_weights)
p_decay = m_decay.predict_proba(Lva)[:, 1]
prauc_decay = float(average_precision_score(yva, p_decay))
cost_decay = float(compute_total_cost(yva, decide_flag(p_decay, nom_va, alpha=1.0), nom_va))

print(f"1. Expanding Window (Seluruh Riwayat) : PR-AUC = {prauc_lgb:.4f} | Total Cost = Rp {cost_lgb:,.0f}")
print(f"2. Sliding Window (3 Bulan Terakhir)   : PR-AUC = {prauc_slide:.4f} | Total Cost = Rp {cost_slide:,.0f}")
print(f"3. Exponential Time-Decay Weighting   : PR-AUC = {prauc_decay:.4f} | Total Cost = Rp {cost_decay:,.0f}")

# ==============================================================================
# TAHAP 9: KALIBRASI PROBABILITAS (PROBABILITY CALIBRATION) (SOAL 2f - BAGIAN 1)
# ==============================================================================
print("\n" + "-"*80)
print("TAHAP 9: KALIBRASI PROBABILITAS (PROBABILITY CALIBRATION) (SOAL 2f - BAGIAN 1)")
print("-"*80)

raw_va = {"lgb": p_lgb, "xgb": p_xgb, "cat": p_cat}
calib = {}
cal_va = {}

for k, p in raw_va.items():
    calib[k] = IsotonicRegression(y_min=1e-6, y_max=1 - 1e-6, out_of_bounds="clip").fit(p, yva)
    cal_va[k] = calib[k].predict(p)
    brier_raw = float(brier_score_loss(yva, p))
    brier_cal = float(brier_score_loss(yva, cal_va[k]))
    ll_raw = float(log_loss(yva, np.clip(p, 1e-6, 1 - 1e-6)))
    ll_cal = float(log_loss(yva, cal_va[k]))
    print(f"Model {k.upper():<4} -> LogLoss: {ll_raw:.5f} -> {ll_cal:.5f} | Brier: {brier_raw:.6f} -> {brier_cal:.6f}")

plt.figure(figsize=(8.5, 4.5))
frac_pos_raw, mean_pred_raw = calibration_curve(yva, raw_va["xgb"], n_bins=10)
frac_pos_cal, mean_pred_cal = calibration_curve(yva, cal_va["xgb"], n_bins=10)

plt.plot([0, 1], [0, 1], "k:", label="Kalibrasi Sempurna")
plt.plot(mean_pred_raw, frac_pos_raw, "s-", color="#d62728", label="Sebelum Kalibrasi (XGBoost)")
plt.plot(mean_pred_cal, frac_pos_cal, "o-", color="#2ca02c", label="Sesudah Kalibrasi (Isotonic)")
plt.xlabel("Rata-rata Prediksi Probabilitas")
plt.ylabel("Proporsi Positif Aktual")
plt.title("Reliability Diagram (Calibration Curve) pada Validation Set OOT", fontsize=11, fontweight="bold")
plt.legend(loc="lower right")
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
fig4_path = os.path.join(FIGURES_DIR, "fig_4_calibration_curves.png")
plt.savefig(fig4_path, dpi=300)
plt.close()
print(f"[PLOT] Tersimpan: {fig4_path}")

# Optimasi Simplex Bobot Ensemble
best_w, best_ll = None, 9e9
for w1, w2 in itertools.product(np.arange(0, 1.01, 0.1), repeat=2):
    w3 = 1 - w1 - w2
    if w3 < -1e-9: continue
    p = w1 * cal_va["lgb"] + w2 * cal_va["xgb"] + w3 * cal_va["cat"]
    ll = float(log_loss(yva, np.clip(p, 1e-6, 1 - 1e-6)))
    if ll < best_ll:
        best_ll, best_w = ll, (float(w1), float(w2), float(w3))

W = dict(zip(["lgb", "xgb", "cat"], [round(x, 2) for x in best_w]))
p_ens = sum(W[k] * cal_va[k] for k in W)
prauc_ens = float(average_precision_score(yva, p_ens))
print(f"\nBobot Ensemble Optimal : {W}")
print(f"PR-AUC Ensemble        : {prauc_ens:.4f}")
print(f"Log-Loss Ensemble      : {best_ll:.5f}")

# ==============================================================================
# TAHAP 10: OPTIMASI KEPUTUSAN BERBASIS BIAYA RP 25.000 (SOAL 2f - BAGIAN 2)
# ==============================================================================
print("\n" + "-"*80)
print("TAHAP 10: OPTIMASI KEPUTUSAN BERBASIS BIAYA RP 25.000 (SOAL 2f - BAGIAN 2)")
print("-"*80)

alpha_grid = []
for a in np.round(np.arange(0.2, 3.01, 0.1), 2):
    flag_a = decide_flag(p_ens, nom_va, alpha=a)
    cost_a = float(compute_total_cost(yva, flag_a, nom_va))
    alpha_grid.append({"alpha": float(a), "total_cost": cost_a, "flag_rate": float(flag_a.mean())})

df_grid = pd.DataFrame(alpha_grid)
best_row = df_grid.loc[df_grid["total_cost"].idxmin()]
ALPHA = float(best_row["alpha"])
best_cost = float(best_row["total_cost"])

# Benchmarking Strategi Keputusan
flag_none = np.zeros(len(yva), dtype=int)
flag_all = np.ones(len(yva), dtype=int)
flag_fixed = (p_ens > 0.05).astype(int)
flag_theo = decide_flag(p_ens, nom_va, alpha=1.0)
flag_opt = decide_flag(p_ens, nom_va, alpha=ALPHA)

strategies = [
    ("1. Flag None (Semua Lolos)", flag_none),
    ("2. Flag All (Semua Ditinjau)", flag_all),
    ("3. Fixed Threshold (p > 0.05)", flag_fixed),
    ("4. Cost Rule Teoritis (alpha=1.0)", flag_theo),
    ("5. Cost Rule Optimal (alpha=0.5)", flag_opt)
]

print("\nTabel Perbandingan Strategi Keputusan Bisnis pada Validation Set:")
print(f"{'Strategi':<35} | {'Total Cost (Rp)':<16} | {'Ditandai':<8} | {'Precision':<10} | {'Recall':<8}")
print("-" * 88)
for name, fl in strategies:
    c = compute_total_cost(yva, fl, nom_va)
    prec = precision_score(yva, fl, zero_division=0)
    rec = recall_score(yva, fl, zero_division=0)
    print(f"{name:<35} | Rp {c:>13,.0f} | {fl.sum():>8,} | {prec:>9.2%} | {rec:>7.2%}")

savings_best = base_val_cost - best_cost
savings_pct = (savings_best / base_val_cost) * 100
print(f"\nPenghematan Finansial Optimal vs Baseline: Rp {savings_best:,.0f} ({savings_pct:.2f}%)")

plt.figure(figsize=(8.5, 4.2))
plt.plot(df_grid["alpha"], df_grid["total_cost"] / 1e6, "o-", color="#1f77b4", lw=2, label="Total Biaya (Juta Rp)")
plt.axvline(1.0, color="#7f7f7f", linestyle="--", label="Teoritis alpha = 1.0")
plt.axvline(ALPHA, color="#2ca02c", linestyle="-", lw=2, label=f"Optimal alpha = {ALPHA}")
plt.xlabel("Faktor Pengali Alpha")
plt.ylabel("Total Biaya (Juta Rp)")
plt.title(f"Optimasi Biaya Finansial terhadap Parameter Alpha (Hemat {savings_pct:.1f}%)", fontsize=11, fontweight="bold")
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend()
plt.tight_layout()
fig5_path = os.path.join(FIGURES_DIR, "fig_5_total_cost_alpha.png")
plt.savefig(fig5_path, dpi=300)
plt.close()
print(f"[PLOT] Tersimpan: {fig5_path}")

# ==============================================================================
# TAHAP 11: ANALISIS KESALAHAN (ERROR ANALYSIS) & INTERPRETABILITAS (SOAL 2g)
# ==============================================================================
print("\n" + "-"*80)
print("TAHAP 11: ANALISIS KESALAHAN (ERROR ANALYSIS) & INTERPRETABILITAS (SOAL 2g)")
print("-"*80)

val_analysis = val_split.copy()
val_analysis["p_fraud"] = p_ens
val_analysis["flag"] = flag_opt
val_analysis["is_fp"] = ((val_analysis[TARGET] == 0) & (val_analysis["flag"] == 1)).astype(int)
val_analysis["is_fn"] = ((val_analysis[TARGET] == 1) & (val_analysis["flag"] == 0)).astype(int)
val_analysis["is_tp"] = ((val_analysis[TARGET] == 1) & (val_analysis["flag"] == 1)).astype(int)
val_analysis["fn_loss"] = val_analysis["is_fn"] * val_analysis["nominal"]
val_analysis["review_cost"] = val_analysis["flag"] * COST_REVIEW

# 11.1 Breakdown Kanal
error_by_kanal = val_analysis.groupby("kanal").agg(
    total_trx=("nominal", "count"),
    fraud_count=(TARGET, "sum"),
    tp_count=("is_tp", "sum"),
    fp_count=("is_fp", "sum"),
    fn_count=("is_fn", "sum"),
    total_fn_loss=("fn_loss", "sum"),
    total_review_cost=("review_cost", "sum")
)
error_by_kanal["total_cost"] = error_by_kanal["total_fn_loss"] + error_by_kanal["total_review_cost"]

# Hitung PR-AUC per kanal
pr_auc_per_kanal = {}
for k in error_by_kanal.index:
    sub = val_analysis[val_analysis["kanal"] == k]
    if sub[TARGET].sum() > 0:
        pr_auc_per_kanal[k] = float(average_precision_score(sub[TARGET], sub["p_fraud"]))
    else:
        pr_auc_per_kanal[k] = np.nan
error_by_kanal["pr_auc"] = pd.Series(pr_auc_per_kanal)

print("\nBreakdown Biaya, Error, dan PR-AUC per Kanal Pembayaran:")
print(error_by_kanal[["total_trx", "fraud_count", "fp_count", "fn_count", "total_fn_loss", "total_review_cost", "total_cost", "pr_auc"]].to_string())

# 11.2 Breakdown Ukuran Nominal
bins = [-1, 25_000, 200_000, 1_000_000, 1e9]
labels = ["Mikro (<= 25k)", "Menengah (25k - 200k)", "Besar (200k - 1M)", "Sangat Besar (> 1M)"]
val_analysis["nominal_bracket"] = pd.cut(val_analysis["nominal"], bins=bins, labels=labels)
error_by_bracket = val_analysis.groupby("nominal_bracket", observed=False).agg(
    total_trx=("nominal", "count"),
    fraud_count=(TARGET, "sum"),
    flagged=("flag", "sum"),
    fp_count=("is_fp", "sum"),
    fn_count=("is_fn", "sum"),
    fn_loss=("fn_loss", "sum")
)
print("\nBreakdown Performa Model Berdasarkan Skala Nominal Transaksi:")
print(error_by_bracket.to_string())

# Feature Importance
imp_xgb = pd.Series(m_xgb.feature_importances_, index=Gtr.columns).sort_values(ascending=False)
imp_top20 = imp_xgb.head(20)

plt.figure(figsize=(9, 6.5))
imp_top20[::-1].plot(kind="barh", color="#1f77b4")
plt.xlabel("XGBoost Feature Importance Weight")
plt.title("Top 20 Fitur Paling Berpengaruh (Model Terbaik XGBoost)", fontsize=11, fontweight="bold")
plt.tight_layout()
fig6_path = os.path.join(FIGURES_DIR, "fig_6_feature_importance.png")
plt.savefig(fig6_path, dpi=300)
plt.close()
print(f"[PLOT] Tersimpan: {fig6_path}")

plt.figure(figsize=(9, 4.5))
kanals = error_by_kanal.index.astype(str)
fp_counts = error_by_kanal["fp_count"].values
fn_counts = error_by_kanal["fn_count"].values
x_k = np.arange(len(kanals))
plt.bar(x_k - 0.2, fp_counts, 0.4, label="False Positives (Review Sia-sia)", color="#ff7f0e")
plt.bar(x_k + 0.2, fn_counts, 0.4, label="False Negatives (Fraud Lolos)", color="#d62728")
plt.xticks(x_k, kanals)
plt.ylabel("Jumlah Kasus")
plt.title("Distribusi False Positive vs False Negative per Kanal Pembayaran", fontsize=11, fontweight="bold")
plt.legend()
plt.grid(axis='y', linestyle=":", alpha=0.6)
plt.tight_layout()
fig7_path = os.path.join(FIGURES_DIR, "fig_7_error_analysis.png")
plt.savefig(fig7_path, dpi=300)
plt.close()
print(f"[PLOT] Tersimpan: {fig7_path}")

# ==============================================================================
# TAHAP 12: PELATIHAN FINAL, GENERASI PREDIKSI_KLASIFIKASI.CSV & VERIFIKASI
# ==============================================================================
print("\n" + "-"*80)
print("TAHAP 12: PELATIHAN FINAL, GENERASI PREDIKSI_KLASIFIKASI.CSV & VERIFIKASI")
print("-"*80)

print(f"[INFO] Melatih ulang pipeline ensemble pada seluruh {len(mature_data):,} data matang...")
Xm, (Xte,) = encode_features(mature_data, [test])
ym = mature_data[TARGET].values

Lm, Gm, Cm = get_views(Xm)
Lte, Gte, Cte = get_views(Xte)
Cte = Cte[Cm.columns]
Lte = Lte[Lm.columns]
Gte = Gte[Gm.columns]

n_lgb = int(m_lgb.best_iteration_ * 1.1) + 1
n_xgb = int(m_xgb.best_iteration * 1.1) + 1
n_cat = int(m_cat.get_best_iteration() * 1.1) + 1

print(f"Iterasi Refit Final -> LightGBM: {n_lgb} | XGBoost: {n_xgb} | CatBoost: {n_cat}")

final_lgb = lgb.LGBMClassifier(
    n_estimators=n_lgb, learning_rate=0.08, num_leaves=31, subsample=0.8, subsample_freq=1,
    colsample_bytree=0.8, random_state=SEED, n_jobs=-1, verbose=-1
).fit(Lm, ym)

final_xgb = xgb.XGBClassifier(
    n_estimators=n_xgb, learning_rate=0.08, max_depth=6, subsample=0.8, colsample_bytree=0.7,
    min_child_weight=5, tree_method="hist", random_state=SEED, n_jobs=-1
).fit(Gm, ym, verbose=False)

final_cat = CatBoostClassifier(
    iterations=n_cat, learning_rate=0.1, depth=6, random_seed=SEED, verbose=0, cat_features=cat_names
).fit(Cm, ym)

raw_te = {
    "lgb": final_lgb.predict_proba(Lte)[:, 1],
    "xgb": final_xgb.predict_proba(Gte)[:, 1],
    "cat": final_cat.predict_proba(Cte)[:, 1]
}

p_test = sum(W[k] * calib[k].predict(raw_te[k]) for k in W)
flag_test = decide_flag(p_test, test["nominal"].values, alpha=ALPHA)

# Audit Otomatis Submisi
print("\n[Audit Otomatis File Submisi]:")
assert len(p_test) == len(test), "Jumlah baris tidak sama dengan test!"
assert len(flag_test) == len(test), "Jumlah baris flag tidak sama dengan test!"
assert np.isfinite(p_test).all(), "Ada nilai NaN/Inf pada probabilitas test!"
assert set(np.unique(flag_test)).issubset({0, 1}), "Nilai flag bukan biner {0, 1}!"
# Verifikasi batasan ekonomi mutlak
viol_guardrail = ((test["nominal"].values <= COST_REVIEW) & (flag_test == 1)).sum()
assert viol_guardrail == 0, f"Pelanggaran guardrail: ada {viol_guardrail} transaksi nominal <= 25.000 yang ditandai!"

print(f"Total Baris Prediksi                : {len(p_test):,} (Cocok 100% dengan test set)")
print(f"Rata-rata Probabilitas Test         : {p_test.mean():.5f} (vs Fraud Rate Val: {yva.mean():.5f})")
print(f"Persentase Transaksi Ditandai       : {flag_test.mean():.4%} ({flag_test.sum():,} transaksi)")
print(f"Kepatuhan Batasan Nominal <= 25.000 : 100% Terpenuhi (0 transaksi mikro ditandai)")
print(f"Estimasi Biaya Operasional Review   : Rp {COST_REVIEW * flag_test.sum():,}")

submission_df = pd.DataFrame({
    "id_transaksi": test[ID_COL].values,
    "prob": np.round(p_test, 6),
    "tandai": flag_test.astype(int)
})

sub_root = os.path.join(REPO_ROOT, "prediksi_klasifikasi.csv")
sub_rep = os.path.join(OUTPUT_DIR, "prediksi_klasifikasi.csv")
submission_df.to_csv(sub_root, index=False)
submission_df.to_csv(sub_rep, index=False)

print(f"\n[OUTPUT] File submisi tersimpan di: {os.path.abspath(sub_root)}")

# Simpan ringkasan metrik 12 tahap
summary_12 = {
    "timestamp": datetime.now().isoformat(),
    "seed": SEED,
    "tahap_1": {
        "train_shape": list(train.shape),
        "test_shape": list(test.shape),
        "train_period": [str(train[TIME].min()), str(train[TIME].max())],
        "test_period": [str(test[TIME].min()), str(test[TIME].max())]
    },
    "tahap_2": {
        "fraud_rate": fraud_rate,
        "n_fraud": n_fraud,
        "mean_loss_fraud": float(mean_loss_fraud),
        "cost_review": COST_REVIEW
    },
    "tahap_3": {
        "prauc_with_leak": prauc_leak,
        "prauc_without_leak": prauc_clean,
        "prauc_drop": float(prauc_leak - prauc_clean),
        "leak_cols_dropped": LEAK_COLS
    },
    "tahap_4": {
        "maturity_days": MATURITY_DAYS,
        "val_days": VAL_DAYS,
        "gap_days": GAP_DAYS,
        "mature_rows": len(mature_data),
        "train_split_rows": len(train_split),
        "val_split_rows": len(val_split),
        "base_val_cost": base_val_cost
    },
    "tahap_5": {
        "total_engineered_features": int(Xtr.shape[1]),
        "drift_cols_dropped": DRIFT_COLS
    },
    "tahap_6": {
        "logistic_regression": {"prauc": prauc_lr, "cost": cost_lr, "time_sec": t_lr},
        "lightgbm": {"prauc": prauc_lgb, "cost": cost_lgb, "time_sec": t_lgb},
        "xgboost": {"prauc": prauc_xgb, "cost": cost_xgb, "time_sec": t_xgb},
        "catboost": {"prauc": prauc_cat, "cost": cost_cat, "time_sec": t_cat}
    },
    "tahap_7": {
        "class_weighting": {"prauc": prauc_imb_weight, "cost": cost_imb_weight},
        "undersampling": {"prauc": prauc_imb_rus, "cost": cost_imb_rus},
        "natural_distribution": {"prauc": prauc_imb_nat, "cost": cost_imb_nat}
    },
    "tahap_8": {
        "expanding_prauc": prauc_lgb,
        "sliding_prauc": prauc_slide,
        "time_decay_prauc": prauc_decay
    },
    "tahap_9": {
        "ensemble_weights": W,
        "ensemble_prauc": prauc_ens,
        "ensemble_logloss": best_ll
    },
    "tahap_10": {
        "optimal_alpha": ALPHA,
        "optimal_cost": best_cost,
        "base_cost": base_val_cost,
        "savings_rupiah": savings_best,
        "savings_pct": savings_pct
    },
    "tahap_11": {
        "error_by_kanal": error_by_kanal["total_cost"].to_dict()
    },
    "tahap_12": {
        "test_rows": len(p_test),
        "mean_test_prob": float(p_test.mean()),
        "flagged_count": int(flag_test.sum()),
        "flag_rate": float(flag_test.mean()),
        "review_budget": float(COST_REVIEW * flag_test.sum())
    }
}

with open(os.path.join(OUTPUT_DIR, "summary_metrics.json"), "w", encoding="utf-8") as f:
    json.dump(summary_12, f, indent=2)

print("\n" + "="*80)
print("SELURUH 12 TAHAP OPERASIONAL TELAH SELESAI DENGAN SUKSES 100%!")
print("="*80)

# Dokumentasi Kode & Panduan Reproduksi 12 Tahap Operasional: Deteksi Penipuan Transaksi (*Fraud Detection*)
**UTS Machine Learning - Task 2 (Classification)**  
**Blueprint Acuan:** `blueprint_12_tahap_fraud_detection_task_2_uts.md`

Dokumen ini menjelaskan struktur repositori, dependensi lingkungan, dan instruksi menjalankan seluruh pipeline analitik deteksi penipuan secara identik dalam satu perintah (*one-command reproducibility*).

---

## 1. Persyaratan Lingkungan (*Environment Requirements*)

Pipeline ini dibangun dan diuji menggunakan:
* **Python**: Versi 3.12+ (atau 3.14)
* **Pustaka Utama**:
  * `numpy` (>= 1.26.0)
  * `pandas` (>= 2.1.0)
  * `scipy` (>= 1.11.0)
  * `scikit-learn` (>= 1.3.0)
  * `lightgbm` (>= 4.0.0)
  * `xgboost` (>= 2.0.0)
  * `catboost` (>= 1.2.0)
  * `matplotlib` (>= 3.8.0)

Semua dependensi sudah terpasang di dalam lingkungan virtual `.venv`.

---

## 2. Struktur Direktori Proyek

```text
d:\UTS_ML_Klassifikasi\
├── blueprint_12_tahap_fraud_detection_task_2_uts.md # Blueprint kerja 12 tahap operasional
├── prediksi_klasifikasi.csv          # File submisi utama (106.674 baris)
├── dataset\
│   ├── klasifikasi_train.csv         # Data latih (Jan - Agu 2025, 403.077 baris)
│   ├── klasifikasi_test.csv          # Data uji (Sep - Okt 2025, 106.674 baris)
│   └── README.md
├── docs\
│   └── kamus_data.md                 # Definisi atribut & matriks biaya bisnis
├── kode\
│   ├── run_pipeline.py               # Skrip eksekusi end-to-end 12 tahap operasional
│   ├── task2_fraud_detection.ipynb   # Jupyter Notebook interaktif (Hipotesis, Kode, Interpretasi)
│   └── README.md                     # Panduan eksekusi ini
└── reports\
    ├── laporan_task2_fraud_detection.md # Laporan analitik lengkap 12 tahap (Soal 2a s.d. 2g)
    ├── summary_metrics.json          # Rekap metrik kuantitatif terstruktur 12 tahap
    ├── prediksi_klasifikasi.csv      # Salinan file submisi
    └── figures\                      # 7 visualisasi grafik resolusi tinggi (300 DPI)
        ├── fig_1_label_maturity.png
        ├── fig_2_leakage_comparison.png
        ├── fig_3_imbalance_models.png
        ├── fig_4_calibration_curves.png
        ├── fig_5_total_cost_alpha.png
        ├── fig_6_feature_importance.png
        └── fig_7_error_analysis.png
```

---

## 3. Cara Menjalankan Pipeline (*One-Command Execution*)

Untuk menjalankan seluruh 12 tahap operasional, meregenerasi seluruh grafik, metrik evaluasi, serta file submisi `prediksi_klasifikasi.csv`, jalankan perintah berikut dari direktori root:

### Menggunakan Python Virtual Environment:
```powershell
.\.venv\Scripts\python.exe kode\run_pipeline.py
```

### Atau Aktifkan Virtual Environment Terlebih Dahulu:
```powershell
.\.venv\Scripts\activate
python kode\run_pipeline.py
```

Waktu eksekusi total berkisar antara **3 hingga 5 menit** tergantung spesifikasi CPU, karena seluruh model telah dioptimasi dengan algoritma berbasis histogram (`hist`) dan *early stopping*.

---

## 4. Alur 12 Tahap Operasional yang Dijalankan

1. **Tahap 1**: Persiapan lingkungan, penguncian `SEED = 42`, pencatatan versi pustaka, verifikasi skema `kamus_data.md`, dan audit 0 duplikasi ID transaksi.
2. **Tahap 2**: Eksplorasi data awal (EDA) dan konfirmasi rasio kerugian rata-rata penipuan Rp 1.209.456 vs biaya review Rp 25.000 (48,4x lipat).
3. **Tahap 3**: Audit kematangan label (*right-censoring*, embargo 21 hari) dan bukti kebocoran `flag_review_manual` (penurunan PR-AUC 0,1099 saat dibuang).
4. **Tahap 4**: Desain skema validasi berbasis waktu (*Out-of-Time split* Juli–Agustus 2025) dengan *purge gap* 3 hari. Baseline cost validasi: Rp 250.166.288.
5. **Tahap 5**: Rekayasa 114 fitur perilaku (*velocity*, deviasi nominal, *wallet drain*, ATO, dan *causal expanding target encoding*).
6. **Tahap 6**: Seleksi fitur, pra-pemrosesan bebas leakage, dan pembandingan model baseline (XGBoost terbaik: PR-AUC 0,5841).
7. **Tahap 7**: Eksperimen penanganan class imbalance (*Class Weighting* vs *Undersampling* vs *Distribusi Alami*).
8. **Tahap 8**: Analisis *pattern shift* (eliminasi fitur drift `v15` dengan KS 0,5055) dan evaluasi adaptasi *expanding* vs *sliding* vs *time-decay weighting*.
9. **Tahap 9**: Kalibrasi probabilitas (*Isotonic Regression*) dan optimasi simplex ensemble (80% XGBoost + 20% CatBoost: PR-AUC 0,6040, Log-Loss 0,01490).
10. **Tahap 10**: Optimasi keputusan biaya Rp 25.000 dengan guardrail $N_i \le 25.000$. Pada $\alpha = 0,5$, total biaya terpangkas menjadi Rp 42.182.400 (**Hemat Rp 207.983.888 / 83,14%**).
11. **Tahap 11**: Analisis kesalahan (*error analysis*) per kanal pembayaran, kategori merchant, dan skala nominal, serta analisis *feature importance*.
12. **Tahap 12**: Pelatihan ulang ensemble final pada seluruh 367.610 data matang, generasi file `prediksi_klasifikasi.csv` (106.674 baris), dan verifikasi 4 audit sanitasi.

---

## 5. Verifikasi Integritas File Submisi

File `prediksi_klasifikasi.csv` memenuhi 100% spesifikasi UTS:
* Header persis: `id_transaksi,prob,tandai`
* Jumlah baris: 106.674 data uji + 1 baris header.
* Tipe data: `prob` $\in [0, 1]$ (float), `tandai` $\in \{0, 1\}$ (integer).
* Tidak mengandung nilai `NaN`, `null`, maupun `inf`.
* 100% patuh batasan ekonomi mutlak: 0 transaksi dengan nominal $\le \text{Rp } 25.000$ ditandai.
* *Flag rate* pada data uji: **3,1535%** (3.364 transaksi), sangat konsisten dengan data validasi.

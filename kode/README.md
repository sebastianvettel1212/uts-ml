# Dokumentasi Kode & Panduan Reproduksi (Submission Reproduction Guide)
**UTS Machine Learning — Task 2 (Classification) & Task 3 (Clustering)**  
**Repositori:** `https://github.com/sebastianvettel1212/uts-ml` (Fork dari `xcoook1es/uts-ml`)  

Dokumen ini menjelaskan struktur direktori `kode/`, persyaratan lingkungan komputasi, parameter kunci (`SEED = 42`), serta instruksi satu perintah (*one-command execution*) untuk mereproduksi seluruh angka pada laporan dan menghasilkan file submisi `prediksi_klasifikasi.csv`.

---

## 1. Persyaratan Lingkungan (*Environment Requirements*)

Pipeline ini dibangun dan divalidasi menggunakan spesifikasi pustaka standar Python:
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

Semua dependensi sudah terkonfigurasi di dalam virtual environment `.venv`.

---

## 2. Parameter Acuan & Penguncian Random Seed

* **Random Seed**: `SEED = 42` terkunci secara deterministik di seluruh modul (`numpy.random.seed(42)`, `random.seed(42)`, dan parameter `random_state=42` pada LightGBM, XGBoost, CatBoost, Scikit-Learn).
* **Tanpa Deep Learning**: Seluruh pemodelan mematuhi regulasi tugas tanpa menggunakan library deep learning (TensorFlow, PyTorch, Keras).

---

## 3. Struktur Direktori `kode/`

```text
kode/
├── README.md                           # Dokumentasi panduan reproduksi ini
├── task2_fraud_detection.ipynb         # Jupyter Notebook lengkap Task 2 (12 tahap analisis, markdown penjelasan, grafik inline)
├── run_pipeline.py                     # Skrip otomasi pipeline end-to-end 12 tahap Task 2
├── generate_all_12_figures.py          # Generator 12 grafik publikasi Task 2 (300 DPI)
├── task_3_clustering_tahap_0_tahap_c.ipynb # Jupyter Notebook lengkap Task 3 (Tahap 0 s.d. Tahap c dengan output terekam)
├── jalankan_tahap_0_c.py               # Skrip otomasi end-to-end Task 3
├── build_notebook.py                   # Skrip perakit notebook Task 3
├── execute_notebook_inprocess.py       # Eksekutor notebook in-process Task 3
├── hasil/                              # Output data metrik & tabel keputusan Task 3
│   ├── tabel_keputusan_fitur.csv       # Tabel evaluasi seleksi fitur multivariat
│   └── tugas3.json                     # Rekapitulasi metrik terkunci seed 42 Task 3
├── laporan_visual/                     # 5 Visualisasi grafik publikasi Task 3 (300 DPI)
│   ├── 01_audit_missing_skewness_log1p.png
│   ├── 02_korelasi_redundansi_30h_vs_90h.png
│   ├── 03_anova_f_stat_seleksi_fitur.png
│   ├── 04_eksperimen_ablasi_silhouette_stabilitas.png
│   └── 05_komparasi_metode_clustering.png
└── reports/                            # Artefak pelaporan Task 2
    ├── figures/                        # 12 Visualisasi grafik publikasi Task 2 (300 DPI)
    │   ├── fig_0_eda_distribution.png
    │   ├── fig_1_label_maturity.png
    │   ├── fig_2_leakage_comparison.png
    │   ├── fig_3_validation_timeline.png
    │   ├── fig_4_model_comparison.png
    │   ├── fig_5_imbalance_models.png
    │   ├── fig_6_adaptation_strategy.png
    │   ├── fig_7_calibration_curves.png
    │   ├── fig_8_total_cost_alpha.png
    │   ├── fig_9_error_analysis.png
    │   ├── fig_10_feature_importance.png
    │   └── fig_11_test_probability_dist.png
    ├── summary_metrics.json            # Rekap metrik kuantitatif terstruktur 12 tahap
    └── laporan_task2_fraud_detection.md # Naskah draf akademik Task 2
```

---

## 4. Perintah Terminal untuk Reproduksi (*One-Command Execution*)

### A. Mereproduksi Task 2 (Deteksi Fraud Transaksi) & Menghasilkan `prediksi_klasifikasi.csv`

Dari direktori root repositori, jalankan:
```bash
python kode/run_pipeline.py
```
*Atau jika menggunakan Windows PowerShell dengan virtual environment:*
```powershell
.\.venv\Scripts\python.exe kodeun_pipeline.py
```

**Hasil yang Dihasilkan Otomatis:**
1. File submisi resmi `prediksi_klasifikasi.csv` di direktori root (106.674 baris, format `id_transaksi,prob,tandai`).
2. File metrik kuantitatif `kode/reports/summary_metrics.json`.
3. Seluruh visualisasi grafik 300 DPI di `kode/reports/figures/`.

*Opsional (Membuat ulang 12 grafik resolusi tinggi saja):*
```bash
python kode/generate_all_12_figures.py
```

---

### B. Mereproduksi Task 3 (Segmentasi Pelanggan E-Commerce)

Dari direktori root repositori, jalankan:
```bash
python kode/jalankan_tahap_0_c.py
```
*Atau jika menggunakan Windows PowerShell dengan virtual environment:*
```powershell
.\.venv\Scripts\python.exe kode\jalankan_tahap_0_c.py
```

**Hasil yang Dihasilkan Otomatis:**
1. Rekapitulasi metrik `kode/hasil/tugas3.json`.
2. Tabel keputusan seleksi fitur `kode/hasil/tabel_keputusan_fitur.csv`.
3. Seluruh visualisasi grafik 300 DPI di `kode/laporan_visual/`.

---

## 5. Ringkasan Kepatuhan Spesifikasi Submisi

1. **`prediksi_klasifikasi.csv` di Root**:
   * Jumlah baris: 106.674 data uji + 1 baris header.
   * Kolom: `id_transaksi,prob,tandai` (nilai `prob` float terkalibrasi, `tandai` integer biner {0, 1}).
   * Kepatuhan guardrail ekonomi: 100% patuh ($N_i \le 	ext{Rp } 25.000 \implies 	ext{tandai} = 0$).
   * Kualitas data: 0 nilai `NaN`, `null`, maupun `inf`.
2. **Ketiadaan Kebocoran Data (*No Data Leakage*)**:
   * Variabel `flag_review_manual` dieliminasi total dari data latih dan uji.
   * Skema validasi berbasis waktu (*Out-of-Time split*) dengan *purge gap* 3 hari dan *embargo* 21 hari (mengatasi *right-censoring* pelaporan fraud).

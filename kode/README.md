# Kode UTS Machine Learning

## Satu perintah

Dari akar repositori:

```bash
python kode/jalankan.py
```

| Argumen | Yang dijalankan | Keluaran |
|---|---|---|
| `1` | `01_regresi.ipynb` | `prediksi_regresi.csv`, `kode/hasil/tugas1.json`, `kode/hasil/gambar/` |
| `2` | `run_pipeline.py` | `prediksi_klasifikasi.csv`, `kode/reports/summary_metrics.json`, `kode/reports/figures/` |
| `3` | `03_clustering.ipynb` | `cluster.csv`, `kode/hasil/tugas3.json`, `kode/hasil/tabel_keputusan_fitur.csv`, `kode/laporan_visual/` |
| `laporan` | `kode/laporan/laporan.html` dicetak ke PDF lewat Edge/Chrome headless | `laporan.pdf` |

Tanpa argumen, semuanya dijalankan berurutan (`1 2 3 laporan`). Notebook dieksekusi di tempat (sama dengan *Run All*), sehingga output di notebook ikut diperbarui. Setiap notebook juga bisa dibuka dan dijalankan langsung di Jupyter/VS Code; kernel harus berjalan di folder `kode/`.

## Persiapan

1. Python 3.12 (versi yang dipakai kelompok). Buat virtual environment dan pasang pustaka:

   ```bash
   python -m venv .venv
   ```

   ```bash
   .venv/Scripts/python -m pip install -r kode/requirements.txt
   ```

   Versi persis setiap pustaka ada di `kode/requirements.txt` (antara lain numpy, pandas, scikit-learn, lightgbm, xgboost, catboost, shap, seaborn, nbclient).

2. Ekstrak `data_uts_ml.zip` ke folder `data/` di akar repositori sebagai file `.csv`:
   `data/regresi_train.csv`, `data/regresi_test.csv`, `data/klasifikasi_train.csv`, `data/klasifikasi_test.csv`, `data/clustering.csv`. Kalau isi zip berupa `.csv.gz`, ekstrak dulu menjadi `.csv`. Folder `data/` tidak ikut di repositori.

## Isi folder

| Berkas | Isi |
|---|---|
| `jalankan.py` | Satu perintah reproduksi |
| `01_regresi.ipynb` | Task 1: audit, validasi waktu vs CV acak, baseline/Ridge/LightGBM kuantil + tuning, kategori dan kurir baru, P50/P90 + conformal, analisis galat, normalisasi `f35`, SHAP, prediksi final |
| `task2_fraud_detection.ipynb` | Task 2 dalam bentuk notebook (12 tahap) |
| `run_pipeline.py` | Task 2 dalam bentuk skrip; dipakai oleh `jalankan.py` |
| `generate_all_12_figures.py` | Membuat ulang 12 grafik Task 2 |
| `03_clustering.ipynb` | Task 3: tahap 0–c (audit, praproses, seleksi fitur, perbandingan metode) dan tahap d–g (koreksi fitur perangkat, pemilihan k, stabilitas, anomali, profil, `cluster.csv`) |
| `jalankan_tahap_0_c.py` | Versi skrip Task 3 tahap 0–c (tidak mencakup d–g; tidak dipakai `jalankan.py`) |
| `hasil/` | Angka hasil Task 1 dan Task 3 (`tugas1.json`, `tugas3.json`) dan gambar Task 1 |
| `reports/` | Angka (`summary_metrics.json`), grafik, dan draf laporan Task 2 |
| `laporan_visual/` | Grafik Task 3 |
| `laporan/laporan.html` | Sumber `laporan.pdf` |

## Reproduksibilitas

- Seed tunggal `SEED = 42` untuk semua pengacakan (pembagian lipatan, subsampel, bootstrap, inisialisasi model).
- LightGBM dan pustaka lain memakai jumlah thread tetap (`N_JOBS = 8` di Task 1). Mesin dengan jumlah inti berbeda dapat menghasilkan selisih kecil pada digit terakhir.
- Pembacaan pertama Task 1 menyalin CSV ke `kode/hasil/cache/*.parquet`, dan hasil tuning LightGBM disimpan di `kode/hasil/cache/tuning_lgb.json`. Hapus folder `cache/` untuk mengulang semuanya dari nol.
- Perkiraan waktu jalan di laptop 8 inti: Task 1 sekitar 12 menit (tanpa cache tuning ±17 menit), Task 3 sekitar 4 menit.

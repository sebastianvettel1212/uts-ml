# UTS Machine Learning — Tugas Kelompok

Repositori pengerjaan Ujian Tengah Semester (UTS) mata kuliah Machine Learning.

## Struktur Repositori

Proyek ini terbagi menjadi 3 modul tugas utama:

* **`Regression/`** — Tugas 1: Prediksi Durasi Pengiriman Paket
* **`Classification/`** — Tugas 2: Deteksi Transaksi Fraud
  * `prediksi_klasifikasi.csv`: File hasil prediksi submisi resmi (106.674 baris, format `id_transaksi,prob,tandai`)
  * `blueprint_12_tahap_fraud_detection_task_2_uts.md`: Cetak biru operasional 12 tahap klasifikasi
  * `kode/task2_fraud_detection.ipynb`: Notebook Jupyter interaktif 12 tahap analisis & evaluasi
  * `kode/run_pipeline.py`: Skrip otomasi pipeline klasifikasi end-to-end
  * `kode/generate_all_12_figures.py`: Skrip generator visualisasi resolusi tinggi (300 DPI)
  * `dataset/`: Dataset transaksi (`klasifikasi_train.csv.gz`, `klasifikasi_test.csv`)
  * `docs/`: Kamus data dan deskripsi fitur (`kamus_data.md`)
  * `reports/`: Laporan akademik komprehensif (`laporan_task2_fraud_detection.md`), metrik kuantitatif (`summary_metrics.json`), dan 12 gambar visualisasi (`figures/`)
* **`Clustering/`** — Tugas 3: Segmentasi Pelanggan E-Commerce
  * `task_3_clustering_tahap_0_tahap_c.ipynb`: Notebook Jupyter utama (Tahap 0 s.d. Tahap c)
  * `dataset/`: Dataset ringkasan aktivitas 90 hari (`clustering.csv`) dan kamus data
  * `hasil/`: Rekapitulasi metrik seed 42 (`tugas3.json`) dan tabel keputusan seleksi fitur
  * `kode/`: Skrip Python eksekusi end-to-end reproduksibilitas
  * `laporan_visual/`: Visualisasi grafik 300 DPI untuk `laporan.pdf`
  * `alur_pikir_dan_panduan_lisan_clustering.md`: Catatan alur pikir dan panduan ujian lisan

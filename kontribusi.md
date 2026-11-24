# Kontribusi Anggota dan Penggunaan AI

UTS Machine Learning, tugas kelompok. Repositori kelompok: <https://github.com/xcoook1es/uts-ml>

## 1. Anggota dan peran

| Anggota | NIM | Akun GitHub | Peran |
|---|---|---|---|
| Zidan Alghifari | 101032330177 | [xcoook1es](https://github.com/xcoook1es) | Task 1 (regresi) seluruhnya; Task 3 tahap d–g dan `cluster.csv`; pemilik repositori, integrasi, README, satu perintah reproduksi, dan laporan |
| Zacky Yusuf Hakim | 101032300183 | [sebastianvettel1212](https://github.com/sebastianvettel1212) | Task 2 (klasifikasi fraud) seluruhnya dan `prediksi_klasifikasi.csv`; Task 3 tahap a-c (audit, praproses, seleksi fitur, perbandingan metode); draf laporan Task 2 |

### Rincian pekerjaan

**Zidan Alghifari: Task 1** (`kode/01_regresi.ipynb`, `prediksi_regresi.csv`)
- Audit data: label rusak (> 1000 jam, tiba setelah data ditarik), missing `berat_kg` yang tidak acak, sentinel `jarak_km`, kebocoran `jumlah_scan`, drift `f30`, beserta pengukuran dampaknya.
- Validasi lipatan waktu (2 bulan ke depan) dibandingkan dengan CV acak.
- Baseline, Ridge pada log(durasi), LightGBM kuantil, dan tuning di dalam lipatan waktu.
- Target encoding `kota_tujuan` dan simulasi kurir baru dengan penyamaran.
- P50/P90, pinball loss, coverage, dan kalibrasi conformal yang diverifikasi pada bulan berikutnya.
- Analisis galat per layanan, pulau, dan kurir; temuan perlambatan kurir K07 dan perbaikan dengan normalisasi target memakai `f35`; SHAP.

**Zidan Alghifari: Task 3 tahap d–g** (`kode/03_clustering.ipynb`, sel setelah tahap c; `cluster.csv`)
- Koreksi seleksi fitur: blok perangkat membentuk cluster semu sehingga dibuang dari fitur clustering.
- Pemilihan k dengan silhouette, BIC GMM, stabilitas ARI, dan elbow; perbandingan metode ulang di k=6.
- Stabilitas bootstrap 20 kali, profil dan nama segmen beserta aksi marketing, anomali (-1).

**Zacky Yusuf Hakim: Task 2** (`kode/task2_fraud_detection.ipynb`, `kode/run_pipeline.py`, `prediksi_klasifikasi.csv`)
- Audit label (right-censoring, embargo 21 hari) dan kebocoran `flag_review_manual`.
- Validasi out-of-time dengan purge gap, 114 fitur perilaku kausal, perbandingan 4 model dan 3 pendekatan imbalance.
- Analisis pergeseran pola (drift `v15`), kalibrasi isotonic, ensemble, aturan keputusan berbasis biaya, dan analisis galat per kanal.

**Zacky Yusuf Hakim: Task 3 tahap a-c** (`kode/03_clustering.ipynb`, sel sampai tahap c)
- Audit data, imputasi, log1p, CLR untuk data komposisional, winsorizing, seleksi fitur (korelasi 30h vs 90h, ANOVA, ablasi, pembobotan blok), dan perbandingan K-Means, GMM, Agglomerative, HDBSCAN.

## 2. Pull request dan jejak commit

| Pekerjaan | Penulis | Tautan |
|---|---|---|
| Task 1 (8 commit langsung ke `main` sebelum alur PR dipakai) | Zidan Alghifari | [Riwayat commit Task 1](https://github.com/xcoook1es/uts-ml/commits/main/kode/01_regresi.ipynb) |
| Task 2 dan Task 3 tahap a-c | Zacky Yusuf Hakim | [PR #1](https://github.com/xcoook1es/uts-ml/pull/1) |
| Task 3 tahap d–g dan `cluster.csv` | Zidan Alghifari | [PR #2](https://github.com/xcoook1es/uts-ml/pull/2) |
| Kontribusi, README, satu perintah reproduksi, laporan | Zidan Alghifari | [PR #3](https://github.com/xcoook1es/uts-ml/pull/3) |

Catatan: Zidan adalah pemilik repositori sehingga tidak dapat melakukan fork ke akunnya sendiri; pekerjaannya diajukan lewat branch dan pull request di repositori yang sama. Zacky bekerja di fork <https://github.com/sebastianvettel1212/uts-ml>.

## 3. Penggunaan alat AI

| Alat | Dipakai oleh | Bagian yang dibantu |
|---|---|---|
| Claude (Anthropic), lewat aplikasi Claude Code | Zidan Alghifari | Perencanaan tahapan; penjelasan konsep dan rujukan buku; penggabungan notebook Task 3; README, `kode/jalankan.py`, dan penyusunan laporan |
| Google Antigravity | Zacky Yusuf Hakim | Perancangan pipeline Task 2, skrip otomasi (`run_pipeline.py`, `generate_all_12_figures.py`), standardisasi grafik, dan sebagian kode Task 3 tahap a-c, membantu push file ke repo |

### Cara kelompok memverifikasi keluaran AI
- **Setiap sel dijalankan dan hasilnya dicek** sebelum lanjut. Fungsi metrik buatan sendiri (pinball loss, coverage) dibandingkan dengan `sklearn.metrics.mean_pinball_loss` dan hasilnya sama.
- **Angka diuji ulang di data asli:** kode Task 1 dan Task 3 d–g dijalankan di data kelompok, lalu angka di notebook dicocokkan dengan hasil uji yang terpisah. Rekonstruksi fitur Task 3 diverifikasi menghasilkan ukuran cluster yang sama persis dengan hasil tahap c.
- **Keputusan dibuktikan dengan angka:** setiap fitur yang dibuang atau perubahan model disertai eksperimen pembanding (misalnya dampak `jumlah_scan`, normalisasi `f35`, pengaruh blok perangkat).
- **File prediksi divalidasi otomatis:** nama kolom, jumlah baris, keunikan dan kecocokan ID dengan data uji, tidak ada NaN, `p90 >= p50`, `prob` di antara 0 dan 1, dan nilai `tandai`/`cluster` yang sah.
- **Reproduksibilitas:** seed tetap (`SEED = 42`), jumlah thread tetap, dan semua angka laporan dibaca dari `kode/hasil/*.json` dan `kode/reports/summary_metrics.json`.
- **Pemahaman:** setiap anggota mempelajari dan menjelaskan ulang bagian anggota lain untuk persiapan ujian lisan.

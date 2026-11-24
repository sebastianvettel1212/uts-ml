# Deklarasi Kontribusi Anggota Kelompok & Penggunaan Alat AI
**Ujian Tengah Semester (UTS) — Machine Learning**  
**Repositori Asal (Upstream):** `https://github.com/xcoook1es/uts-ml`  
**Repositori Fork:** `https://github.com/sebastianvettel1212/uts-ml`  

---

## 1. Identitas Kelompok & Pembagian Peran

| No | Nama Anggota | NIM | Peran Utama & Tugas Spesifik | Porsi Kontribusi (%) | Status / Konfirmasi |
|:---:|:---|:---:|:---|:---:|:---:|
| 1 | [Nama Anggota 1] | [NIM 1] | **Task 1: Regresi**<br>• Eksplorasi fitur & imputasi data logistik pengiriman paket<br>• Eksperimen pemodelan baseline hingga Gradient Boosting<br>• Optimasi metrik evaluasi (RMSE/MAE) & analisis residual error | 33.3% | Disetujui |
| 2 | Zacky Yusuf Hakim | [NIM 2] | **Task 2: Klasifikasi**<br>• Desain & implementasi 12 tahap operasional pipeline deteksi fraud<br>• Mitigasi right-censoring label (embargo 21 hari) & eliminasi kebocoran data<br>• Kalibrasi Isotonic, penyesuaian threshold biaya bisnis Rp 25.000, & ensemble model | 33.3% | Disetujui |
| 3 | [Nama Anggota 3] | [NIM 3] | **Task 3: Klastering**<br>• Audit integritas data, seleksi fitur multivariat, & transformasi CLR<br>• Komparasi algoritma (K-Means, Agglomerative, GMM, HDBSCAN)<br>• Evaluasi metrik Silhouette Score, stabilitas ARI, & profiling segmentasi pelanggan | 33.3% | Disetujui |

> **Catatan:** Seluruh anggota kelompok berpartisipasi aktif dalam diskusi desain arsitektur, verifikasi silang (*cross-validation*) hasil reproduksi kode, serta penyusunan dokumen laporan komprehensif (`laporan.pdf`).

---

## 2. Tautan Pull Request (PR) & Jejak Kolaborasi Git

Seluruh pengerjaan modul tugas dikelola melalui alur kerja kolaboratif Git yang rapi dengan komit terstruktur:

| Modul Tugas | Cabang Pengerjaan (*Branch*) | Tautan Pull Request (*PR Link*) | Status | Deskripsi Singkat Pekerjaan |
|:---|:---|:---|:---:|:---|
| **Task 1: Regresi** | `feature/task1-regression` | `https://github.com/xcoook1es/uts-ml/pull/[ID_PR_1]` | Disatukan (*Merged*) | Implementasi model prediksi durasi pengiriman paket logistik |
| **Task 2: Klasifikasi** | `feature/task2-classification` | `https://github.com/xcoook1es/uts-ml/pull/[ID_PR_2]` | Disatukan (*Merged*) | Implementasi end-to-end deteksi fraud 12 tahap & `prediksi_klasifikasi.csv` |
| **Task 3: Klastering** | `feature/task3-clustering` | `https://github.com/xcoook1es/uts-ml/pull/[ID_PR_3]` | Disatukan (*Merged*) | Segmentasi pelanggan RFM+Kategori, audit seleksi fitur, & komparasi metode |
| **Integrasi & Repositori** | `main` | `https://github.com/sebastianvettel1212/uts-ml` | Aktif | Penataan struktur standar submisi UTS, dokumentasi reproduksi, & laporan akhir |

---

## 3. Deklarasi Penggunaan Alat AI (*Artificial Intelligence*)

Sesuai dengan ketentuan integritas akademik dalam pengerjaan UTS Machine Learning, kelompok kami mendeklarasikan pemanfaatan alat bantu AI dengan rincian sebagai berikut:

### a. Alat yang Digunakan
* **Platform / Tool**: Google Antigravity (Agentic AI Assistant berbasis Large Language Model).
* **Fungsi / Kapabilitas**: Analisis kode, eksplorasi strategi pemodelan data, *refactoring* skrip Python, serta otomasi penataan visualisasi publikasi (300 DPI).

### b. Ruang Lingkup Pemanfaatan AI
Alat AI dimanfaatkan secara terarah dan bertanggung jawab untuk:
1. **Perancangan Solusi & Hipotesis**: Membantu memetakan risiko pemodelan seperti fenomena *right-censoring*, potensi *target leakage*, serta pergeseran distribusi (*covariate drift*) pada data transaksi temporal.
2. **Otomasi Pembuatan Skrip Reproduksi**: Mengonversi alur eksplorasi interaktif pada Jupyter Notebook ke dalam bentuk skrip otomasi pipeline yang modular (`run_pipeline.py`, `generate_all_12_figures.py`, dan `jalankan_tahap_0_c.py`).
3. **Penyelarasan Standar Visualisasi**: Membantu standardisasi format grafik `matplotlib` (resolusi 300 DPI, label persentase, callout box, anotasi sumbu ganda) agar memenuhi standar pelaporan akademik.

### c. Metodologi Verifikasi & Validasi Hasil AI
Seluruh kode, logika analitik, dan hasil komputasi yang dihasilkan dengan bantuan AI telah melalui proses audit dan pengujian mandiri oleh anggota tim:
* **Audit Reproduksibilitas Deterministik**: Penguncian `SEED = 42` secara konsisten pada semua library numerik (`numpy`, `random`, `scikit-learn`, `lightgbm`, `xgboost`, `catboost`), memastikan setiap angka metrik pada laporan dapat direproduksi secara identik 100%.
* **Audit Pencegahan Kebocoran Data (*Data Leakage*)**: Memastikan variabel terlarang `flag_review_manual` dibuang sejak awal, dan seluruh rekayasa fitur temporal (*rolling velocity*, *expanding target encoding*) hanya menggunakan informasi kausal masa lalu.
* **Validasi Batasan Bisnis Riil**: Memverifikasi secara komputasional bahwa tidak ada transaksi bernilai mikro ($\le 	ext{Rp } 25.000$) yang ditandai untuk review manual (100% patuh guardrail efisiensi operasional).
* **Sanity Check File Submisi**: Memvalidasi integritas baris `prediksi_klasifikasi.csv` tepat sejumlah 106.674 baris tanpa nilai `NaN` atau `null`.

Kelompok kami bertanggung jawab penuh atas seluruh metodologi, interpretasi hasil, dan integritas akademik dari solusi yang dikumpulkan.

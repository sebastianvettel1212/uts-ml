# UTS Machine Learning — Kelompok

Repositori pengerjaan Ujian Tengah Semester (UTS) mata kuliah Machine Learning.  
Repositori ini merupakan fork dari `https://github.com/xcoook1es/uts-ml`.

## Komponen Submisi Resmi (Submission Components)

Sesuai aturan pengumpulan tugas UTS Machine Learning, struktur utama repositori tersusun sebagai berikut:

1. **`laporan.pdf`** *(Draft/Final)* — Dokumen laporan komprehensif gabungan Task 1 (Regresi), Task 2 (Klasifikasi), dan Task 3 (Klastering).
2. **`prediksi_klasifikasi.csv`** — File hasil prediksi resmi Task 2 Deteksi Fraud Transaksi (106.674 baris, format `id_transaksi,prob,tandai`).
3. **`kontribusi.md`** — Matriks pembagian peran anggota kelompok, tautan Pull Request (PR), dan deklarasi penggunaan alat AI beserta metodologi verifikasinya.
4. **`kode/`** — Seluruh kode program, notebook interaktif, rekapitulasi metrik, dan skrip otomasi yang mereproduksi seluruh angka laporan dan file prediksi secara deterministik (`SEED = 42`).
   * Panduan eksekusi dan dependensi lingkungan terdapat pada [`kode/README.md`](kode/README.md).

---

## Ringkasan Perintah Reproduksi Cepat

Untuk menjalankan seluruh pipeline dan mereproduksi file submisi dari direktori root:

```bash
# Task 2: Deteksi Fraud Transaksi (menghasilkan prediksi_klasifikasi.csv di root)
python kode/run_pipeline.py

# Task 3: Segmentasi Pelanggan E-Commerce (menghasilkan tugas3.json di kode/hasil/)
python kode/jalankan_tahap_0_c.py
```

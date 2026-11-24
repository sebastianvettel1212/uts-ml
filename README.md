# UTS Machine Learning: Regresi, Klasifikasi, dan Clustering

Tugas kelompok UTS Machine Learning. Anggota: Zidan Alghifari (101032330177) dan Zacky Yusuf Hakim (101032300183).

## Isi branch `main`

| Item | Keterangan |
|---|---|
| `laporan.pdf` | Laporan Task 1–3 |
| `kode/` | Semua kode, dengan panduan di [`kode/README.md`](kode/README.md) |
| `prediksi_regresi.csv` | Task 1: `id_paket,p50,p90` (60.000 baris, p90 ≥ p50) |
| `prediksi_klasifikasi.csv` | Task 2: `id_transaksi,prob,tandai` (106.674 baris) |
| `cluster.csv` | Task 3: `id_pelanggan,cluster` (120.000 baris, -1 = anomali) |
| `kontribusi.md` | Peran anggota, tautan pull request, dan penggunaan AI |

## Satu perintah

Dari akar repositori, setelah data diekstrak ke `data/` dan pustaka dipasang (lihat `kode/README.md`):

```bash
python kode/jalankan.py
```

Perintah ini menjalankan ketiga task secara berurutan, menulis ulang ketiga file prediksi, semua angka di `kode/hasil/` dan `kode/reports/`, lalu membuat ulang `laporan.pdf`.

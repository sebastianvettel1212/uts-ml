# Rencana Kerja & Panduan Eksekusi Task 3: Clustering (Segmentasi Pelanggan)

**Mata Kuliah:** Machine Learning (UTS — Tugas Kelompok)

**Dataset:** `clustering.csv` / `clustering.csv.gz` ($\approx 120.000$ baris pelanggan, ringkasan 90 hari terakhir, *unlabeled*)

**Metrik Evaluasi Utama:** Adjusted Rand Index (ARI) terhadap segmen referensi dosen (label `-1` untuk anomali)

**Target Output:** `cluster.csv` (`id_pelanggan,cluster`), `hasil/tugas3.json` (rekapitulasi angka terkunci *seed*), bagian Task 3 di `laporan.pdf` (4–5 halaman), dan skrip reproduksibilitas di folder `kode/`.

## 1. Kamus Data & Struktur Dataset (`clustering.csv.gz`)

Satu baris merepresentasikan satu pelanggan (`id_pelanggan`) dengan ringkasan aktivitas 90 hari terakhir tanpa label segmen.

| Kolom | Keterangan | Kelompok Perlakuan |
| :--- | :--- | :--- |
| `id_pelanggan` | ID unik pelanggan | Identitas (Index/Output) |
| `recency_hari` | Jumlah hari sejak transaksi terakhir (*recency*) | RFM & Nilai Uang |
| `frekuensi_90h` | Frekuensi transaksi dalam 90 hari terakhir | RFM & Nilai Uang |
| `belanja_90h` | Total nominal belanja dalam 90 hari terakhir (rupiah) | RFM & Nilai Uang |
| `rata2_keranjang` | Nilai keranjang rata-rata per transaksi (rupiah) | RFM & Nilai Uang |
| `item_per_pesanan` | Rata-rata jumlah barang per pesanan | RFM & Nilai Uang |
| `porsi_<kategori>` | Porsi belanja per kategori (16 kolom, total jumlah per baris = $1$) | Data Komposisional (Kategori) |
| `porsi_jam_pagi`, `porsi_jam_siang`, `porsi_jam_sore`, `porsi_jam_malam` | Porsi waktu belanja berdasarkan pembagian waktu (4 kolom, total jumlah per baris = $1$) | Data Komposisional (Waktu) |
| `persen_voucher`, `persen_flash_sale`, `persen_paylater`, `persen_cod` | Porsi pesanan yang menggunakan voucher, flash sale, paylater, dan COD | Persentase Perilaku |
| `retur_90h`, `rating_beri_rata2`, `ulasan_90h` | Jumlah retur 90 hari, rata-rata rating yang diberikan, dan jumlah ulasan | Aktivitas & Interaksi |
| `sesi_per_minggu`, `durasi_sesi_menit`, `rasio_konversi` | Frekuensi sesi aplikasi, durasi rata-rata sesi (menit), dan rasio konversi | Aktivitas Aplikasi |
| `lama_member_bulan`, `umur` | Lama bergabung sebagai member (bulan) dan usia pelanggan | Profil Pelanggan (Numerik) |
| `kota_tier`, `perangkat` | Tier kota domisili dan jenis perangkat yang digunakan | Profil Pelanggan (Kategorikal) |
| `jumlah_alamat`, `wishlist`, `klik_iklan_30h` | Jumlah alamat tersimpan, jumlah barang di wishlist, dan klik iklan 30 hari | Aktivitas & Interaksi |
| `belanja_30h`, `frekuensi_30h` | Total belanja dan frekuensi transaksi dalam 30 hari terakhir | Redundan / Jangka Pendek |
| `n01`–`n10` | Atribut tambahan dari sistem CRM (10 kolom anonim) | Fitur Anonim CRM |

## 2. Peta Alur Keseluruhan Task 3 (Gambaran Besar)

Sebelum mulai menulis kode di Jupyter Notebook, berikut adalah gambaran utuh alur berpikir dan tahapan kerja kita dari data mentah hingga siap dikumpulkan dan diujikan saat ujian lisan (*oral exam*):

```
[Tahap 0: Audit & Pemetaan Kolom] (Bersama - 30 Menit)
       │
       ▼
[Tahap a: Pipeline Praproses] ──► Tangani missing value, skewness (log), komposisi (CLR/drop-1), kategori, & outlier
       │
       ▼
[Tahap b: Seleksi Fitur] ───────► Buang fitur redundan (30h vs 90h) & uji fitur anonim CRM (n01–n10) agar ARI optimal
       │
       ▼
[Tahap c: Komparasi 3+ Metode] ─► Bandingkan K-Means, GMM, dan HDBSCAN/Agglomerative (pada subsampel & penuh)
       │
       ▼
[Tahap d: Pemilihan Jumlah k] ──► Evaluasi k = 2..10 menggunakan Silhouette, BIC, Inertia, & Stabilitas
       │
       ▼
[Tahap e: Uji Stabilitas] ──────► Ulangi clustering (10–20x bootstrap/subsampel) dan ukur konsistensi ARI
       │
       ▼
[Tahap f: Profiling Segmen] ────► Hitung median fitur asli per cluster, beri nama segmen & usulkan aksi marketing
       │
       ▼
[Tahap g: Deteksi Anomali (-1)] ► Tetapkan ambang ketat (1–3% data ekstrem) agar tidak menurunkan skor ARI
       │
       ▼
[Tahap Akhir: Validasi & Laporan] ► Ekspor cluster.csv, kunci hasil/tugas3.json, & susun narasi laporan.pdf
```

### Strategi Pembagian Kerja Paralel (2 Orang Anggota) & Alur GitHub

Mengingat penilaian mempertimbangkan kontribusi GitHub (*fork* $\rightarrow$ *branch/commit* $\rightarrow$ *Pull Request*) dan setiap anggota wajib memahami seluruh bagian saat ujian lisan 20 menit:

* **Tahap 0 (Bersama):** Duduk bersama (atau diskusi daring) selama 30 menit untuk menyepakati pemetaan kolom berdasarkan tabel kamus data di atas dan daftar awal fitur.
* **Tahap a & Tahap b (Paralel):**
  * **Anggota 1:** Fokus membangun fungsi praproses untuk kelompok RFM, perilaku, aktivitas, missing value, dan outlier, serta menguji korelasi/redundansi fitur (`belanja_30h` vs `belanja_90h`, `frekuensi_30h` vs `frekuensi_90h`).
  * **Anggota 2:** Fokus menangani transformasi data komposisional (`porsi_<kategori>` dan `porsi_jam_*`), encoding kategorikal (`kota_tier`, `perangkat`), serta melakukan eksperimen ablasi/uji statistik untuk menyaring atribut CRM anonim `n01`–`n10`.
  * *Sinkronisasi 1:* Gabungkan hasil Tahap a dan b melalui Pull Request menjadi satu fungsi praproses & seleksi fitur final (`X_selected`).
* **Tahap c (Paralel):**
  * **Anggota 1:** Menjalankan dan mengevaluasi baseline **K-Means** serta **Gaussian Mixture Model (GMM)** beserta pencatatan waktu dan metriknya.
  * **Anggota 2:** Menjalankan metode berbasis kepadatan/hierarki (**HDBSCAN** atau **Agglomerative Clustering**) pada subsampel serta membangun mekanisme pemetaan ke sisa 120.000 data, lalu menyusun tabel perbandingan ketiga metode.

## 3. Panduan & Perintah Pengerjaan (Tahap 0 hingga Tahap c)

Setiap bagian di bawah ini dirancang agar bisa langsung dijadikan struktur sel di Jupyter Notebook: **Narasi Alur Pikir (Sel Markdown)** diikuti oleh **Perintah Eksekusi (Sel Kode)**.

### Tahap 0: Kenali Data (Data Auditing & Pemetaan Kolom)

#### Alur Pikir Mahasiswa (Untuk Narasi Sel Markdown)

> *"Sebelum terburu-buru mencoba berbagai algoritma clustering, langkah pertama yang harus kita lakukan adalah membedah isi dataset `clustering.csv` / `clustering.csv.gz` dan mencocokkannya dengan definisi resmi di `kamus_data.md`. Kita berhadapan dengan sekitar 120.000 baris ringkasan perilaku pelanggan selama 90 hari terakhir tanpa label jawaban. Jika ada `id_pelanggan` yang ganda, nilai kosong yang tidak wajar, atau kesalahan pemahaman satuan fitur, seluruh perhitungan jarak di tahap berikutnya akan melenceng. Selain itu, dataset ini punya berbagai jenis variabel—mulai dari nominal uang jutaan rupiah (`belanja_90h`) hingga porsi kategori (`porsi_<kategori>`) dan porsi jam (`porsi_jam_*`) yang masing-masing jumlahnya selalu 1—sehingga kita perlu mengelompokkan kolom sejak awal agar perlakuannya adil dan masuk akal."*

#### Perintah Pengerjaan (Untuk Eksekusi Sel Kode)

1. **Inisialisasi & Reproduksibilitas:**
   * Impor library standar (`numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`, `sklearn`, `json`, `time`, `os`).
   * Tetapkan konstanta `SEED = 42` di bagian paling atas notebook dan cetak versi Python serta versi library utama sebagai syarat reproduksibilitas laporan.
2. **Audit Integritas Dataset:**
   * Muat dataset (`pd.read_csv('clustering.csv')` atau `'clustering.csv.gz'`) dan tampilkan dimensi dataset (`jumlah_baris`, `jumlah_kolom`).
   * Periksa apakah kolom `id_pelanggan` memiliki nilai kosong atau duplikat. Pastikan jumlah ID unik sama persis dengan jumlah baris.
   * Tampilkan tipe data setiap kolom dan hitung jumlah serta persentase *missing value* per kolom (urutkan dari yang paling banyak kosong).
3. **Klasifikasi & Validasi Kelompok Kolom (Sesuai `kamus_data.md`):**
   * Kelompokkan seluruh nama kolom (selain `id_pelanggan`) ke dalam kelompok berikut dalam bentuk struktur dictionary Python:
     1. **RFM & Nilai Uang:** `['recency_hari', 'frekuensi_90h', 'belanja_90h', 'rata2_keranjang', 'item_per_pesanan']`.
     2. **Komposisi Kategori (16 kolom):** Semua kolom berawalan `porsi_` kecuali 4 kolom waktu. Verifikasi secara matematis apakah jumlah tiap barisnya bernilai $\approx 1.0$.
     3. **Komposisi Waktu (4 kolom):** `['porsi_jam_pagi', 'porsi_jam_siang', 'porsi_jam_sore', 'porsi_jam_malam']`. Verifikasi apakah jumlah tiap barisnya bernilai $\approx 1.0$.
     4. **Persentase Perilaku:** `['persen_voucher', 'persen_flash_sale', 'persen_paylater', 'persen_cod']` (cek rentang nilai min–maks, apakah berada pada interval $[0, 1]$).
     5. **Aktivitas & Interaksi:** `['retur_90h', 'rating_beri_rata2', 'ulasan_90h', 'sesi_per_minggu', 'durasi_sesi_menit', 'rasio_konversi', 'jumlah_alamat', 'wishlist', 'klik_iklan_30h']`.
     6. **Profil Pelanggan:** Numerik `['lama_member_bulan', 'umur']` serta kategorikal `['kota_tier', 'perangkat']` (cek kategori unik dan proporsi kemunculannya).
     7. **Redundan (Jangka Pendek 30 Hari):** `['belanja_30h', 'frekuensi_30h']`.
     8. **Atribut CRM Anonim:** `['n01', 'n02', ..., 'n10']` (cek statistik deskriptif: mean, std, min, median, max, dan skewness).
4. **Output Kesepakatan Tahap 0:**
   * Cetak ringkasan statistik deskriptif per kelompok kolom dan simpan hasil audit dasar (jumlah baris, daftar kolom dengan missing value, dan verifikasi jumlah komposisi) ke variabel dictionary untuk dicatat di `hasil/tugas3.json`.

### Tahap a: Praproses Data (Preprocessing Pipeline)

#### Alur Pikir Mahasiswa (Untuk Narasi Sel Markdown)

> *"Setelah melihat karakteristik tiap kelompok kolom dari kamus data, kita menemukan empat tantangan utama yang wajib dibereskan sebelum menghitung jarak antar pelanggan. Pertama, adanya nilai kosong (missing value) yang harus dicek apakah terjadi secara acak atau memang menandakan perilaku tertentu (misalnya `rating_beri_rata2` kosong karena pelanggan belum pernah memberi rating). Kedua, fitur nominal uang dan frekuensi seperti `belanja_90h`, `frekuensi_90h`, dan `rata2_keranjang` sangat miring ke kanan (right-skewed)—ada pelanggan yang belanja puluhan juta sementara mayoritas ratusan ribu—sehingga perlu transformasi logaritmik agar distribusinya lebih simetris. Ketiga, 16 kolom `porsi_<kategori>` dan 4 kolom `porsi_jam_*` bersifat komposisional (jumlahnya selalu 1), yang artinya tidak saling bebas secara linear; kita perlu menerapkan transformasi Centered Log-Ratio (CLR) atau membuang satu kolom referensi agar tidak terjadi multikolinearitas. Terakhir, karena perbedaan skala antar fitur dan adanya segelintir pelanggan ekstrem, kita perlu melakukan clipping outlier ringan dan penskalaan (scaling) agar pusat cluster tidak tertarik oleh segelintir titik ekstrem."*

#### Perintah Pengerjaan (Untuk Eksekusi Sel Kode)

1. **Audit & Imputasi Missing Value:**
   * Periksa pola *missing value* pada setiap kolom (terutama `rating_beri_rata2`, atribut CRM `n01`–`n10`, atau profil pelanggan) untuk melihat apakah missing terjadi secara acak atau terkait aktivitas tertentu.
   * Lakukan imputasi menggunakan **median** untuk fitur numerik (karena tahan terhadap outlier) dan **modus** untuk fitur kategorikal (`kota_tier`, `perangkat`) jika terdapat nilai kosong. Catat daftar kolom yang diimputasi beserta nilai mediannya.
2. **Transformasi Variabel Miring (*Skewed*):**
   * Hitung nilai *skewness* sebelum transformasi untuk kelompok RFM (`recency_hari`, `frekuensi_90h`, `belanja_90h`, `rata2_keranjang`, `item_per_pesanan`), kelompok redundan (`belanja_30h`, `frekuensi_30h`), serta fitur aktivitas (`retur_90h`, `ulasan_90h`, `sesi_per_minggu`, `durasi_sesi_menit`, `wishlist`, `klik_iklan_30h`).
   * Terapkan transformasi $\log(1 + x)$ menggunakan `np.log1p` pada fitur non-negatif dengan skewness tinggi (misalnya $\vert{}\text{skewness}\vert{} > 1.0$).
   * Buat visualisasi berdampingan (histogram sebelum vs sesudah transformasi) untuk fitur representatif (`belanja_90h` dan `frekuensi_90h`) serta tabel perbandingan nilai skewness sebelum dan sesudah transformasi sebagai bukti di laporan.
3. **Transformasi Data Komposisional:**
   * Tentukan dan jalankan salah satu metode berikut secara terpisah untuk blok `porsi_<kategori>` ($D=16$ kolom) dan blok `porsi_jam_*` ($D=4$ kolom):
     * **Pilihan Utama — Centered Log-Ratio (CLR):** Tambahkan konstanta kecil $\epsilon = 10^{-5}$ pada setiap elemen untuk menghindari $\log(0)$, lalu terapkan rumus:
       $$
       \text{CLR}(x_i) = \ln(x_i + \epsilon) - \frac{1}{D} \sum_{j=1}^{D} \ln(x_j + \epsilon)
       $$
       di mana $D=16$ untuk kelompok kategori dan $D=4$ untuk kelompok jam.
     * **Pilihan Alternatif — Drop-One Reference:** Pertahankan nilai proporsi asli, tetapi buang satu kolom dengan proporsi rata-rata terkecil dari masing-masing kelompok (`porsi_<kategori>` dan `porsi_jam_*`).
   * Tuliskan alasan pemilihan metode tersebut secara eksplisit di sel markdown (misalnya: CLR mempertahankan simetri antar-komponen dan memetakan data dari ruang simpleks ke ruang Euclidean standar yang cocok untuk K-Means dan GMM).
4. **Encoding Variabel Kategorikal (`kota_tier`, `perangkat`):**
   * Lakukan *One-Hot Encoding* (`pd.get_dummies` atau `OneHotEncoder`) pada kolom `kota_tier` dan `perangkat`. Pastikan seluruh kolom hasil encoding bertipe numerik (`float64`).
5. **Penanganan Outlier & Standardisasi Skala:**
   * Lakukan *clipping* (winsorizing) pada batas persentil bawah dan atas yang konservatif (misalnya persentil ke-$0.5$ dan ke-$99.5$) pada fitur numerik kontinu agar outlier ekstrem tidak merusak perhitungan rata-rata/varians saat scaling, namun tetap tandai pelanggan yang melewati batas persentil ekstrem sebagai catatan kandidat awal untuk Tahap g (anomali).
   * Terapkan `StandardScaler` sehingga seluruh fitur memiliki mean $\approx 0$ dan standar deviasi $\approx 1$.
6. **Pembungkusan Fungsi Pipeline:**
   * Gabungkan seluruh langkah di atas menjadi satu fungsi bersih dan deterministik, misalnya `jalankan_praproses(df_mentah)`, yang mengembalikan matriks fitur siap pakai `X_preprocessed` serta daftar nama kolomnya (pastikan tidak ada nilai `NaN` atau `inf`).

### Tahap b: Seleksi Fitur (Feature Selection)

#### Alur Pikir Mahasiswa (Untuk Narasi Sel Markdown)

> *"Ini adalah tahap yang paling menentukan akurasi segmentasi (ARI) kita. Dalam dataset sintetis sebesar 120.000 baris ini, terdapat 10 atribut tambahan dari sistem CRM (`n01` sampai `n10`) yang bersifat anonim—sangat besar kemungkinannya sebagian dari kolom tersebut hanyalah noise acak yang disisipkan untuk menguji ketelitian kita. Selain itu, ada pasangan fitur jangka pendek dan jangka panjang yang saling tumpang tindih, yaitu `belanja_30h` dengan `belanja_90h` serta `frekuensi_30h` dengan `frekuensi_90h`. Algoritma clustering tanpa pengawasan tidak bisa membedakan mana fitur sinyal dan mana fitur pengecoh; semuanya dihitung sama rata dalam rumus jarak. Oleh karena itu, kita harus membuktikannya lewat uji korelasi, uji ANOVA F-test, dan eksperimen ablasi Silhouette/stabilitas."*

#### Perintah Pengerjaan (Untuk Eksekusi Sel Kode)

1. **Penyaringan Varians Rendah & Redundansi Korelasi (30 Hari vs 90 Hari):**
   * Periksa standar deviasi/varians tiap fitur sebelum standardisasi; tandai fitur yang hampir konstan (misalnya $> 99\%$ baris bernilai sama).
   * Hitung matriks korelasi Pearson dan Spearman antar seluruh fitur numerik, khususnya mengevaluasi pasangan `belanja_30h` vs `belanja_90h` dan `frekuensi_30h` vs `frekuensi_90h`.
   * Jika korelasinya sangat tinggi ($\vert{}r\vert{} > 0.80$), buang versi 30 hari (karena ringkasan 90 hari lebih stabil dan komprehensif) atau uji fitur turunan rasio aktivitas terkini ($\text{belanja\_30h} / (\text{belanja\_90h} + 1)$) dan pertahankan hanya jika terbukti memberi informasi baru yang tidak berkorelasi tinggi.
2. **Uji Signifikansi Statistik (ANOVA F-Test) pada Atribut CRM (`n01`–`n10`) & Kategorikal:**
   * Bangun **clustering referensi sementara** menggunakan K-Means ($k = 4, 5, 6$) pada subsampel ($15.000$ baris) dengan hanya memakai fitur perilaku inti yang jelas maknanya (RFM 90 hari, komposisi kategori & jam, persentase perilaku, aktivitas aplikasi).
   * Dengan menggunakan label cluster sementara tersebut sebagai target kelompok, hitung nilai statistik **ANOVA F-value** dan $p$**-value** (`sklearn.feature_selection.f_classif`) untuk setiap fitur, khususnya `n01`–`n10` serta hasil encoding `kota_tier` dan `perangkat`.
   * Urutkan seluruh fitur berdasarkan nilai F-statistic. Atribut CRM (`n01`–`n10`) yang memiliki F-statistic sangat kecil ($p\text{-value} > 0.05$ atau jauh di bawah rata-rata fitur perilaku inti) terindikasi kuat sebagai *noise* acak yang seragam di semua segmen.
3. **Eksperimen Ablasi (Dengan vs Tanpa Fitur Meragukan):**
   * Lakukan pengujian langsung pada subsampel tetap ($15.000$ baris, `SEED = 42`):
     * Bandingkan skor **Silhouette** dan **Stabilitas ARI** (antar 5 kali inisialisasi acak) antara:
       1. *Set Fitur Inti saja* (tanpa `n01`–`n10`).
       2. *Set Fitur Inti + masing-masing fitur `n01`–`n10` yang lolos uji F-test*.
       3. *Set Fitur Inti + seluruh `n01`–`n10`*.
       4. *Set Fitur dengan vs tanpa `kota_tier` dan `perangkat`*.
   * Pertahankan fitur anonim atau kategorikal **hanya jika** fitur tersebut terbukti berbeda nyata antar-segmen (F-statistic tinggi) dan meningkatkan atau menjaga kualitas kohesi/stabilitas cluster.
4. **Evaluasi Penyeimbangan Bobot Blok Fitur (Opsional Eksploratif):**
   * Periksa apakah blok `porsi_<kategori>` (16 kolom CLR) mendominasi total jarak Euclidean dibandingkan blok RFM (5 kolom). Uji apakah pembobotan berbasis blok meningkatkan skor Silhouette dan kejelasan pemisahan segmen.
5. **Pembuatan Tabel Keputusan Fitur (Wajib untuk Laporan):**
   * Susun sebuah DataFrame/tabel ringkasan yang memuat seluruh kolom dari `kamus_data.md` beserta kolom: `Nama Fitur`, `Kelompok`, `Status Akhir (Dipakai / Diberi Bobot / Dibuang)`, `Metrik Bukti (Korelasi / F-stat / Delta Silhouette)`, dan `Alasan Teknis`. Simpan tabel ini dan daftar final kolom terpilih (`fitur_terpilih`) untuk digunakan di Tahap c.

### Tahap c: Komparasi Minimal 3 Metode Clustering

#### Alur Pikir Mahasiswa (Untuk Narasi Sel Markdown)

> *"Setelah mendapatkan matriks fitur yang bersih dan bebas noise dari Tahap b, kita tidak bisa langsung mengasumsikan bahwa satu algoritma saja sudah paling benar. Setiap algoritma clustering punya asumsi bentuk kelompok yang berbeda: K-Means mengasumsikan cluster berbentuk bulat (spherical) dengan ukuran yang relatif seimbang; Gaussian Mixture Model (GMM) lebih fleksibel karena mengizinkan cluster berbentuk elips dengan kovariansi berbeda serta menghasilkan probabilitas keanggotaan; sedangkan HDBSCAN atau Agglomerative Clustering mampu menangkap struktur kepadatan atau hierarki tanpa memaksa setiap titik masuk ke cluster secara kaku. Karena metode hierarki dan kepadatan memakan memori dan waktu yang sangat besar jika dijalankan langsung pada 120.000 baris, kita akan menerapkan strategi subsampling terukur (15.000–20.000 baris) yang representatif, lalu memetakan sisa data ke cluster terdekat agar perbandingan ketiga metode berlangsung adil pada keseluruhan dataset."*

#### Perintah Pengerjaan (Untuk Eksekusi Sel Kode)

1. **Persiapan Data Evaluasi & Subsampel Representatif:**
   * Gunakan matriks fitur hasil seleksi Tahap b (`X_selected`, $\approx 120.000 \times p$).
   * Buat subsampel terstratifikasi (misalnya berdasarkan kuantil `belanja_90h` atau `recency_hari`) sebanyak $N_{\text{sub}} = 15.000$ atau $20.000$ baris dengan `SEED = 42`. Subsampel ini digunakan untuk melatih metode yang berat secara komputasi sekaligus menjadi indeks evaluasi yang konsisten untuk menghitung **Silhouette Score** seluruh metode agar *apple-to-apple*.
   * Tetapkan jumlah cluster pembanding awal yang seragam (misalnya $k = 5$) untuk metode partisional/distribusi (K-Means, GMM, Agglomerative) dan parameter kepadatan yang setara untuk HDBSCAN, sebelum mencari $k$ paling optimal di Tahap d.
2. **Metode 1 — K-Means (Baseline Partisional):**
   * Latih `KMeans(n_clusters=k, n_init=20, max_iter=300, random_state=SEED)` pada seluruh $120.000$ baris.
   * Catat waktu komputasi (`time.perf_counter()`).
   * Hitung Silhouette Score (pada indeks subsampel $N_{\text{sub}}$), stabilitas awal (ARI antar 3 kali run dengan seed berbeda), serta distribusi jumlah dan persentase anggota tiap cluster.
3. **Metode 2 — Gaussian Mixture Model / GMM (Berbasis Distribusi Probabilistik):**
   * Latih `GaussianMixture(n_components=k, covariance_type='full', n_init=5, random_state=SEED)` pada seluruh $120.000$ baris (atau latih pada subsampel $20.000$ baris jika komputasi lambat, lalu panggil `.predict(X_selected)` dan `.score_samples(X_selected)` untuk seluruh data).
   * Catat waktu komputasi, nilai **BIC**, **AIC**, Silhouette Score (pada indeks subsampel $N_{\text{sub}}$), stabilitas awal, dan distribusi ukuran tiap cluster.
   * Simpan distribusi nilai *log-likelihood* (`score_samples`) sebagai catatan potensi untuk deteksi anomali di Tahap g.
4. **Metode 3 — HDBSCAN atau Agglomerative Clustering (Berbasis Kepadatan / Hierarki):**
   * Jalankan pada subsampel $N_{\text{sub}} = 15.000\text{--}20.000$ baris dan catat waktu komputasi pelatihan + pemetaan ke sisa data:
     * **Jika menggunakan Agglomerative Clustering:** Jalankan `AgglomerativeClustering(n_clusters=k, linkage='ward')` pada subsampel. Hitung centroid tiap cluster hasil hierarki pada subsampel, lalu gunakan `NearestCentroid` (atau `KNeighborsClassifier(n_neighbors=5)`) untuk memprediksi label seluruh $120.000$ baris.
     * **Jika menggunakan HDBSCAN:** Jalankan `HDBSCAN(min_cluster_size=300, min_samples=30, prediction_data=True)` (sesuaikan parameter agar menghasilkan jumlah cluster yang masuk akal, misal $4\text{--}8$ cluster, dan proporsi noise `-1` tidak berlebihan). Gunakan fungsi `approximate_predict` untuk melabeli seluruh $120.000$ baris.
   * Hitung Silhouette Score (pada titik non-noise di indeks subsampel $N_{\text{sub}}$), stabilitas awal (pada 3 subsampel berbeda), serta distribusi ukuran tiap cluster (termasuk persentase label `-1` jika memakai HDBSCAN).
5. **Kompilasi Tabel Perbandingan Metode & Keputusan Awal:**
   * Susun tabel perbandingan yang memuat baris untuk setiap metode (`K-Means`, `GMM`, `Agglomerative / HDBSCAN`) dan kolom berikut:
     1. `Metode`
     2. `Jumlah Cluster Terbentuk`
     3. `Silhouette Score (Subsampel)`
     4. `Stabilitas Awal (Mean ARI)`
     5. `Distribusi Ukuran Cluster (Min% - Max% / Rincian)`
     6. `Waktu Komputasi (Detik)`
   * Simpan seluruh angka hasil perbandingan ini ke `hasil/tugas3.json`.
   * Tuliskan kesimpulan analisis di sel markdown: jelaskan mengapa satu metode (atau dua metode teratas) lebih unggul secara struktur segmen, keseimbangan ukuran cluster, maupun efisiensi komputasi untuk dilanjutkan ke tahap penentuan jumlah cluster $k$ optimal (Tahap d) dan uji stabilitas mendalam (Tahap e).
# Catatan Alur Pikir & Panduan Ujian Lisan (Oral Exam)
## Task 3: Clustering (Segmentasi Pelanggan E-Commerce) — Tahap 0 Hingga Tahap c

Dokumen ini memuat **alur pikir konseptual (mental model)**, landasan teoritis, dan panduan menjawab pertanyaan saat **ujian lisan (20 menit)** di hadapan dosen penguji, sengaja dipisahkan dari notebook utama (`task_3_clustering_tahap_0_tahap_c.ipynb`) agar kode tetap bersih, profesional, dan fokus pada eksekusi teknis.

---

## 1. Tahap 0: Audit Integritas Data & Pemetaan Kolom

### Alur Pikir & Logika Dasar
Sebelum terburu-buru mencoba berbagai algoritma clustering, langkah pertama yang harus dilakukan adalah membedah isi dataset `clustering.csv` dan mencocokkannya dengan definisi di `kamus_data.md`. 
Kita berhadapan dengan sekitar 120.000 baris ringkasan perilaku pelanggan selama 90 hari terakhir tanpa label jawaban (*unsupervised*). Jika ada `id_pelanggan` yang ganda, nilai kosong yang tidak wajar, atau kesalahan pemahaman satuan fitur, seluruh perhitungan jarak di tahap berikutnya akan bias dan melenceng.
Selain itu, dataset ini memiliki beragam jenis variabel—mulai dari nominal uang jutaan rupiah (`belanja_90h`), frekuensi transaksi, porsi belanja per kategori (`porsi_<kategori>`), porsi jam aktif (`porsi_jam_*`), hingga atribut anonim CRM (`n01`–`n15`). Oleh karena itu, pengelompokan kolom sejak awal mutlak diperlukan agar setiap kelompok variabel mendapat perlakuan matematika yang adil dan tepat.

### Poin Kunci untuk Ujian Lisan:
1. **Mengapa kolom `rata2_keranjang` memiliki 6.736 nilai kosong?**
   * *Jawaban:* Nilai kosong tersebut terjadi secara **struktural (structural missingness)**, bukan *missing completely at random* (MCAR). Seluruh 6.736 pelanggan tersebut memiliki `frekuensi_90h == 0` dan `belanja_90h == 0` (pelanggan pasif yang belum pernah transaksi dalam 90 hari terakhir). Rata-rata keranjang belanja adalah rasio $\frac{\text{belanja\_90h}}{\text{frekuensi\_90h}}$, sehingga ketika frekuensi bernilai 0 terjadi pembagian $\frac{0}{0}$ yang secara matematis *undefined*.
2. **Mengapa ID pelanggan harus diverifikasi keunikannya?**
   * *Jawaban:* Untuk memastikan bahwa satu baris tepat merepresentasikan satu entitas unik pelanggan. Duplikasi ID akan menggandakan bobot titik data di ruang fitur dan mendistorsi penentuan pusat klaster (*cluster centroid*).

---

## 2. Tahap a: Pipeline Praproses Data (Preprocessing Pipeline)

### Alur Pikir & Logika Dasar
Dari audit data, ada empat tantangan metodologis utama:
1. **Missing Value:** Pada `rata2_keranjang`, imputasi dilakukan menggunakan **median** (Rp 183.900,00) sesuai panduan teknis, karena median tahan terhadap pengaruh nilai ekstrem (*outlier robust*) dibanding mean.
2. **Skewness Ekstrem:** Variabel nominal uang dan frekuensi (`belanja_90h`, `frekuensi_90h`, `rata2_keranjang`) memiliki kemiringan ke kanan yang sangat tajam (*right-skewed*, skewness $> 3.5$). Mayoritas pelanggan bertransaksi ratusan ribu rupiah, sementara segelintir pelanggan mencapai puluhan juta. Jika langsung dihitung jarak Euclidean-nya, segelintir data ekstrem ini akan menarik titik pusat klaster (*centroid pulling*). Oleh karena itu, diterapkan transformasi logaritmik $\log(1 + x)$ (`np.log1p`) pada seluruh fitur kontinu non-negatif dengan $|\text{skewness}| > 1.0$.
3. **Karakteristik Data Komposisional (Simpleks):** 16 kolom `porsi_<kategori>` dan 4 kolom `porsi_jam_*` memiliki sifat jumlah baris selalu sama dengan $1.0$ ($\sum x_i = 1$). Dalam statistika, data seperti ini disebut data komposisional (*compositional data*) yang hidup di ruang simpleks $S^D$, bukan di ruang Euclidean bebas $\mathbb{R}^D$. Komponen-komponennya terikat secara linear (*spurious correlation* / multikolinearitas sempurna jika langsung dipakai). Oleh sebab itu, kita menerapkan transformasi **Centered Log-Ratio (CLR)**:
   $$\text{CLR}(x_i) = \ln(x_i + \epsilon) - \frac{1}{D} \sum_{j=1}^{D} \ln(x_j + \epsilon)$$
   dengan $\epsilon = 10^{-5}$ untuk menangani elemen bernilai nol. CLR memetakan data dari simpleks ke ruang Euclidean standar secara simetris tanpa harus membuang satu kategori secara sewenang-wenang.
4. **Outlier Clipping & Scaling:** Untuk mengamankan perhitungan varians dan mean pada `StandardScaler`, diterapkan winsorizing clipping pada persentil konservatif ($0.5\%$ bawah dan $99.5\%$ atas) khusus pada fitur numerik kontinu, tanpa menghapus baris agar data tetap utuh.

### Poin Kunci untuk Ujian Lisan:
1. **Mengapa tidak menggunakan Min-Max Scaler?**
   * *Jawaban:* Min-Max Scaler sangat rentan terhadap outlier karena batas $[0, 1]$ ditentukan oleh nilai ekstrem mutlak ($\min$ dan $\max$). `StandardScaler` (mean=0, std=1) yang dikombinasikan dengan winsorizing jauh lebih stabil untuk algoritma berbasis jarak Euclidean seperti K-Means.
2. **Mengapa transformasi CLR lebih unggul daripada membiarkan proporsi asli $[0, 1]$?**
   * *Jawaban:* Proporsi asli memiliki batas tegas di $0$ dan $1$ serta saling bergantung secara negatif (jika porsi kategori A naik, porsi kategori lain pasti turun). Jarak Euclidean biasa di atas data proporsi melanggar asumsi geometri ruang bebas. CLR membuka batas simpleks dan merefleksikan rasio logaritmik antar-komponen secara adil.

---

## 3. Tahap b: Seleksi Fitur (Feature Selection)

### Alur Pikir & Logika Dasar
Tahap ini paling menentukan keberhasilan metrik evaluasi akhir (Adjusted Rand Index / ARI). Pada algoritma tanpa pengawasan (*unsupervised*), algoritma tidak memiliki informasi label dan akan menganggap semua kolom memiliki derajat kepentingan yang sama dalam rumus jarak:
$$d(x, y) = \sqrt{\sum_{j=1}^p (x_j - y_j)^2}$$
Jika kita memasukkan fitur sampah/noise, jarak antar titik akan didominasi oleh derau acak (*curse of dimensionality*).

Di dataset ini terdapat dua potensi jebakan besar:
1. **Fitur Redundan:** Pasangan 30 hari vs 90 hari (`belanja_30h` vs `belanja_90h`, `frekuensi_30h` vs `frekuensi_90h`). Korelasi Pearson keduanya sangat tinggi ($r = 0.9263 > 0.80$). Memasukkan keduanya sama saja dengan menggandakan bobot belanja dan frekuensi secara tidak adil. Maka fitur 30 hari dibuang demi stabilitas agregasi 90 hari.
2. **Atribut Anonim CRM (`n01`–`n15`):** Secara teoritis atribut ini dicurigai sebagai noise acak sintetis. Kita membuktikannya lewat dua uji empiris:
   * **Uji ANOVA F-Test:** Membangun cluster referensi sementara ($k=5$) dari fitur perilaku inti pada subsampel 15.000 data, lalu mengukur signifikansi setiap fitur kandidat. Hasilnya: fitur `perangkat` menghasilkan nilai $F = 456.8$ ($p < 10^{-200}$), membuktikan adopsi gawai sangat membedakan segmen pelanggan. Sebaliknya, seluruh fitur `n01`–`n15` memiliki $F < 2.0$ dengan $p > 0.10$ (tidak signifikan secara statistik).
   * **Eksperimen Ablasi:** Menguji Silhouette dan Stabilitas ARI antar-inisialisasi. Menambahkan `perangkat` menaikkan Silhouette menjadi $0.0623$ dan stabilitas ARI menjadi $0.8914$. Namun saat `n01`–`n15` dimasukkan, skor Silhouette langsung anjlok ke $0.0483$. Bukti ini mengunci keputusan untuk **membuang seluruh `n01`–`n15`**.
3. **Penyeimbangan Bobot Blok (*Block-Weighting*):** Jika 16 kolom CLR kategori dibiarkan berskala sama dengan 5 fitur RFM, jarak Euclidean akan didominasi sebesar $\frac{16}{16+5+...} \approx 37\%$ oleh kategori belanja, menutupi sinyal moneter. Dengan membagi setiap blok fitur dengan $\sqrt{p_{\text{blok}}}$, total varians tiap blok menjadi seimbang ($1.0$). Hasilnya spektakuler: **Silhouette Score melonjak $+67.3\%$ dari $0.0623$ ke $0.1043$**.

### Poin Kunci untuk Ujian Lisan:
1. **Bagaimana membuktikan bahwa `n01`–`n15` adalah noise?**
   * *Jawaban:* Menggunakan *methodological triangulation*: (1) Analisis distribusi membuktikan kemiripan dengan distribusi teoritis murni (Standard Normal, Uniform, Beta), (2) Korelasi dengan fitur inti $< 0.008$, (3) Uji ANOVA F-Test menghasilkan $p > 0.10$ (tidak ada variansi antar kelompok), dan (4) Ablasi membuktikan penurunan Silhouette sebesar $> 22\%$.

---

## 4. Tahap c: Komparasi Minimal 3 Metode Clustering

### Alur Pikir & Logika Dasar
Setiap keluarga algoritma memiliki asumsi bentuk klaster yang berbeda:
* **K-Means:** Mengasumsikan klaster berbentuk bulat (*spherical/convex*) dengan densitas dan ukuran yang seimbang.
* **Gaussian Mixture Model (GMM):** Model probabilistik parametrik yang mengasumsikan data berasal dari campuran distribusi Gaussian dengan kovariansi elips.
* **Agglomerative Hierarchical Clustering:** Membangun pohon hierarki *bottom-up* dengan kriteria peminimalan varians (*Ward's minimum variance criterion*).
* **HDBSCAN:** Algoritma berbasis kepadatan (*density-based*) yang mencari wilayah kerapatan tinggi yang dipisahkan oleh wilayah jarang, mampu mendeteksi *noise* secara otomatis.

Pada dataset 120.000 baris, Agglomerative dan HDBSCAN membutuhkan matriks jarak $\mathcal{O}(N^2)$ yang tidak muat di memori RAM jika dihitung langsung. Solusi cerdasnya adalah melatih model pada subsampel representatif terstratifikasi ($N_{\text{sub}} = 15.000$), lalu memetakan ke 120.000 baris menggunakan *NearestCentroid Classifier*.

### Hasil & Poin Kunci untuk Ujian Lisan:
1. **Mengapa K-Means dipilih sebagai metode terbaik untuk tahap selanjutnya?**
   * *Jawaban:* 
     1. Meraih **Silhouette Score tertinggi** ($0.1041$ pada subsampel evaluasi yang seragam).
     2. Menunjukkan **stabilitas inisialisasi hampir sempurna** ($\text{ARI} = 0.9871$ pada 3 inisialisasi acak berbeda).
     3. Menghasilkan **distribusi ukuran klaster yang sangat masuk akal untuk bisnis** (klaster terkecil $8.8\%$, terbesar $33.3\%$, tidak ada klaster yang kolaps/kerdil).
     4. Waktu eksekusi sangat cepat ($3.07$ detik untuk keseluruhan 120.000 data).
2. **Mengapa Agglomerative Clustering penting meskipun K-Means sudah unggul?**
   * *Jawaban:* Agglomerative Clustering (linkage='ward') menghasilkan struktur partisi yang sangat serupa dan kongruen dengan K-Means ($\text{ARI} = 0.9790$, Silhouette $= 0.0901$). Ini menjadi bukti validasi silang metodologis (*triangulasi*) bahwa struktur klaster yang kita temukan benar-benar ada di dalam topologi data, bukan sekadar ilusi titik acak K-Means.
3. **Mengapa HDBSCAN gagal memberikan partisi yang baik pada dataset ini?**
   * *Jawaban:* HDBSCAN mengalami masalah *curse of dimensionality* pada ruang berdimensi 43 fitur kontinu. Dalam dimensi tinggi, jarak antar titik menjadi relatif seragam (*distance concentration phenomenon*), sehingga kontras kepadatan menjadi kabur. Akibatnya, HDBSCAN menyatukan $89.4\%$ data ke dalam 1 klaster tunggal raksasa.

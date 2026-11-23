# Blueprint 12 Tahap Operasional: Deteksi Penipuan Transaksi (*Fraud Detection* – Task 2 UTS)

Dokumen ini merupakan panduan kerja end-to-end untuk **Task 2 (Classification: Fraud Detection)**. Alur dipecah menjadi **12 tahap modular** agar mudah dieksekusi sel demi sel di Antigravity IDE, bebas dari kebocoran data (*data leakage*), dan langsung menghasilkan narasi analitis untuk `laporan.pdf` serta file akhir `prediksi_klasifikasi.csv`.

## Konteks Bisnis, Data, dan Aturan Evaluasi

* **File Dataset Lokal:** `klasifikasi_train.csv.gz` (data latih hingga penarikan **31 Agustus 2025**), `klasifikasi_test.csv.gz` (periode uji **September–Oktober 2025**), dan `kamus_data.md`.
* **Struktur Kolom Utama:** `id_transaksi`, `id_pelanggan`, `id_merchant`, `waktu`, `nominal`, `kanal`, `kategori_merchant`, `flag_review_manual`, fitur anonim internal `v01` hingga `v50`, serta target `is_fraud` (hanya pada data latih).
* **Karakteristik Masalah:** Ketimpangan kelas ekstrem (*severe class imbalance*) dengan rasio transaksi *fraud* sekitar $\approx 1\%$.
* **Fungsi Biaya (*Cost Function*):**
  * Menandai transaksi (`tandai = 1`) untuk tinjauan manual dikenakan biaya tetap $\text{Rp } 25.000$ per transaksi.
  * Meloloskan transaksi *fraud* (`is_fraud = 1` dan `tandai = 0`) menimbulkan kerugian finansial sebesar nilai `nominal` transaksi tersebut.
  * Formulasi total biaya yang harus diminimalkan:
    $$
    \text{Total Cost} = \sum_{i \in \text{Missed Fraud}} \text{nominal}_i + 25.000 \times \sum_{i} \text{tandai}_i
    $$
* **Metrik Penilaian (*Hidden Test Set*):**
  1. **PR-AUC (*Average Precision*)** dari kolom `prob` ($3$ poin).
  2. **Total Cost** dari kolom `tandai` ($3$ poin).
* **Format Keluaran Wajib:** File `prediksi_klasifikasi.csv` di cabang `main` repositori dengan header persis `id_transaksi,prob,tandai` ($\text{prob} \in [0, 1]$ dan $\text{tandai} \in \{0, 1\}$).

## Peta Keselarasan 12 Tahap terhadap Rubrik Laporan (`2a`–`2g`)

| Tahap Operasional (Kode / Notebook) | Sub-Bab Laporan & Rubrik | Fokus Utama & Keluaran Kunci |
| :--- | :--- | :--- |
| **Tahap 1 & 2** | Pendahuluan & Profil Data | Reproducibility, verifikasi skema `kamus_data.md`, dan profil imbalance |
| **Tahap 3** | **Soal 2a** – Audit Label & Fitur | Investigasi *delayed labels* & bukti kuantitatif *leakage* (`flag_review_manual`) |
| **Tahap 4** | **Soal 2b** – Validasi Berbasis Waktu | *Time-based split* & penerapan *embargo window* untuk label belum matang |
| **Tahap 5** | **Soal 2c** – Fitur Perilaku (*Behavioral*) | Rekayasa fitur historis kausal ($t' < t$) pada pelanggan & merchant |
| **Tahap 6 & 7** | **Soal 2d** – Model & *Class Imbalance* | Perbandingan model klasikal & $\ge 3$ strategi penanganan imbalance |
| **Tahap 8** | **Soal 2e** – Analisis *Pattern Shift* | Deteksi *drift* bulanan hingga Sept–Okt 2025 & strategi adaptasi jendela latih |
| **Tahap 9 & 10** | **Soal 2f** – Kalibrasi & Aturan Biaya | Kalibrasi `prob` & aturan keputusan berbasis biaya $\text{Rp } 25.000$ |
| **Tahap 11** | **Soal 2g** – Analisis Kesalahan (*Error*) | Bedah FP/FN per `kanal`, `kategori_merchant`, dan pola nominal |
| **Tahap 12** | Submisi & Reproduksibilitas | Generasi `prediksi_klasifikasi.csv` & verifikasi *one-command run* |

## Panduan Gaya Penulisan Sel untuk Antigravity Agent

Saat menghasilkan kode Python/Notebook dari dokumen ini:
1. Setiap tahap wajib diawali sel penjelasan berbahasa Indonesia yang memaparkan **pertanyaan analitis / hipotesis** sebelum menjalankan kode, dan diakhiri **interpretasi hasil** berdasarkan angka aktual yang muncul.
2. Seluruh operasi harus deterministik menggunakan `SEED = 42` dan hemat memori (memanfaatkan tipe data `float32`/`category` bila diperlukan).
3. Dilarang keras memakai pustaka *deep learning*; gunakan ekosistem *classical machine learning* (`scikit-learn`, `lightgbm`, `xgboost`, `catboost`).

## Tahap 1: Persiapan Lingkungan, Reproducibility & Inspeksi Skema Awal

* **Tujuan:** Menyiapkan fondasi eksperimen yang bersih, dapat direproduksi dengan satu perintah, dan sesuai dengan definisi di `kamus_data.md`.
* **Langkah Kerja:**
  1. Mengatur *random seed* global (`SEED = 42`) dan mencatat versi pustaka utama (`python`, `numpy`, `pandas`, `scikit-learn`, `lightgbm`, `xgboost`, `catboost`).
  2. Memuat `klasifikasi_train.csv.gz` dan `klasifikasi_test.csv.gz`, lalu mengonversi kolom `waktu` menjadi objek `datetime` yang terurut secara kronologis.
  3. Memverifikasi kesesuaian seluruh nama kolom dengan `kamus_data.md`, mengecek duplikasi `id_transaksi`, serta memvalidasi batas tanggal data latih (hingga **31 Agustus 2025**) dan data uji (**September–Oktober 2025**).
* **Alur Berpikir Laporan:** Memastikan integritas teknis dataset sebelum melakukan asumsi pemodelan apa pun.

## Tahap 2: Eksplorasi Data Awal (EDA) & Profil Ketimpangan Kelas

* **Tujuan:** Memahami distribusi target, nilai transaksi, saluran pembayaran, serta karakteristik fitur internal `v01` hingga `v50`.
* **Langkah Kerja:**
  1. Menghitung proporsi global `is_fraud` untuk mengonfirmasi tingkat ketimpangan kelas ($\approx 1\%$).
  2. Memeriksa distribusi `nominal` (kuantil, nilai minimum, maksimum, dan perbandingan sebaran `nominal` antara transaksi sah vs *fraud*).
  3. Memetakan volume transaksi dan tingkat *fraud* awal berdasarkan kategori `kanal` dan `kategori_merchant`.
  4. Menganalisis pola *missing values* dan tipe data pada kolom `v01` hingga `v50` (membedakan mana yang bersifat kontinu, diskrit, atau indikator biner).
* **Alur Berpikir Laporan:** Menunjukkan bahwa rata-rata kerugian per kasus *fraud* jauh melampaui biaya tinjau manual $\text{Rp } 25.000$, sehingga metrik akurasi biasa tidak relevan.

## Tahap 3: Audit Pembentukan Label & Deteksi Data Leakage (*Soal 2a*)

* **Tujuan:** Mengidentifikasi kolom yang membocorkan informasi masa depan (*post-event leakage*) dan memeriksa kelengkapan label menjelang tanggal penarikan data (**31 Agustus 2025**).
* **Langkah Kerja:**
  1. **Audit Kolom Mencurigakan:** Mengevaluasi seluruh fitur terhadap waktu kejadian transaksi (`waktu`), khususnya kolom seperti `flag_review_manual` dan variabel `v01` hingga `v50` yang memiliki korelasi ekstrem atau pola *missing* yang bergantung pada status akhir transaksi.
  2. **Audit Kematangan Label (*Delayed Labels*):** Memplot jumlah transaksi dan persentase `is_fraud` secara harian pada 30 hari terakhir sebelum **31 Agustus 2025**. Penurunan tajam rasio *fraud* di hari-hari terakhir mengindikasikan adanya *fraud* yang belum selesai diinvestigasi (*right-censored labels*).
  3. **Eksperimen Bukti Dampak Leakage:** Melatih model *baseline* cepat menggunakan seluruh fitur mentah (termasuk `flag_review_manual`) lalu membandingkannya dengan model setelah fitur bocor dibuang. Mencatat penurunan PR-AUC dan pergeseran *feature importance* sebagai bukti kuantitatif dampak kebocoran data.
* **Alur Berpikir Laporan:** Membuktikan secara empiris mengapa skor validasi yang terlalu tinggi akibat fitur pasca-transaksi adalah ilusi yang akan gagal saat inferensi pada September–Oktober 2025.

## Tahap 4: Desain Skema Validasi Berbasis Waktu (*Soal 2b*)

* **Tujuan:** Membangun skema validasi yang mencerminkan kondisi deployment nyata di masa depan serta mengamankan model dari label yang belum matang.
* **Langkah Kerja:**
  1. Menetapkan pemisahan berbasis waktu (*Out-of-Time Split*): misalnya menjadikan periode **Juli–Agustus 2025** (atau **Agustus 2025**) sebagai *validation set* utama, dan bulan-bulan sebelumnya sebagai *training set* (disertai skema *expanding window* multi-bulan untuk menguji stabilitas).
  2. Menerapkan **jendela embargo (*embargo / buffer period*)** sepanjang $H$ hari (sesuai temuan jeda konfirmasi label pada Tahap 3) di ujung data latih sebelum batas potong (*cutoff*), sehingga baris dengan label `0` yang belum final tidak meracuni proses pelatihan.
  3. Mengunci set validasi ini agar seluruh eksperimen pada Tahap 5 hingga Tahap 10 dievaluasi pada data yang identik.
* **Alur Berpikir Laporan:** Menjelaskan mengapa *Random K-Fold* tidak valid untuk data transaksi berurutan waktu dan bagaimana *embargo window* mencegah bias *false negative* pada data latih terbaru.

## Tahap 5: Rekayasa Fitur Perilaku (*Behavioral Feature Engineering*) (*Soal 2c*)

* **Tujuan:** Menangkap anomali perilaku transaksi pelanggan dan merchant secara historis tanpa menyentuh informasi masa depan ($t' < t$).
* **Langkah Kerja:**
  1. **Kecepatan Transaksi (*Transaction Velocity*):** Menghitung frekuensi transaksi `id_pelanggan` dan `id_merchant` dalam jendela waktu bergulir $1\text{ jam}$, $6\text{ jam}$, $24\text{ jam}$, dan $7\text{ hari}$ terakhir sebelum `waktu` transaksi saat ini, serta selisih detik sejak transaksi sebelumnya (*time-since-last-tx*).
  2. **Deviasi & Rasio Nominal:** Menghitung rasio `nominal` saat ini terhadap rata-rata, median, dan nilai maksimum historis `id_pelanggan` terkait, serta *z-score* `nominal` terhadap riwayat pelanggan maupun `kategori_merchant`.
  3. **Kebaruan & Eksplorasi Entitas (*Novelty*):** Membuat indikator apakah ini transaksi pertama pelanggan di `id_merchant`, `kategori_merchant`, atau `kanal` tersebut, serta jumlah variasi merchant unik dalam $24\text{ jam}$ terakhir.
  4. **Fitur Temporal & Profil Risiko Historis:** Mengekstrak jam transaksi (indikator transaksi dini hari), hari dalam seminggu, serta *lagged smoothed fraud rate* per `id_merchant` dan `kategori_merchant` yang dihitung hanya dari jendela masa lalu yang sudah melewati batas *embargo*.
  5. **Verifikasi Kausalitas:** Melakukan pengujian sanitasi untuk memastikan tidak ada fitur perilaku yang berubah nilainya jika data masa depan dimodifikasi.
* **Alur Berpikir Laporan:** Menghubungkan setiap kelompok fitur dengan intuisi modus operandi *fraud* (seperti *card testing*, *account takeover*, atau pengurasan saldo cepat).

## Tahap 6: Seleksi Fitur, Pra-pemrosesan & Pembandingan Model Baseline

* **Tujuan:** Menyaring fitur yang relevan dan membandingkan performa beberapa keluarga algoritma *classical machine learning* pada skema validasi waktu yang sama.
* **Langkah Kerja:**
  1. Menerapkan pra-pemrosesan bebas kebocoran (*fit* hanya pada *training fold*): pengkodean variabel kategorikal (`kanal`, `kategori_merchant`, dll.) dan penanganan nilai hilang pada `v01` hingga `v50`.
  2. Melatih dan membandingkan empat kandidat model: **Logistic Regression** (sebagai *linear baseline*), **LightGBM**, **XGBoost**, dan **CatBoost** dengan konfigurasi dasar.
  3. Mengevaluasi keempat model menggunakan metrik utama **PR-AUC (*Average Precision*)** dan mencatat waktu komputasi serta stabilitasnya.
* **Alur Berpikir Laporan:** Menetapkan tolok ukur (*benchmark*) objektif sebelum melakukan rekayasa penanganan *class imbalance*.

## Tahap 7: Eksperimen Penanganan Class Imbalance (*Soal 2d*)

* **Tujuan:** Mengevaluasi pengaruh perlakuan *class imbalance* terhadap kualitas peringkat probabilitas (PR-AUC) maupun dampaknya pada biaya bisnis.
* **Langkah Kerja:**
  1. Menggunakan 1–2 keluarga model terbaik dari Tahap 6, membandingkan **tiga pendekatan utama** secara terisolasi:
     * **Pendekatan A – *Class Weighting & Cost-Sensitive Weighting*:** Menguji pembobotan kelas (`scale_pos_weight` / `class_weight='balanced'`) serta pembobotan sampel berbasis kerugian finansial (memberi bobot lebih tinggi pada kasus *fraud* ber-`nominal` besar).
     * **Pendekatan B – *Resampling* pada Lipatan Latih:** Menguji *Random Undersampling* (RUS) kelas mayoritas dan/atau *oversampling* minoritas yang diterapkan **hanya pada data latih**, sementara data validasi dipertahankan pada distribusi asli ($\approx 1\%$).
     * **Pendekatan C – *Distribusi Asli + Threshold Tuning*:** Melatih model pada distribusi kelas alami tanpa pembobotan maupun *resampling* agar probabilitas dasar tidak terdistorsi.
  2. Membandingkan PR-AUC dan simulasi **Total Cost** awal dari ketiga pendekatan tersebut pada *validation set*.
* **Alur Berpikir Laporan:** Menjelaskan mengapa peningkatan *recall* dari *resampling* atau *weighting* tidak selalu menurunkan *Total Cost* jika memicu lonjakan *false positive* yang masing-masing berbiaya $\text{Rp } 25.000$.

## Tahap 8: Analisis Pergeseran Pola (*Pattern Shift*) & Adaptasi Model (*Soal 2e*)

* **Tujuan:** Menguji apakah pola transaksi dan modus *fraud* mengalami pergeseran antar-waktu serta menentukan strategi pelatihan yang paling tahan terhadap *drift*.
* **Langkah Kerja:**
  1. **Diagnostik Pergeseran:** Membandingkan tren bulanan *fraud rate*, median `nominal` *fraud*, komposisi `kanal`, dan nilai *Population Stability Index* (PSI) fitur-fitur utama antar-bulan pada data latih hingga ke fitur data uji **September–Oktober 2025**.
  2. **Uji Degradasi Temporal:** Mengevaluasi bagaimana performa model yang dilatih pada bulan-bulan awal menurun ketika diuji secara berurutan pada bulan-bulan berikutnya.
  3. **Eksperimen Strategi Adaptasi:** Membandingkan pelatihan menggunakan seluruh riwayat (*Expanding Window*), hanya beberapa bulan terakhir (*Sliding Window*), serta pemberian bobot peluruhan waktu (*Exponential Time-Decay Sample Weights*) yang memprioritaskan transaksi terbaru.
* **Alur Berpikir Laporan:** Memberikan alasan berbasis bukti mengenai rentang data historis dan pembobotan waktu yang dipilih untuk melatih model final menuju periode September–Oktober 2025.

## Tahap 9: Kalibrasi Probabilitas (*Probability Calibration*) (*Soal 2f – Bagian 1*)

* **Tujuan:** Memastikan nilai keluaran `prob` merepresentasikan peluang kejadian *fraud* yang sesungguhnya agar layak digunakan dalam perhitungan ekspektasi kerugian rupiah.
* **Langkah Kerja:**
  1. Mengevaluasi kualitas kalibrasi model terbaik sebelum kalibrasi menggunakan **Brier Score**, **Log-Loss**, dan **Calibration Curve (*Reliability Diagram*)**.
  2. Menerapkan dua metode kalibrasi pasca-pemrosesan pada prediksi validasi (menggunakan skema *out-of-fold* atau split kalibrasi terpisah agar bebas *overfit*):
     * **Platt Scaling (*Sigmoid Calibration*)**
     * **Isotonic Regression**
  3. Memilih metode kalibrasi yang menghasilkan Brier Score dan kurva reliabilitas terbaik tanpa menurunkan PR-AUC secara signifikan.
* **Alur Berpikir Laporan:** Menjelaskan perbedaan fungsi `prob` untuk peringkat (PR-AUC) dan fungsi `prob` sebagai input perkalian nilai uang pada keputusan penandaan (`tandai`).

## Tahap 10: Optimasi Keputusan Berbasis Biaya Rp 25.000 (*Soal 2f – Bagian 2*)

* **Tujuan:** Menerjemahkan probabilitas terkalibrasi $p_i$ dan `nominal` transaksi $N_i$ menjadi keputusan biner `tandai` yang meminimalkan **Total Cost**.
* **Langkah Kerja:**
  1. **Penurunan Aturan Keputusan Matematis:**
     * Ekspektasi kerugian jika transaksi dibiarkan lolos (`tandai = 0`) adalah $p_i \times N_i$, sedangkan biaya jika ditandai (`tandai = 1`) adalah $\text{Rp } 25.000$.
     * Keputusan optimal secara teoretis adalah menandai transaksi apabila:
       $$
       p_i \times N_i > 25.000 \iff p_i > \frac{25.000}{N_i}
       $$
     * **Batasan Ekonomi Mutlak:** Transaksi dengan $N_i \le \text{Rp } 25.000$ tidak akan pernah ditandai (`tandai = 0`), karena biaya pemeriksaan manualnya sudah melebihi potensi kerugian maksimalnya ($p_i \le 1$).
  2. **Kalibrasi Faktor Pengali Ambang ($\alpha$):** Menguji variasi faktor pengali $\alpha$ pada aturan $p_i > \alpha \cdot \frac{25.000}{N_i}$ di *validation set* untuk mengompensasi ketidakpastian estimasi probabilitas dan menemukan titik minimum **Total Cost**.
  3. **Benchmarking Strategi Keputusan:** Menyajikan tabel perbandingan **Total Cost**, jumlah transaksi ditandai, *Precision*, dan *Recall* antara:
     * *Baseline 1 – Flag None* (`tandai = 0` untuk semua transaksi).
     * *Baseline 2 – Flag All* (`tandai = 1` untuk semua transaksi).
     * *Baseline 3 – Fixed Probability Threshold* (ambang probabilitas tunggal tanpa mempertimbangkan `nominal`).
     * *Proposed – Cost-Sensitive Calibrated Rule* (aturan berbasis biaya terkalibrasi).
* **Alur Berpikir Laporan:** Mendemonstrasikan penghematan rupiah nyata yang dihasilkan model dibandingkan kebijakan ekstrem (*flag all* / *flag none*) maupun ambang probabilitas konvensional.

## Tahap 11: Analisis Kesalahan (*Error Analysis*) & Interpretabilitas (*Soal 2g*)

* **Tujuan:** Membedah segmen kekuatan dan kelemahan model sebagai dasar rekomendasi operasional dan persiapan diskusi ujian lisan.
* **Langkah Kerja:**
  1. **Breakdown Performa & Kerugian per `kanal`:** Menghitung PR-AUC, jumlah *False Positive* (pemborosan biaya review $\text{Rp } 25.000$), dan total kerugian *False Negative* (nominal *fraud* lolos) di setiap kanal transaksi.
  2. **Breakdown per `kategori_merchant`:** Mengidentifikasi kategori merchant dengan kebocoran kerugian tertinggi maupun tingkat alarm palsu tertinggi.
  3. **Analisis Berdasarkan Pola Fraud & Ukuran Transaksi:** Mengevaluasi perilaku model pada transaksi bernilai mikro ($N_i \le 25.000$), menengah, hingga transaksi bernilai sangat tinggi, serta pola akun baru vs akun lama.
  4. **Interpretasi Fitur:** Menampilkan peringkat kontribusi fitur (*Feature Importance* / *SHAP summary*) untuk menjelaskan faktor utama pendorong prediksi model.
* **Alur Berpikir Laporan:** Memberikan wawasan bisnis yang tajam, misalnya menjelaskan bahwa *false negative* pada nominal kecil adalah keputusan rasional berbasis biaya, sementara fokus perbaikan harus diarahkan pada kebocoran di segmen nominal besar.

## Tahap 12: Pelatihan Final, Generasi `prediksi_klasifikasi.csv` & Verifikasi

* **Tujuan:** Menghasilkan prediksi periode **September–Oktober 2025** dan memastikan seluruh artefak proyek mematuhi spesifikasi pengumpulan.
* **Langkah Kerja:**
  1. Melatih ulang pipeline lengkap terbaik menggunakan data historis terkini yang sudah melewati jendela *embargo* sebelum **31 Agustus 2025**, menerapkan model kalibrasi terpilih, dan mengeksekusi inferensi pada `klasifikasi_test.csv.gz`.
  2. Mengekspor file `prediksi_klasifikasi.csv` ke direktori utama repositori dengan header persis `id_transaksi,prob,tandai`.
  3. **Audit Otomatis File Submisi:**
     * Memastikan jumlah baris dan urutan `id_transaksi` identik $100\%$ dengan `klasifikasi_test.csv.gz`.
     * Memastikan tidak ada nilai `NaN`, `null`, atau `inf`.
     * Memastikan $0 \le \text{prob} \le 1$, $\text{tandai} \in \{0, 1\}$, dan tidak ada transaksi dengan $\text{nominal} \le 25.000$ yang diberi nilai `tandai = 1`.
     * Memverifikasi bahwa persentase transaksi yang ditandai pada data uji konsisten dengan distribusi pada data validasi.
  4. Merapikan skrip di folder `kode/` beserta `kode/README.md` agar seluruh proses mulai dari pembacaan data hingga pembuatan `prediksi_klasifikasi.csv` dapat dijalankan ulang dengan satu perintah tunggal.

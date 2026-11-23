# Kamus data UTS Machine Learning

Data latih regresi dan klasifikasi ditarik pada **31 Agustus 2025**. Data uji mencakup
1 September sampai 31 Oktober 2025. Semua kelompok menerima data yang sama.

## Tugas 1: regresi_train.csv.gz dan regresi_test.csv.gz

Satu baris per paket. Prediksi dibuat saat paket diambil kurir.

| Kolom | Keterangan |
| --- | --- |
| id_paket | ID paket |
| tgl_pickup, jam_pickup | Tanggal dan jam paket diambil kurir |
| layanan | Jenis layanan pengiriman |
| mitra_kurir | Mitra kurir yang mengantar |
| hub_asal, kota_asal | Hub penyortiran dan kota asal |
| pulau_tujuan, kota_tujuan | Pulau dan kota tujuan |
| jarak_km | Jarak rute menurut sistem (km) |
| berat_kg | Berat yang dideklarasikan penjual (kg) |
| volume_liter, jumlah_item, nilai_barang | Volume, jumlah barang, dan nilai barang (rupiah) |
| asuransi, cod | 1 jika diasuransikan, 1 jika bayar di tempat |
| alamat_skor | Skor kelengkapan alamat penerima (0-1) |
| pengalaman_kurir_bulan | Pengalaman kurir penjemput (bulan) |
| rating_penjual, umur_akun_penjual_hari, volume_penjual_30h | Profil penjual |
| antrean_hub_asal | Jumlah paket menunggu di hub asal saat paket diambil |
| indeks_kemacetan | Indeks kemacetan kota asal saat pengambilan (0-1) |
| hujan_asal_mm, hujan_tujuan_mm | Prakiraan curah hujan hari itu (mm) |
| jumlah_scan | Jumlah scan pelacakan yang tercatat di sistem |
| f01-f50 | Fitur internal sistem logistik (anonim), dihitung saat paket diambil |
| durasi_jam | Label: jam dari pengambilan sampai paket diterima. Hanya di data latih |

## Tugas 2: klasifikasi_train.csv.gz dan klasifikasi_test.csv.gz

Satu baris per transaksi. Skor dipakai saat transaksi terjadi.

| Kolom | Keterangan |
| --- | --- |
| id_transaksi, waktu | ID dan waktu transaksi |
| id_pengguna, id_merchant, kategori_merchant | Pengguna, merchant, dan kategorinya |
| kanal, tipe_perangkat | Kanal pembayaran dan jenis perangkat |
| nominal | Nominal transaksi (rupiah) |
| jam, hari_libur | Jam transaksi, 1 jika hari libur nasional |
| umur_akun_hari, level_kyc, umur_pengguna_tahun, email_domain | Profil akun pengguna |
| umur_perangkat_hari | Umur perangkat sejak pertama terlihat di akun ini (hari) |
| perangkat_baru | 1 jika perangkat pertama kali terlihat dalam 3 hari terakhir |
| jumlah_akun_per_perangkat | Jumlah akun yang pernah memakai perangkat ini |
| ganti_password_7h, gagal_login_24j | Perubahan kata sandi 7 hari terakhir, gagal masuk 24 jam terakhir |
| jarak_lokasi_km, negara_ip_beda | Jarak dari lokasi kebiasaan pengguna, 1 jika IP dari negara lain |
| kartu_tidak_hadir | 1 jika transaksi kartu tanpa kartu fisik |
| menit_sejak_trx_terakhir, trx_1j, trx_24j, trx_7h, nominal_24j | Riwayat pengguna sebelum transaksi ini |
| rasio_nominal_rata2 | Nominal dibagi rata-rata nominal pengguna sebelumnya |
| saldo_sebelum, persen_saldo | Saldo dompet sebelum transaksi dan porsi saldo yang dipakai |
| umur_merchant_hari, refund_merchant_30h, volume_merchant_30h | Profil merchant (refund dalam persen) |
| flag_review_manual | 1 jika transaksi pernah ditinjau manual oleh tim risiko |
| v01-v50 | Fitur internal sistem risiko (anonim) |
| is_fraud | Label: 1 jika transaksi terkonfirmasi fraud saat data ditarik. Hanya di data latih |

Parameter biaya: setiap transaksi yang ditandai untuk ditinjau berbiaya **Rp 25.000**.
Transaksi fraud yang ditandai dianggap tertahan tanpa kerugian. Transaksi fraud yang
lolos merugikan sebesar nominalnya.

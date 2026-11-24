"""Satu perintah untuk mereproduksi semua angka laporan dan file prediksi.

Dari akar repositori:
    python kode/jalankan.py            # semua task, lalu laporan.pdf
    python kode/jalankan.py 1          # hanya Task 1
    python kode/jalankan.py 2 3        # Task 2 dan Task 3
    python kode/jalankan.py laporan    # hanya membuat ulang laporan.pdf
"""
import shutil
import subprocess
import sys
import time
from pathlib import Path

KODE = Path(__file__).resolve().parent
AKAR = KODE.parent

TUGAS = {
    "1": ("notebook", "01_regresi.ipynb"),          # -> prediksi_regresi.csv, hasil/tugas1.json
    "2": ("skrip", "run_pipeline.py"),              # -> prediksi_klasifikasi.csv, reports/summary_metrics.json
    "3": ("notebook", "03_clustering.ipynb"),       # -> cluster.csv, hasil/tugas3.json
}
EDGE = [
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
]


def jalankan_notebook(nama):
    import nbformat
    from nbclient import NotebookClient

    path = KODE / nama
    nb = nbformat.read(path, as_version=4)
    NotebookClient(nb, timeout=None, kernel_name="python3",
                   resources={"metadata": {"path": str(KODE)}}).execute()
    nbformat.write(nb, path)


def jalankan_skrip(nama):
    subprocess.run([sys.executable, str(KODE / nama)], cwd=AKAR, check=True)


def buat_laporan():
    sumber = KODE / "laporan" / "laporan.html"
    tujuan = AKAR / "laporan.pdf"
    peramban = next((str(p) for p in EDGE if p.exists()), None) or shutil.which("msedge") \
        or shutil.which("chromium") or shutil.which("google-chrome")
    if peramban is None:
        print("Peramban Edge/Chrome tidak ditemukan; buka kode/laporan/laporan.html lalu cetak ke PDF.")
        return
    subprocess.run([peramban, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={tujuan}", sumber.as_uri()], check=True)
    print(f"laporan.pdf dibuat: {tujuan}")


def main(argumen):
    pilihan = argumen or ["1", "2", "3", "laporan"]
    for p in pilihan:
        mulai = time.time()
        if p == "laporan":
            buat_laporan()
        else:
            jenis, nama = TUGAS[p]
            print(f"[Task {p}] menjalankan {nama} ...", flush=True)
            jalankan_notebook(nama) if jenis == "notebook" else jalankan_skrip(nama)
        print(f"[{p}] selesai dalam {time.time() - mulai:.0f} detik", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])

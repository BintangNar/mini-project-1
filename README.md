# Laporan Restorasi Citra: Lena Denoising

**Nama:** Bintang Narindra Putra Pratama  
**NRP:** 5024231038

## 1. Penjelasan Pipeline Restorasi
Proses restorasi citra ini menggunakan pendekatan sekuensial yang menggabungkan teknik *denoising* (penghilangan derau) dan *enhancement* (peningkatan kualitas). Berikut adalah urutan dan alasan pemilihan teknik tersebut:

1.  **`median_filter(image, 3)`**: Langkah awal untuk menghilangkan *salt-and-pepper noise* (derau bintik hitam putih) dengan tetap mempertahankan integritas tepi objek.
2.  **`gaussian_filter(step1, 3)`**: Digunakan untuk menghaluskan sisa-sisa derau Gaussian (derau halus) yang masih tertinggal setelah filter median.
3.  **`histogram_equalization(step2)`**: Dilakukan untuk memperbaiki distribusi intensitas warna. Ini penting agar detail citra yang sempat pudar akibat proses penghalusan (*blurring*) menjadi lebih kontras dan jelas.
4.  **`median_filter(step3, 5)`**: Dilakukan kembali dengan ukuran kernel yang lebih besar (5x5) untuk memastikan sisa-sisa derau yang lebih kompleks benar-benar tereliminasi setelah kontras dinaikkan.
5.  **`gaussian_filter(step4)`**: Tahap penghalusan akhir untuk memberikan tampilan citra yang lebih natural dan lembut sebelum tahap penajaman.
6.  **`unsharp_mask(step5)`**: Langkah terakhir untuk menajamkan kembali tepi-tepi objek (*edge enhancement*) yang mungkin menjadi terlalu halus akibat penggunaan filter bertahap.

## 2. Perbandingan Visual

| Sebelum (Noisy Source) | Sesudah (Restoration Result) | Target Restorasi |
| :---: | :---: | :---: |
| ![Noisy](bahan\test_image_lena_noisy.png) | ![Result](results\resuls.png) |![Result](bahan\test_image_lena_ori.png)


## 3. Analisis Singkat
* **Keberhasilan**: Pipeline ini sangat efektif dalam menekan derau pada citra input sehingga tampilan objek utama menjadi jauh lebih bersih. Penggunaan *Unsharp Masking* di akhir berhasil mengembalikan definisi pada area mata dan topi yang sebelumnya sedikit buram.
* **Peningkatan**: Karena proses filtering dilakukan sebanyak empat kali (2x median, 2x gaussian), terdapat potensi kehilangan detail tekstur halus pada area kulit. Penggunaan teknik seperti *Bilateral Filter* di masa depan dapat dipertimbangkan untuk menjaga tekstur asli,
Histogram Equalization membuat banyak noise yang belum hilang sempurna dari filter yang digunakan, penggunaan manual dan tanpa vektorisasi membuat program berjalan lambat.

## 4. Cara Menjalankan Program
Pastikan Anda memiliki Python dan library `opencv-python` serta `numpy`.

1. Letakkan file gambar `test_image_lena_noisy.jpg` di folder yang sama dengan script.
2. Jalankan perintah berikut di terminal:
   ```bash
   python restoration.py

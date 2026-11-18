# Analisis Data End-to-End & Dashboard Interaktif: Pasar Properti Jabodetabek

Proyek portofolio **Data Analyst / Data Science** end-to-end yang menganalisis faktor-faktor penentu harga rumah di wilayah **Jabodetabek (Jakarta, Bogor, Depok, Tangerang, Bekasi)**, dilengkapi pipeline pembersihan data teraudit, analisis berbasis **SQL (SQLite)**, model prediksi **Machine Learning (Random Forest)**, serta **Dashboard Interaktif Streamlit**.

---

## 1. Pertanyaan Bisnis (Business Questions)

1. **Perbandingan Wilayah:** Bagaimana distribusi harga rumah dan harga tanah per m2 di setiap kota/kabupaten di Jabodetabek?
2. **Pengaruh Legalitas:** Seberapa besar selisih valuasi antara rumah bersertifikat **SHM (Sertifikat Hak Milik)** dibandingkan **HGB** atau sertifikat lainnya?
3. **Dampak Aksesibilitas & Transportasi:** Apakah kedekatan dengan akses pintu tol dan stasiun transportasi massal (KRL/MRT/LRT) meningkatkan harga jual secara signifikan?
4. **Rekomendasi Best Value:** Kecamatan mana saja yang menawarkan rasio luas bangunan terhadap harga tanah paling kompetitif bagi pembeli hunian pertama?

---

## 2. Arsitektur & Struktur Proyek

```text
Analisis-Data-End-to-End-Dashboard/
â”œâ”€â”€ data/
â”‚   â”œâ”€â”€ raw_jabodetabek_houses.csv      # Dataset mentah hasil pengumpulan/scraping
â”‚   â”œâ”€â”€ clean_jabodetabek_houses.csv    # Dataset bersih siap analisis
â”‚   â”œâ”€â”€ jabodetabek_houses.db           # Database SQLite untuk query SQL
â”‚   â””â”€â”€ cleaning_audit_log.json         # Dokumentasi log pembersihan data
â”œâ”€â”€ models/
â”‚   â””â”€â”€ house_price_rf.pkl              # Model Random Forest terlatih
â”œâ”€â”€ notebooks/
â”‚   â””â”€â”€ eda_modeling.ipynb              # Jupyter Notebook EDA, SQL & Modeling (visual langsung di GitHub)
â”œâ”€â”€ reports/
â”‚   â”œâ”€â”€ figures/                        # Visualisasi statis EDA (PNG)
â”‚   â”œâ”€â”€ sql_business_insights.json      # Hasil agregasi query SQL
â”‚   â””â”€â”€ model_evaluation.json           # Metrik evaluasi model (MAE, RMSE, MAPE, R2)
â”œâ”€â”€ src/
â”‚   â”œâ”€â”€ 01_collect_data.py              # Pengumpulan & penyiapan dataset mentah
â”‚   â”œâ”€â”€ 02_data_cleaning.py             # Pembersihan data, feature engineering & SQL loader
â”‚   â”œâ”€â”€ 03_eda_sql_analysis.py          # Eksekusi query SQL & visualisasi EDA
â”‚   â””â”€â”€ 04_train_model.py               # Pelatihan & evaluasi model prediksi
â”œâ”€â”€ app.py                              # Aplikasi Dashboard Interaktif Streamlit
â”œâ”€â”€ requirements.txt                    # Daftar dependensi Python
â”œâ”€â”€ .gitignore
â””â”€â”€ README.md
```

---

## 3. Dokumentasi Keputusan Pembersihan Data (Data Cleaning)

Dataset mentah mengandung berbagai anomali yang umum ditemui pada data hasil *web scraping* portal properti. Setiap langkah pembersihan dicatat otomatis ke dalam `data/cleaning_audit_log.json`:

1. **Deduplikasi Data:** Menghapus baris duplikat persis dan duplikat `id_listing` dengan mempertahankan entri pertama (`keep='first'`).
2. **Standardisasi Kategori Wilayah:** Membersihkan spasi berlebih dan menyeragamkan penulisan kota yang tidak konsisten (misalnya `jak-sel`, `JakTim`, `Kab. Bogor`) ke dalam 8 wilayah standar Jabodetabek.
3. **Penghapusan Target Null:** Menghapus baris dengan `harga_juta_rp` kosong (`NaN`) agar tidak menimbulkan bias sintetis pada variabel target.
4. **Koreksi Satuan & Filter Outlier Domain:** Mendeteksi kesalahan input satuan Rupiah penuh (nilai > 500.000 juta) dan mengonversinya ke satuan Juta Rupiah (`/ 1.000.000`), serta menyaring anomali tidak masuk akal (luas tanah <= 0 m2, luas bangunan > 2.500 m2, harga < Rp 100 juta).
5. **Imputasi Missing Values Berbasis Kelompok:** Mengisi nilai kosong pada `luas_bangunan_m2` dan `jarak_stasiun_km` menggunakan **median per kota** (lebih tahan terhadap outlier dibanding rata-rata), `garasi` dengan median global, serta `sertifikat` dengan modus (`SHM`).
6. **Feature Engineering:** Membuat fitur analitik baru berupa `harga_miliar_rp`, `harga_per_m2_tanah_juta`, `rasio_bangunan_tanah`, dan `kategori_jarak_stasiun`.

---

## 4. Ringkasan Insight Bisnis (Bahasa Non-Teknis)

- **Lokasi Tetap Penentu Utama:** Harga tanah per meter persegi di **Jakarta Selatan** tercatat lebih dari **3 kali lipat** lebih tinggi dibandingkan wilayah penyangga seperti Bogor dan Bekasi.
- **Luas Tanah & Bangunan Menggerakkan Harga:** Dimensi fisik properti (luas tanah dan luas bangunan) memiliki korelasi paling kuat terhadap total harga rumah dibandingkan sekadar jumlah kamar tidur.
- **Premi Legalitas SHM:** Properti dengan status **Sertifikat Hak Milik (SHM)** memiliki nilai jual per meter persegi yang lebih tinggi dan stabil dibandingkan HGB atau surat lainnya karena memberikan kepastian kepemilikan jangka panjang.
- **Konektivitas Transportasi Meningkatkan Nilai Jual:** Rumah yang berada dalam radius **kurang dari 2 km dari stasiun** serta memiliki **akses tol langsung** menikmati premi harga yang konsisten di seluruh wilayah Jabodetabek.
- **Area Best Value:** Wilayah **Tangerang** (seperti Bintaro, BSD, Pamulang) dan **Depok** (Cinere, Beji) menawarkan titik keseimbangan paling menarik antara akses menuju pusat bisnis Jakarta dan harga per meter persegi yang masih kompetitif.

---

## 5. Cara Pemakaian (Instalasi & Menjalankan Proyek)

### Prasyarat
- Python 3.10 atau lebih baru
- Git

### Langkah 1: Clone Repository & Instalasi Dependensi
```bash
git clone [https://github.com/Iqbalarbani/Analisis-Data-End-to-End-Dashboard.git](https://github.com/Iqbalarbani/Analisis-Data-End-to-End-Dashboard.git)
cd Analisis-Data-End-to-End-Dashboard

# Buat virtual environment (opsional tapi disarankan)
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install library yang dibutuhkan
pip install -r requirements.txt
```

### Langkah 2: Jalankan Pipeline Data & Model Secara Berurutan
```bash
python src/01_collect_data.py
python src/02_data_cleaning.py
python src/03_eda_sql_analysis.py
python src/04_train_model.py
```

### Langkah 3: Jalankan Dashboard Interaktif Streamlit
```bash
streamlit run app.py
```
Dashboard akan otomatis terbuka di browser pada alamat `http://localhost:8501`.

---

## 6. Cara Upload ke GitHub

Jika Anda menjalankan proyek ini dari lokal dan ingin mengunggahnya ke repository GitHub Anda:

```bash
git init
git add .
git commit -m "feat: initial commit end-to-end data analysis & streamlit dashboard"
git branch -M main
git remote add origin [https://github.com/Iqbalarbani/Analisis-Data-End-to-End-Dashboard.git](https://github.com/Iqbalarbani/Analisis-Data-End-to-End-Dashboard.git)
git push -u origin main
```

> **Tips Publikasi:** Anda juga dapat mendeploy dashboard ini secara gratis menggunakan **Streamlit Community Cloud** dengan menghubungkan akun GitHub Anda dan memilih file utama `app.py`.

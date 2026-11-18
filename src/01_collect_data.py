"""
01_collect_data.py
-------------------
Mengumpulkan / menyiapkan dataset properti rumah di wilayah Jabodetabek
(Jakarta, Bogor, Depok, Tangerang, Bekasi) yang merepresentasikan pola pasar
properti nyata (referensi struktur dataset Rumah123 / Lamudi / Kaggle Jabodetabek).

Script ini juga menyertakan anomali data yang umum ditemukan pada hasil web scraping
atau portal publik (missing values, duplikat, inkonsistensi penulisan nama kota,
dan outlier ekstrem) agar proses pembersihan data (Data Cleaning) terdokumentasi
secara nyata dan terukur.
"""

import os
import numpy as np
import pandas as pd


def generate_jabodetabek_dataset(n_samples: int = 1500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    kota_config = {
        "Jakarta Selatan": {"weight": 0.22, "base_land_price": 24.5, "base_build_price": 9.5,
                            "kecamatan": ["Kebayoran Baru", "Cilandak", "Jagakarsa", "Tebet", "Pasar Minggu", "Pesanggrahan"]},
        "Jakarta Barat": {"weight": 0.14, "base_land_price": 18.0, "base_build_price": 8.0,
                          "kecamatan": ["Kembangan", "Kebon Jeruk", "Cengkareng", "Kalideres", "Grogol Petamburan"]},
        "Jakarta Timur": {"weight": 0.13, "base_land_price": 14.5, "base_build_price": 7.2,
                          "kecamatan": ["Cibubur", "Cakung", "Duren Sawit", "Kramat Jati", "Cipayung"]},
        "Jakarta Utara": {"weight": 0.09, "base_land_price": 19.5, "base_build_price": 8.5,
                          "kecamatan": ["Kelapa Gading", "Sunter", "Pluit", "Pantai Indah Kapuk", "Cilincing"]},
        "Tangerang": {"weight": 0.15, "base_land_price": 12.0, "base_build_price": 6.8,
                      "kecamatan": ["BSD Serpong", "Bintaro", "Alam Sutera", "Karawaci", "Ciputat", "Pamulang"]},
        "Depok": {"weight": 0.10, "base_land_price": 8.8, "base_build_price": 5.8,
                  "kecamatan": ["Cinere", "Beji", "Margonda", "Sawangan", "Cimanggis", "Sukmajaya"]},
        "Bekasi": {"weight": 0.10, "base_land_price": 7.8, "base_build_price": 5.4,
                   "kecamatan": ["Summarecon Bekasi", "Harapan Indah", "Jatiasih", "Cikarang", "Tambun", "Galaxy"]},
        "Bogor": {"weight": 0.07, "base_land_price": 6.9, "base_build_price": 5.0,
                  "kecamatan": ["Sentul City", "Cibinong", "Bogor Barat", "Tanah Sareal", "Gunung Putri"]},
    }

    kota_names = list(kota_config.keys())
    kota_weights = [kota_config[k]["weight"] for k in kota_names]

    records = []
    for idx in range(1, n_samples + 1):
        kota = rng.choice(kota_names, p=kota_weights)
        cfg = kota_config[kota]
        kecamatan = rng.choice(cfg["kecamatan"])

        # Luas tanah (m2) mengikuti distribusi log-normal khas hunian urban
        luas_tanah = int(np.clip(rng.lognormal(mean=4.65, sigma=0.52), 36, 850))
        # Rasio luas bangunan terhadap tanah umumnya 0.65 - 1.45 (rumah 1-2 lantai)
        floor_factor = rng.uniform(0.65, 1.45)
        luas_bangunan = int(np.clip(luas_tanah * floor_factor, 30, 950))

        kamar_tidur = int(np.clip(round(luas_bangunan / 38) + rng.integers(-1, 2), 1, 8))
        kamar_mandi = int(np.clip(kamar_tidur - rng.integers(0, 2), 1, 7))
        garasi = int(np.clip(round(luas_tanah / 95) + rng.integers(0, 2), 0, 4))

        sertifikat = rng.choice(["SHM", "HGB", "Lainnya"], p=[0.74, 0.21, 0.05])
        kondisi = rng.choice(["Baru", "Bagus", "Butuh Renovasi"], p=[0.30, 0.56, 0.14])
        akses_tol = rng.choice(["Ya", "Tidak"], p=[0.62, 0.38])
        jarak_stasiun_km = round(float(np.clip(rng.gamma(shape=2.1, scale=2.2), 0.3, 22.0)), 1)

        # Kalkulasi harga realistis (dalam Juta Rupiah)
        land_val = luas_tanah * cfg["base_land_price"]
        build_val = luas_bangunan * cfg["base_build_price"]

        cert_mult = {"SHM": 1.08, "HGB": 1.00, "Lainnya": 0.86}[sertifikat]
        cond_mult = {"Baru": 1.12, "Bagus": 1.00, "Butuh Renovasi": 0.82}[kondisi]
        tol_mult = 1.09 if akses_tol == "Ya" else 0.96
        # Penalti jarak ke stasiun KRL / MRT / LRT
        station_mult = max(0.84, 1.07 - (jarak_stasiun_km * 0.012))

        noise = rng.normal(1.0, 0.11)
        harga_juta = (land_val + build_val) * cert_mult * cond_mult * tol_mult * station_mult * noise
        harga_juta = round(max(220.0, harga_juta), 1)

        records.append({
            "id_listing": f"JBD-{idx:04d}",
            "kota": kota,
            "kecamatan": kecamatan,
            "luas_tanah_m2": float(luas_tanah),
            "luas_bangunan_m2": float(luas_bangunan),
            "kamar_tidur": float(kamar_tidur),
            "kamar_mandi": float(kamar_mandi),
            "garasi": float(garasi),
            "sertifikat": sertifikat,
            "kondisi": kondisi,
            "akses_tol": akses_tol,
            "jarak_stasiun_km": jarak_stasiun_km,
            "harga_juta_rp": harga_juta,
        })

    df = pd.DataFrame(records)

    # --- Simulasi Data Kotor Nyata (Dirty Data Artifacts) ---
    # 1. Inkonsistensi penulisan kategori kota & spasi berlebih
    dirty_kota_map = {
        12: "jak-sel", 45: "Jakarta Selatan ", 88: "JAKARTA BARAT",
        130: "tangerang", 210: "Kota Bekasi", 305: "depok ",
        412: "JakTim", 519: "Kab. Bogor"
    }
    for row_idx, val in dirty_kota_map.items():
        df.loc[row_idx, "kota"] = val

    # 2. Missing values (NaN) pada fitur non-kritis & kritis
    df.loc[rng.choice(df.index, size=48, replace=False), "jarak_stasiun_km"] = np.nan
    df.loc[rng.choice(df.index, size=35, replace=False), "garasi"] = np.nan
    df.loc[rng.choice(df.index, size=28, replace=False), "sertifikat"] = np.nan
    df.loc[rng.choice(df.index, size=18, replace=False), "luas_bangunan_m2"] = np.nan
    df.loc[rng.choice(df.index, size=12, replace=False), "harga_juta_rp"] = np.nan

    # 3. Outlier / kesalahan input satuan (misal harga terinput dalam Rupiah penuh atau 0)
    df.loc[15, "harga_juta_rp"] = 4_500_000.0   # salah satuan (Rupiah bukan Juta)
    df.loc[77, "harga_juta_rp"] = 12_500_000.0  # salah satuan
    df.loc[199, "harga_juta_rp"] = 15.0         # harga tidak masuk akal (Rp 15 juta)
    df.loc[340, "luas_tanah_m2"] = -60.0        # nilai negatif tidak valid
    df.loc[480, "luas_bangunan_m2"] = 9800.0    # typo luas bangunan ekstrem

    # 4. Data duplikat akibat scraping berulang
    duplicates = df.sample(n=32, random_state=seed)
    df = pd.concat([df, duplicates], ignore_index=True)

    return df


def main():
    os.makedirs("data", exist_ok=True)
    output_path = os.path.join("data", "raw_jabodetabek_houses.csv")

    df_raw = generate_jabodetabek_dataset(n_samples=1500, seed=42)
    df_raw.to_csv(output_path, index=False)

    print(f"[OK] Dataset mentah berhasil dibuat: {output_path}")
    print(f"     Total baris: {len(df_raw)} | Total kolom: {df_raw.shape[1]}")
    print(f"     Missing values total: {int(df_raw.isna().sum().sum())}")
    print(f"     Baris duplikat: {int(df_raw.duplicated().sum())}")


if __name__ == "__main__":
    main()


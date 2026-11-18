"""
02_data_cleaning.py
-------------------
Membersihkan dataset mentah properti Jabodetabek (data/raw_jabodetabek_houses.csv),
mendokumentasikan setiap keputusan pembersihan data ke dalam log audit JSON/Markdown,
dan memuat dataset bersih ke database SQLite (data/jabodetabek_houses.db) serta CSV.
"""

import json
import os
import sqlite3
import numpy as np
import pandas as pd


CITY_STANDARDIZATION = {
    "jak-sel": "Jakarta Selatan",
    "jakarta selatan": "Jakarta Selatan",
    "jakarta barat": "Jakarta Barat",
    "jaktim": "Jakarta Timur",
    "jakarta timur": "Jakarta Timur",
    "jakarta utara": "Jakarta Utara",
    "tangerang": "Tangerang",
    "kota tangerang": "Tangerang",
    "tangerang selatan": "Tangerang",
    "depok": "Depok",
    "kota depok": "Depok",
    "bekasi": "Bekasi",
    "kota bekasi": "Bekasi",
    "kab. bekasi": "Bekasi",
    "bogor": "Bogor",
    "kota bogor": "Bogor",
    "kab. bogor": "Bogor",
}


def standardize_city(val: str) -> str:
    if pd.isna(val):
        return np.nan
    cleaned = str(val).strip().lower()
    return CITY_STANDARDIZATION.get(cleaned, str(val).strip().title())


def clean_dataset(raw_path: str) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(raw_path)
    audit_log = {
        "initial_rows": int(len(df)),
        "initial_missing": int(df.isna().sum().sum()),
        "steps": [],
    }

    # 1. Menghapus baris duplikat persis & duplikat id_listing
    before = len(df)
    df = df.drop_duplicates().drop_duplicates(subset=["id_listing"], keep="first").copy()
    audit_log["steps"].append({
        "step": "1. Deduplikasi Data",
        "action": "Menghapus baris duplikat identik dan duplikat id_listing (keep='first').",
        "rows_removed": int(before - len(df)),
        "remaining_rows": int(len(df)),
    })

    # 2. Standardisasi nama kota (whitespace & variasi penulisan) ini update dikit 
    
    df["kota"] = df["kota"].apply(standardize_city)
    df["kecamatan"] = df["kecamatan"].astype(str).str.strip()
    audit_log["steps"].append({
        "step": "2. Standardisasi Kategori Lokasi",
        "action": "Menghapus spasi berlebih dan memetakan variasi penulisan kota (mis. 'jak-sel', 'JakTim', 'Kab. Bogor') ke 8 wilayah administratif standar Jabodetabek.",
        "unique_cities": sorted(df["kota"].dropna().unique().tolist()),
    })

    # 3. Menghapus baris dengan target variable (harga_juta_rp) kosong
    before = len(df)
    df = df.dropna(subset=["harga_juta_rp"]).copy()
    audit_log["steps"].append({
        "step": "3. Penghapusan Target Null",
        "action": "Menghapus baris dengan harga_juta_rp bernilai NaN karena merupakan variabel target utama dan imputasi berisiko menimbulkan bias harga.",
        "rows_removed": int(before - len(df)),
        "remaining_rows": int(len(df)),
    })

    # 4. Koreksi satuan & filtering anomali ekstrem (domain logic)
    # Jika harga > 500.000 juta (500 Miliar), terdeteksi salah input dalam Rupiah penuh -> bagi 1.000.000
    unit_fix_mask = df["harga_juta_rp"] > 500_000
    unit_fixed_count = int(unit_fix_mask.sum())
    df.loc[unit_fix_mask, "harga_juta_rp"] = df.loc[unit_fix_mask, "harga_juta_rp"] / 1_000_000.0

    before = len(df)
    valid_mask = (
        (df["harga_juta_rp"] >= 100.0)
        & (df["harga_juta_rp"] <= 100_000.0)
        & (df["luas_tanah_m2"] >= 20.0)
        & (df["luas_tanah_m2"] <= 2_500.0)
        & (df["luas_bangunan_m2"].isna() | ((df["luas_bangunan_m2"] >= 18.0) & (df["luas_bangunan_m2"] <= 2_500.0)))
    )
    df = df[valid_mask].copy()
    audit_log["steps"].append({
        "step": "4. Koreksi Satuan & Filter Outlier Domain",
        "action": "Mengonversi salah input satuan Rupiah ke Juta Rupiah (/1.000.000) dan menghapus nilai tidak masuk akal (luas tanah <= 0, luas bangunan > 2.500 m2, harga < Rp 100 juta).",
        "unit_corrected_rows": unit_fixed_count,
        "rows_removed": int(before - len(df)),
        "remaining_rows": int(len(df)),
    })

    # 5. Imputasi missing values berbasis kelompok wilayah
    # luas_bangunan_m2 diimputasi menggunakan median per kota
    df["luas_bangunan_m2"] = df["luas_bangunan_m2"].fillna(
        df.groupby("kota")["luas_bangunan_m2"].transform("median")
    )
    # jarak_stasiun_km diimputasi menggunakan median per kota
    df["jarak_stasiun_km"] = df["jarak_stasiun_km"].fillna(
        df.groupby("kota")["jarak_stasiun_km"].transform("median")
    )
    # garasi diimputasi dengan nilai modus/median (0 atau 1 berdasarkan luas tanah)
    df["garasi"] = df["garasi"].fillna(df["garasi"].median())
    # sertifikat diimputasi dengan kategori paling umum ('SHM')
    df["sertifikat"] = df["sertifikat"].fillna(df["sertifikat"].mode()[0])

    audit_log["steps"].append({
        "step": "5. Imputasi Missing Values",
        "action": "Imputasi median per kota untuk luas_bangunan_m2 dan jarak_stasiun_km; imputasi median global untuk garasi; imputasi modus ('SHM') untuk sertifikat.",
        "remaining_missing": int(df.isna().sum().sum()),
    })

    # 6. Feature Engineering & Penyesuaian Tipe Data
    int_cols = ["luas_tanah_m2", "luas_bangunan_m2", "kamar_tidur", "kamar_mandi", "garasi"]
    for col in int_cols:
        df[col] = df[col].round().astype(int)

    df["jarak_stasiun_km"] = df["jarak_stasiun_km"].round(1)
    df["harga_juta_rp"] = df["harga_juta_rp"].round(1)
    df["harga_miliar_rp"] = (df["harga_juta_rp"] / 1000.0).round(3)
    df["harga_per_m2_tanah_juta"] = (df["harga_juta_rp"] / df["luas_tanah_m2"]).round(2)
    df["rasio_bangunan_tanah"] = (df["luas_bangunan_m2"] / df["luas_tanah_m2"]).round(2)
    df["kategori_jarak_stasiun"] = pd.cut(
        df["jarak_stasiun_km"],
        bins=[0, 2.0, 5.0, 10.0, 100.0],
        labels=["Sangat Dekat (<2 km)", "Dekat (2-5 km)", "Sedang (5-10 km)", "Jauh (>10 km)"],
    ).astype(str)

    audit_log["steps"].append({
        "step": "6. Feature Engineering & Casting Tipe Data",
        "action": "Menambahkan fitur analitik: harga_miliar_rp, harga_per_m2_tanah_juta, rasio_bangunan_tanah, dan kategori_jarak_stasiun.",
        "final_rows": int(len(df)),
        "final_columns": int(df.shape[1]),
    })

    audit_log["final_rows"] = int(len(df))
    return df, audit_log


def save_to_sqlite(df: pd.DataFrame, db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    try:
        df.to_sql("houses", conn, if_exists="replace", index=False)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_houses_kota ON houses(kota);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_houses_sertifikat ON houses(sertifikat);")
        conn.commit()
    finally:
        conn.close()


def main():
    raw_path = os.path.join("data", "raw_jabodetabek_houses.csv")
    clean_csv_path = os.path.join("data", "clean_jabodetabek_houses.csv")
    db_path = os.path.join("data", "jabodetabek_houses.db")
    log_path = os.path.join("data", "cleaning_audit_log.json")

    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"File tidak ditemukan: {raw_path}. Jalankan src/01_collect_data.py terlebih dahulu.")

    df_clean, audit_log = clean_dataset(raw_path)
    df_clean.to_csv(clean_csv_path, index=False)
    save_to_sqlite(df_clean, db_path)

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(audit_log, f, indent=2, ensure_ascii=False)

    print(f"[OK] Pembersihan data selesai!")
    print(f"     Baris awal: {audit_log['initial_rows']} -> Baris bersih: {audit_log['final_rows']}")
    print(f"     Disimpan ke: {clean_csv_path} & {db_path} (tabel: 'houses')")
    print(f"     Log audit: {log_path}")


if __name__ == "__main__":
    main()

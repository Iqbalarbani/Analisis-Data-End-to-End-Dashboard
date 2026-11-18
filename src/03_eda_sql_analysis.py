"""
03_eda_sql_analysis.py
----------------------
Melakukan Exploratory Data Analysis (EDA) dan eksekusi query SQL pada database
SQLite (data/jabodetabek_houses.db) untuk menjawab 4 pertanyaan bisnis utama:
1. Bagaimana distribusi harga rumah dan harga tanah per m2 di setiap wilayah Jabodetabek?
2. Seberapa besar pengaruh jenis legalitas sertifikat (SHM vs HGB) terhadap nilai jual rumah?
3. Apakah kedekatan dengan akses tol dan stasiun transportasi (KRL/MRT/LRT) meningkatkan harga?
4. Kecamatan mana saja yang menawarkan rasio luas bangunan terhadap harga paling kompetitif?
"""

import json
import os
import sqlite3
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


SQL_QUERIES = {
    "q1_harga_per_kota": """
        SELECT
            kota,
            COUNT(*) AS jumlah_listing,
            ROUND(AVG(harga_miliar_rp), 2) AS rata_rata_harga_miliar,
            ROUND(AVG(harga_per_m2_tanah_juta), 2) AS rata_rata_harga_tanah_juta_m2,
            ROUND(AVG(luas_tanah_m2), 1) AS rata_rata_lt_m2,
            ROUND(AVG(luas_bangunan_m2), 1) AS rata_rata_lb_m2
        FROM houses
        GROUP BY kota
        ORDER BY rata_rata_harga_miliar DESC;
    """,
    "q2_pengaruh_sertifikat": """
        SELECT
            sertifikat,
            COUNT(*) AS jumlah_listing,
            ROUND(AVG(harga_miliar_rp), 2) AS rata_rata_harga_miliar,
            ROUND(AVG(harga_per_m2_tanah_juta), 2) AS rata_rata_harga_tanah_juta_m2
        FROM houses
        GROUP BY sertifikat
        ORDER BY rata_rata_harga_tanah_juta_m2 DESC;
    """,
    "q3_akses_transportasi": """
        SELECT
            akses_tol,
            kategori_jarak_stasiun,
            COUNT(*) AS jumlah_listing,
            ROUND(AVG(harga_miliar_rp), 2) AS rata_rata_harga_miliar,
            ROUND(AVG(harga_per_m2_tanah_juta), 2) AS rata_rata_harga_tanah_juta_m2
        FROM houses
        GROUP BY akses_tol, kategori_jarak_stasiun
        ORDER BY akses_tol DESC, rata_rata_harga_tanah_juta_m2 DESC;
    """,
    "q4_top_kecamatan_value": """
        SELECT
            kota,
            kecamatan,
            COUNT(*) AS jumlah_listing,
            ROUND(AVG(harga_miliar_rp), 2) AS rata_rata_harga_miliar,
            ROUND(AVG(luas_bangunan_m2), 1) AS rata_rata_lb_m2,
            ROUND(AVG(harga_per_m2_tanah_juta), 2) AS harga_tanah_juta_m2
        FROM houses
        GROUP BY kota, kecamatan
        HAVING COUNT(*) >= 15
        ORDER BY harga_tanah_juta_m2 ASC
        LIMIT 10;
    """,
}


def run_sql_analysis(db_path: str) -> dict[str, pd.DataFrame]:
    conn = sqlite3.connect(db_path)
    results = {}
    try:
        for name, query in SQL_QUERIES.items():
            results[name] = pd.read_sql_query(query, conn)
    finally:
        conn.close()
    return results


def save_static_charts(df: pd.DataFrame, sql_results: dict[str, pd.DataFrame], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # Chart 1: Rata-rata Harga Tanah per m2 & Harga Rumah per Kota
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    q1 = sql_results["q1_harga_per_kota"]

    sns.barplot(data=q1, x="rata_rata_harga_miliar", y="kota", ax=axes[0], palette="Blues_r", hue="kota", legend=False)
    axes[0].set_title("Rata-rata Harga Rumah per Wilayah (Miliar Rp)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Harga Rata-rata (Miliar Rp)")
    axes[0].set_ylabel("")

    sns.barplot(data=q1, x="rata_rata_harga_tanah_juta_m2", y="kota", ax=axes[1], palette="GnBu_r", hue="kota", legend=False)
    axes[1].set_title("Rata-rata Harga Tanah per m2 (Juta Rp/m2)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Harga Tanah (Juta Rp/m2)")
    axes[1].set_ylabel("")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "01_harga_per_wilayah.png"), dpi=150)
    plt.close()

    # Chart 2: Korelasi Fitur Numerik
    num_cols = ["luas_tanah_m2", "luas_bangunan_m2", "kamar_tidur", "kamar_mandi", "garasi", "jarak_stasiun_km", "harga_miliar_rp"]
    corr = df[num_cols].corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, square=True)
    plt.title("Matriks Korelasi Fitur Properti Jabodetabek", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "02_correlation_heatmap.png"), dpi=150)
    plt.close()


def main():
    db_path = os.path.join("data", "jabodetabek_houses.db")
    csv_path = os.path.join("data", "clean_jabodetabek_houses.csv")
    report_dir = os.path.join("reports")
    os.makedirs(report_dir, exist_ok=True)

    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database tidak ditemukan: {db_path}. Jalankan src/02_data_cleaning.py terlebih dahulu.")

    df = pd.read_csv(csv_path)
    sql_results = run_sql_analysis(db_path)
    save_static_charts(df, sql_results, os.path.join(report_dir, "figures"))

    summary_json = {k: v.to_dict(orient="records") for k, v in sql_results.items()}
    with open(os.path.join(report_dir, "sql_business_insights.json"), "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2, ensure_ascii=False)

    print("[OK] Analisis SQL & EDA selesai!")
    for name, table in sql_results.items():
        print(f"\n--- {name.upper()} ---")
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()

import json
import os
import pickle
import sqlite3
import subprocess
import sys
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Jabodetabek Housing Analytics", layout="wide")

DATA_CSV = os.path.join("data", "clean_jabodetabek_houses.csv")
DATA_DB = os.path.join("data", "jabodetabek_houses.db")
AUDIT_LOG = os.path.join("data", "cleaning_audit_log.json")
MODEL_PATH = os.path.join("models", "house_price_rf.pkl")
EVAL_PATH = os.path.join("reports", "model_evaluation.json")


def ensure_pipeline_ready():
    if not os.path.exists(DATA_CSV) or not os.path.exists(MODEL_PATH):
        with st.spinner("Menyiapkan dataset, membersihkan data, dan melatih model..."):
            scripts = [
                "src/01_collect_data.py",
                "src/02_data_cleaning.py",
                "src/03_eda_sql_analysis.py",
                "src/04_train_model.py",
            ]
            for script in scripts:
                subprocess.run([sys.executable, script], check=True)


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_CSV)


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def main():
    ensure_pipeline_ready()
    df = load_data()
    model = load_model()

    st.title("Analisis Pasar & Prediksi Harga Rumah Jabodetabek")
    st.markdown("Dashboard end-to-end untuk mengeksplorasi faktor penentu harga properti di **Jakarta, Bogor, Depok, Tangerang, dan Bekasi**.")

    st.sidebar.header("Filter Eksplorasi Data")
    selected_cities = st.sidebar.multiselect("Pilih Wilayah:", sorted(df["kota"].unique()), default=sorted(df["kota"].unique()))
    selected_certs = st.sidebar.multiselect("Jenis Sertifikat:", sorted(df["sertifikat"].unique()), default=sorted(df["sertifikat"].unique()))
    max_price = float(df["harga_miliar_rp"].max())
    price_range = st.sidebar.slider("Rentang Harga (Miliar Rp):", 0.0, round(max_price, 1), (0.0, round(max_price, 1)), step=0.5)

    filtered_df = df[
        (df["kota"].isin(selected_cities))
        & (df["sertifikat"].isin(selected_certs))
        & (df["harga_miliar_rp"].between(price_range[0], price_range[1]))
    ]

    if filtered_df.empty:
        st.warning("Tidak ada data yang cocok dengan filter.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Properti Valid", f"{len(filtered_df):,} unit")
    c2.metric("Median Harga Rumah", f"Rp {filtered_df['harga_miliar_rp'].median():.2f} M")
    c3.metric("Rata-rata Harga Tanah", f"Rp {filtered_df['harga_per_m2_tanah_juta'].mean():.2f} Jt/m2")
    c4.metric("Rata-rata LT / LB", f"{filtered_df['luas_tanah_m2'].mean():.0f} / {filtered_df['luas_bangunan_m2'].mean():.0f} m2")

    st.divider()
    tab1, tab2, tab3, tab4 = st.tabs(["Eksplorasi & Bisnis", "SQL & Audit Data", "Prediksi Harga", "Ringkasan Insight"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            city_df = filtered_df.groupby("kota", as_index=False)["harga_per_m2_tanah_juta"].mean().sort_values("harga_per_m2_tanah_juta")
            fig1 = px.bar(city_df, x="harga_per_m2_tanah_juta", y="kota", orientation="h", title="1. Rata-rata Harga Tanah per m2 (Juta Rp)")
            st.plotly_chart(fig1, use_container_width=True)
        with col2:
            fig2 = px.scatter(filtered_df, x="luas_tanah_m2", y="harga_miliar_rp", color="kota", size="luas_bangunan_m2", hover_data=["kecamatan", "sertifikat"], title="2. Luas Tanah & Bangunan vs Harga (Miliar Rp)", opacity=0.7)
            st.plotly_chart(fig2, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            fig3 = px.box(filtered_df, x="sertifikat", y="harga_per_m2_tanah_juta", color="sertifikat", title="3. Harga Tanah per m2 Berdasarkan Sertifikat")
            st.plotly_chart(fig3, use_container_width=True)
        with col4:
            tr_df = filtered_df.groupby(["akses_tol", "kategori_jarak_stasiun"], as_index=False)["harga_per_m2_tanah_juta"].mean()
            fig4 = px.bar(tr_df, x="kategori_jarak_stasiun", y="harga_per_m2_tanah_juta", color="akses_tol", barmode="group", title="4. Dampak Akses Tol & Jarak Stasiun")
            st.plotly_chart(fig4, use_container_width=True)

    with tab2:
        st.subheader("Dokumentasi Pembersihan Data (Audit Log)")
        if os.path.exists(AUDIT_LOG):
            with open(AUDIT_LOG, "r", encoding="utf-8") as f:
                audit = json.load(f)
            st.info(f"Awal: **{audit['initial_rows']} baris** ({audit['initial_missing']} null) -> Bersih: **{audit['final_rows']} baris**.")
            st.dataframe(pd.DataFrame(audit["steps"]), use_container_width=True)

        st.subheader("Interactive SQL Explorer (SQLite)")
        default_sql = "SELECT kota, kecamatan, COUNT(*) AS unit, ROUND(AVG(harga_miliar_rp), 2) AS avg_miliar, ROUND(AVG(harga_per_m2_tanah_juta), 2) AS avg_tanah_jt FROM houses GROUP BY kota, kecamatan HAVING COUNT(*) >= 10 ORDER BY avg_tanah_jt DESC LIMIT 15;"
        query = st.text_area("Query SQL (SELECT only):", value=default_sql, height=100)
        if st.button("Jalankan Query"):
            if query.strip().lower().startswith("select"):
                conn = sqlite3.connect(DATA_DB)
                try:
                    st.dataframe(pd.read_sql_query(query, conn), use_container_width=True)
                except Exception as exc:
                    st.error(f"Error: {exc}")
                finally:
                    conn.close()
            else:
                st.error("Hanya query SELECT yang diizinkan.")

    with tab3:
        st.subheader("Simulator Estimasi Harga (Random Forest)")
        if os.path.exists(EVAL_PATH):
            with open(EVAL_PATH, "r", encoding="utf-8") as f:
                ev = json.load(f)["metrics"]["Random Forest Regressor"]
            m1, m2, m3 = st.columns(3)
            m1.metric("R2 Score", f"{ev['R2_score']:.4f}")
            m2.metric("MAE", f"Rp {ev['MAE_juta_rp']:.1f} Juta")
            m3.metric("MAPE", f"{ev['MAPE_persen']:.2f}%")

        with st.form("pred_form"):
            f1, f2, f3 = st.columns(3)
            with f1:
                p_kota = st.selectbox("Kota", sorted(df["kota"].unique()))
                p_lt = st.number_input("Luas Tanah (m2)", 24, 1500, 120)
                p_lb = st.number_input("Luas Bangunan (m2)", 21, 1500, 110)
            with f2:
                p_kt = st.slider("Kamar Tidur", 1, 8, 3)
                p_km = st.slider("Kamar Mandi", 1, 7, 2)
                p_garasi = st.slider("Garasi", 0, 4, 1)
            with f3:
                p_cert = st.selectbox("Sertifikat", ["SHM", "HGB", "Lainnya"])
                p_cond = st.selectbox("Kondisi", ["Baru", "Bagus", "Butuh Renovasi"])
                p_tol = st.selectbox("Akses Tol", ["Ya", "Tidak"])
                p_sta = st.slider("Jarak Stasiun (km)", 0.3, 22.0, 3.5, 0.1)
            submit = st.form_submit_button("Hitung Estimasi")

        if submit:
            inp = pd.DataFrame([{"kota": p_kota, "luas_tanah_m2": p_lt, "luas_bangunan_m2": p_lb, "kamar_tidur": p_kt, "kamar_mandi": p_km, "garasi": p_garasi, "sertifikat": p_cert, "kondisi": p_cond, "akses_tol": p_tol, "jarak_stasiun_km": p_sta}])
            pred = float(model.predict(inp)[0])
            st.success(f"Estimasi Harga Pasar: **Rp {pred/1000:.2f} Miliar** (Rp {pred:,.1f} Juta)")

    with tab4:
        st.subheader("Ringkasan Insight Bisnis (Bahasa Non-Teknis)")
        st.markdown("1. **Lokasi Tetap Raja:** Harga tanah per m2 di Jakarta Selatan mencapai lebih dari **3x lipat** dibanding Bogor dan Bekasi.\n2. **Dimensi Fisik Dominan:** Luas tanah dan luas bangunan merupakan penggerak utama harga rumah dibanding jumlah kamar.\n3. **Premi Legalitas SHM:** Properti berstatus **SHM** memiliki valuasi per m2 lebih tinggi dibanding HGB atau surat lainnya.\n4. **Efek Konektivitas:** Rumah berjarak **<2 km dari stasiun** dengan **akses tol** mencatat premi harga yang konsisten.\n5. **Best Value Area:** **Tangerang** dan **Depok** menawarkan rasio luas bangunan terhadap harga yang paling kompetitif.")


if __name__ == "__main__":
    main()

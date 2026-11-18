"""
04_train_model.py
-----------------
Melatih dan mengevaluasi model prediksi harga rumah di Jabodetabek menggunakan
Linear Regression (baseline) dan Random Forest Regressor (ensemble) dengan
metrik evaluasi standar industri (MAE, RMSE, R2 Score).
"""

import json
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


FEATURE_COLS = [
    "kota",
    "luas_tanah_m2",
    "luas_bangunan_m2",
    "kamar_tidur",
    "kamar_mandi",
    "garasi",
    "sertifikat",
    "kondisi",
    "akses_tol",
    "jarak_stasiun_km",
]
TARGET_COL = "harga_juta_rp"


def train_and_evaluate(df: pd.DataFrame) -> dict:
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    categorical_cols = ["kota", "sertifikat", "kondisi", "akses_tol"]
    numeric_cols = [
        "luas_tanah_m2",
        "luas_bangunan_m2",
        "kamar_tidur",
        "kamar_mandi",
        "garasi",
        "jarak_stasiun_km",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    models = {
        "Linear Regression (Baseline)": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=200, max_depth=14, min_samples_leaf=2, random_state=42, n_jobs=-1
        ),
    }

    results = {}
    trained_pipes = {}
    best_model = None
    best_r2 = -np.inf

    for name, estimator in models.items():
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        trained_pipes[name] = pipe

        mae = float(mean_absolute_error(y_test, preds))
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        r2 = float(r2_score(y_test, preds))
        mape = float(np.mean(np.abs((y_test - preds) / y_test)) * 100)

        results[name] = {
            "MAE_juta_rp": round(mae, 2),
            "RMSE_juta_rp": round(rmse, 2),
            "MAPE_persen": round(mape, 2),
            "R2_score": round(r2, 4),
        }

        if r2 > best_r2:
            best_r2 = r2
            best_model = pipe

    # Ekstraksi Feature Importance dari Random Forest
    rf_pipe = trained_pipes["Random Forest Regressor"]
    ohe_features = (
        rf_pipe.named_steps["preprocessor"]
        .named_transformers_["cat"]
        .get_feature_names_out(categorical_cols)
        .tolist()
    )
    all_features = numeric_cols + ohe_features
    importances = rf_pipe.named_steps["model"].feature_importances_

    fi_df = (
        pd.DataFrame({"feature": all_features, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(10)
    )

    return {
        "metrics": results,
        "top_features": fi_df.to_dict(orient="records"),
        "best_pipeline": best_model,
    }


def main():
    csv_path = os.path.join("data", "clean_jabodetabek_houses.csv")
    model_dir = os.path.join("models")
    report_dir = os.path.join("reports")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset bersih tidak ditemukan: {csv_path}. Jalankan src/02_data_cleaning.py terlebih dahulu.")

    df = pd.read_csv(csv_path)
    output = train_and_evaluate(df)

    model_path = os.path.join(model_dir, "house_price_rf.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(output["best_pipeline"], f)

    eval_report = {
        "metrics": output["metrics"],
        "top_features": output["top_features"],
    }
    eval_path = os.path.join(report_dir, "model_evaluation.json")
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_report, f, indent=2, ensure_ascii=False)

    print("[OK] Pelatihan & Evaluasi Model Selesai!")
    for model_name, m in output["metrics"].items():
        print(f"  -> {model_name}: R2={m['R2_score']} | MAE=Rp {m['MAE_juta_rp']} Juta | MAPE={m['MAPE_persen']}%")
    print(f"     Model disimpan di: {model_path}")


if __name__ == "__main__":
    main()

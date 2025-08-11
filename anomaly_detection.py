import os
import glob
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


# ------------------------------
# Helper: load top features
# ------------------------------
def get_top_features(feature_ranking_csv, top_n=7):
    df = pd.read_csv(feature_ranking_csv)
    return df['feature'].head(top_n).tolist()


# ------------------------------
# TRAINING - single model only
# ------------------------------
def train_time_series_anomaly_model(
    base_dir,
    feature_ranking_csv,
    model_path,
    contamination=0.05,
    top_n=7
):
    """
    Train a single IsolationForest model on top N features from all KPI CSVs in base_dir.
    Saves model and feature list to disk.
    """
    # Load all KPI data
    kpi_files = glob.glob(os.path.join(base_dir, "run_*", "kpis.csv"))
    if not kpi_files:
        raise FileNotFoundError("No kpis.csv files found in run folders.")

    dfs = [pd.read_csv(f) for f in kpi_files]
    df_all = pd.concat(dfs, ignore_index=True)

    if "timestamp" not in df_all.columns:
        raise KeyError("Timestamp column missing in KPI data.")

    df_all["timestamp"] = pd.to_datetime(df_all["timestamp"])
    df_all.sort_values("timestamp", inplace=True)
    df_all.reset_index(drop=True, inplace=True)

    # Select top features
    top_features = get_top_features(feature_ranking_csv, top_n=top_n)
    feature_cols = [col for col in top_features if col in df_all.columns]
    df_features = df_all[feature_cols].fillna(method='ffill').fillna(method='bfill').fillna(0)

    # Train single model
    model = IsolationForest(contamination=contamination, random_state=42)
    model.fit(df_features)

    # Save model + metadata
    joblib.dump({
        "mode": "single",
        "model": model,
        "feature_cols": feature_cols,
        "contamination": contamination
    }, model_path)

    print(f"Saved single IsolationForest model to {model_path}")
    return model_path


# ------------------------------
# INFERENCE - single model only
# ------------------------------
def detect_anomalies_with_trained_model(
    kpi_csv_path,
    model_path,
    output_csv=None
):
    """
    Detect anomalies in a KPI CSV using a saved single IsolationForest model.
    """
    if not os.path.exists(kpi_csv_path):
        raise FileNotFoundError(f"KPI CSV not found: {kpi_csv_path}")

    saved = joblib.load(model_path)
    if saved.get("mode") != "single":
        raise ValueError("The saved model is not in 'single' mode.")

    model = saved["model"]
    feature_cols = saved["feature_cols"]

    df = pd.read_csv(kpi_csv_path)
    if "timestamp" not in df.columns:
        raise KeyError("Timestamp column missing in KPI data.")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Features for prediction
    df_features = df[feature_cols].fillna(method='ffill').fillna(method='bfill').fillna(0)

    # Get scores and classify anomalies
    scores = model.decision_function(df_features)
    threshold = np.median(scores)
    df["anomaly"] = (scores < threshold).astype(int)

    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"Anomaly detection results saved to {output_csv}")

    return df


# # ------------------------------
# # Example usage
# # ------------------------------
# if __name__ == "__main__":
#     data_dir = "output_data"               # Where run_* folders live
#     model_dir = "model"
#     os.makedirs(model_dir, exist_ok=True)

#     feature_ranking_csv = "feature_ranking.csv"
#     model_path = os.path.join(model_dir, "iforest_single.pkl")
#     output_csv = os.path.join(data_dir, "inference_results.csv")

#     # Train
#     train_time_series_anomaly_model(
#         base_dir=data_dir,
#         feature_ranking_csv=feature_ranking_csv,
#         model_path=model_path,
#         contamination=0.05,
#         top_n=7
#     )

#     # Inference
#     detect_anomalies_with_trained_model(
#         kpi_csv_path=os.path.join(data_dir, "run_001", "kpis.csv"),
#         model_path=model_path,
#         output_csv=output_csv
#     )

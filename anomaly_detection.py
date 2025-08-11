import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from datetime import datetime
import os
import glob

# Simple time-series anomaly detection for telecom KPIs

def detect_anomalies_in_kpis(kpi_csv, output_csv=None, contamination=0.05):
    """
    Detect anomalies in the KPI time-series data using Isolation Forest.
    Adds an 'anomaly' column to the output CSV (1=anomaly, 0=normal).
    """
    if not os.path.exists(kpi_csv):
        raise FileNotFoundError(f"KPI CSV not found: {kpi_csv}")
    
    df = pd.read_csv(kpi_csv)
    # Select numeric columns for anomaly detection
    features = ['latency_ms', 'throughput_mbps', 'packet_loss_percent', 'jitter_ms', 'rtt_ms',
                'call_setup_success_rate', 'data_session_success_rate', 'handover_success_rate', 'call_drop_rate',
                'cpu_utilization_percent', 'memory_utilization_percent', 'bandwidth_utilization_percent', 'disk_utilization_percent']
    feature_cols = [col for col in features if col in df.columns]
    df_features = df[feature_cols].fillna(0)

    # Fit Isolation Forest
    model = IsolationForest(contamination=contamination, random_state=42)
    df['anomaly'] = model.fit_predict(df_features)
    # IsolationForest: -1=anomaly, 1=normal. Convert to 1/0
    df['anomaly'] = (df['anomaly'] == -1).astype(int)

    # Save or return
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"Anomaly detection results saved to {output_csv}")
    return df

def get_top_features(feature_ranking_csv, top_n=7):
    df_features = pd.read_csv(feature_ranking_csv)
    return df_features['feature'].head(top_n).tolist()

def detect_time_series_anomalies_on_top_features_all_runs(base_dir, feature_ranking_csv, output_csv=None, window_size=50, contamination=0.05, top_n=7):
    """
    Detect time series anomalies on top N features extracted from feature importance
    across all KPI CSV files in all run folders.
    """
    # Find all KPI CSV files
    kpi_files = glob.glob(os.path.join(base_dir, "run_*", "kpis.csv"))
    if not kpi_files:
        raise FileNotFoundError("No kpis.csv files found in run folders.")
    
    # Load all KPI files and concatenate
    dfs = []
    for f in kpi_files:
        df = pd.read_csv(f)
        dfs.append(df)
    df_all = pd.concat(dfs, ignore_index=True)
    
    if 'timestamp' not in df_all.columns:
        raise KeyError("Timestamp column missing in KPI data.")
    
    df_all['timestamp'] = pd.to_datetime(df_all['timestamp'])
    df_all.sort_values('timestamp', inplace=True)
    df_all.reset_index(drop=True, inplace=True)
    
    # Get top features
    top_features = get_top_features(feature_ranking_csv, top_n=top_n)
    print(f"Using top {top_n} features for anomaly detection: {top_features}")

    # Filter columns by top features, drop rows with missing values on these features
    feature_cols = [col for col in top_features if col in df_all.columns]
    df_features = df_all[feature_cols].fillna(method='ffill').fillna(method='bfill').fillna(0)

    anomaly_scores = np.zeros(len(df_all))

    # Rolling window anomaly detection
    for start in range(0, len(df_all) - window_size + 1):
        window_data = df_features.iloc[start:start + window_size]
        model = IsolationForest(contamination=contamination, random_state=42)
        model.fit(window_data)
        scores = model.decision_function(window_data)  # Higher is normal, lower is anomaly

        # Assign anomaly scores: keep minimum score if overlapping windows
        for i, idx in enumerate(range(start, start + window_size)):
            if anomaly_scores[idx] == 0:
                anomaly_scores[idx] = scores[i]
            else:
                anomaly_scores[idx] = min(anomaly_scores[idx], scores[i])

    # Threshold: median score (adjustable)
    threshold = np.median(anomaly_scores)
    df_all['anomaly'] = (anomaly_scores < threshold).astype(int)

    if output_csv:
        df_all.to_csv(output_csv, index=False)
        print(f"Anomaly detection results saved to {output_csv}")

    return df_all


if __name__ == "__main__":
    base_dir = "output_data"
    feature_ranking_csv = os.path.join(os.getcwd(), "feature_ranking.csv")
    output_csv = os.path.join(base_dir, "all_runs_kpis_top_features_time_series_anomalies.csv")

    detect_time_series_anomalies_on_top_features_all_runs(
        base_dir, feature_ranking_csv, output_csv,
        window_size=50, contamination=0.05, top_n=7
    )
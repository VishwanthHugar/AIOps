from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
import pandas as pd
import numpy as np
import time

def normalize_columns(df, columns):
    """
    Min-Max normalize specified columns to range [0,1].
    """
    df_norm = df.copy()
    for col in columns:
        print(f"Normalizing column: {col}")
        if col in df_norm.columns:
            min_val = df_norm[col].min()
            max_val = df_norm[col].max()
            print(f"  Min: {min_val}, Max: {max_val}")
            # Avoid division by zero
            if pd.isna(min_val) or pd.isna(max_val):
                print(f"  Skipping normalization for {col} due to NaN values.")
                df_norm[col] = 0.0
            if max_val > min_val:
                df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
            else:
                df_norm[col] = 0.0  # If no variation, set to 0
    return df_norm

def push_metrics_batch_to_pushgateway(
    df_batch, pushgateway_url='http://localhost:9091', job_name='telecom_kpis'
):
    registry = CollectorRegistry()

    metrics = [
        'latency_ms', 'throughput_mbps', 'packet_loss_percent', 'jitter_ms', 'rtt_ms',
        'call_setup_success_rate', 'data_session_success_rate', 'handover_success_rate',
        'call_drop_rate', 'cpu_utilization_percent', 'memory_utilization_percent',
        'bandwidth_utilization_percent', 'disk_utilization_percent', 'anomaly'
    ]

    gauges = {}
    for metric in metrics:
        gauges[metric] = Gauge(f"kpi_{metric}", f"KPI metric {metric}", ['cell_id'], registry=registry)

    for _, row in df_batch.iterrows():
        cell_id = str(row['cell_id'])
        for metric in metrics:
            if metric in row and pd.notnull(row[metric]):
                gauges[metric].labels(cell_id=cell_id).set(float(row[metric]))

    push_to_gateway(pushgateway_url, job=job_name, registry=registry)
    print(f"Pushed batch of {len(df_batch)} records to Pushgateway at {pushgateway_url}")

def push_in_batches(df, batch_size=20, delay_sec=5, pushgateway_url='http://localhost:9091', job_name='telecom_kpis'):
    metrics = [
        'latency_ms', 'throughput_mbps', 'packet_loss_percent', 'jitter_ms', 'rtt_ms',
        'call_setup_success_rate', 'data_session_success_rate', 'handover_success_rate',
        'call_drop_rate', 'cpu_utilization_percent', 'memory_utilization_percent',
        'bandwidth_utilization_percent', 'disk_utilization_percent'
    ]

    # Normalize metrics (exclude 'anomaly' which is already binary)
    df_norm = normalize_columns(df, metrics)

    total_rows = len(df_norm)
    for start_idx in range(0, total_rows, batch_size):
        batch_df = df_norm.iloc[start_idx:start_idx + batch_size]
        push_metrics_batch_to_pushgateway(batch_df, pushgateway_url, job_name)
        time.sleep(delay_sec)

if __name__ == "__main__":
    anomaly_csv = "kpis_with_time_series_anomalies.csv"
    df_all = pd.read_csv(anomaly_csv)
    df_all['timestamp'] = pd.to_datetime(df_all['timestamp'])

    push_in_batches(df_all, batch_size=20, delay_sec=5)

import os
import time
import glob
import click
import pandas as pd
import numpy as np
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
from anomaly_detection import detect_anomalies_with_trained_model  # from your anomaly script

# ------------------------------
# Helpers
# ------------------------------
def normalize_columns(df, columns):
    df_norm = df.copy()
    for col in columns:
        if col in df_norm.columns:
            min_val = df_norm[col].min()
            max_val = df_norm[col].max()
            if pd.isna(min_val) or pd.isna(max_val):
                df_norm[col] = 0.0
            elif max_val > min_val:
                df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
            else:
                df_norm[col] = 0.0
    return df_norm


def push_metrics_batch(df_batch, pushgateway_url, job_name):
    registry = CollectorRegistry()
    metrics = [
        'latency_ms', 'throughput_mbps', 'packet_loss_percent', 'jitter_ms', 'rtt_ms',
        'call_setup_success_rate', 'data_session_success_rate', 'handover_success_rate',
        'call_drop_rate', 'cpu_utilization_percent', 'memory_utilization_percent',
        'bandwidth_utilization_percent', 'disk_utilization_percent', 'anomaly'
    ]

    gauges = {
        metric: Gauge(f"kpi_{metric}", f"KPI metric {metric}", ['cell_id'], registry=registry)
        for metric in metrics
    }

    for _, row in df_batch.iterrows():
        cell_id = str(row.get('cell_id', 'unknown'))
        for metric in metrics:
            if metric in row and pd.notnull(row[metric]):
                gauges[metric].labels(cell_id=cell_id).set(float(row[metric]))

    push_to_gateway(pushgateway_url, job=job_name, registry=registry)
    print(f"Pushed {len(df_batch)} rows to Pushgateway")


def push_in_batches(df, batch_size, delay_sec, pushgateway_url, job_name):
    metrics = [
        'latency_ms', 'throughput_mbps', 'packet_loss_percent', 'jitter_ms', 'rtt_ms',
        'call_setup_success_rate', 'data_session_success_rate', 'handover_success_rate',
        'call_drop_rate', 'cpu_utilization_percent', 'memory_utilization_percent',
        'bandwidth_utilization_percent', 'disk_utilization_percent'
    ]

    df_norm = normalize_columns(df, metrics)

    for start_idx in range(0, len(df_norm), batch_size):
        push_metrics_batch(df_norm.iloc[start_idx:start_idx + batch_size],
                           pushgateway_url, job_name)
        time.sleep(delay_sec)


def get_latest_kpi_file(data_dir):
    runs = sorted(glob.glob(os.path.join(data_dir, "run_*")), reverse=True)
    if not runs:
        raise FileNotFoundError(f"No 'run_*' folders found in {data_dir}")
    latest_run = runs[0]
    kpi_path = os.path.join(latest_run, "kpis.csv")
    if not os.path.exists(kpi_path):
        raise FileNotFoundError(f"No kpis.csv in {latest_run}")
    return kpi_path

# ------------------------------
# CLI Commands
# ------------------------------
@click.command()
@click.option("--data-dir", required=True,
              type=click.Path(exists=True, file_okay=False),
              help="Directory containing run_* subfolders with KPI CSVs")
@click.option("--model-path", required=True,
              type=click.Path(exists=True, dir_okay=False),
              help="Path to the trained model .pkl file")
@click.option("--run-file", type=click.Path(exists=True, dir_okay=False),
              help="Specific KPI CSV file to process (if omitted, takes latest run_* file)")
@click.option("--batch-size", default=20, show_default=True, type=int,
              help="Number of rows to push per batch")
@click.option("--delay-sec", default=5, show_default=True, type=int,
              help="Delay in seconds between batches")
@click.option("--pushgateway-url", default="http://localhost:9091", show_default=True,
              help="Prometheus Pushgateway URL")
@click.option("--job-name", default="telecom_kpis", show_default=True,
              help="Job name for metrics in Pushgateway")
def main(data_dir, model_path, run_file, batch_size, delay_sec, pushgateway_url, job_name):
    """Run anomaly inference on KPI CSV and push to Prometheus Pushgateway."""
    # Determine which KPI file to process
    if run_file:
        kpi_path = run_file
    else:
        kpi_path = get_latest_kpi_file(data_dir)
    print(f"Using KPI file: {kpi_path}")

    # Run anomaly detection inference
    df_with_anomalies = detect_anomalies_with_trained_model(
        kpi_csv_path=kpi_path,
        model_path=model_path
    )

    print(f"Detected anomalies in {len(df_with_anomalies)} rows, columns: {df_with_anomalies.columns.tolist()}")
    # Push in batches
    push_in_batches(df_with_anomalies, batch_size, delay_sec, pushgateway_url, job_name)


if __name__ == "__main__":
    main()


# python push_metrics_cli.py \
#     --data-dir output_data \
#     --model-path model/rolling_iforest_models.pkl \
#     --run-file output_data/run_010/kpis.csv \
#     --batch-size 10 \
#     --delay-sec 2 \
#     --pushgateway-url http://pushgw.local:9091

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import subprocess
import datetime


def filter_by_time(df, start_dt, end_dt):
    if 'timestamp' not in df.columns:
        return df
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        filtered = df[(df['timestamp'] >= start_dt) & (df['timestamp'] <= end_dt)]
        return filtered
    except Exception:
        return df

# Helper to get latest run folder
def get_latest_run_folder(base_dir='output_data'):
    if not os.path.exists(base_dir):
        return None
    runs = [d for d in os.listdir(base_dir) if d.startswith('run_')]
    if not runs:
        return None
    latest = sorted(runs)[-1]
    return os.path.join(base_dir, latest)

# Load data
def load_data(run_folder):
    kpis, alarms, logs = None, None, None
    if run_folder:
        kpi_path = os.path.join(run_folder, 'kpis.csv')
        alarm_path = os.path.join(run_folder, 'alarms.csv')
        log_path = os.path.join(run_folder, 'logs.csv')
        if os.path.exists(kpi_path):
            kpis = pd.read_csv(kpi_path)
        if os.path.exists(alarm_path):
            alarms = pd.read_csv(alarm_path)
        if os.path.exists(log_path):
            logs = pd.read_csv(log_path)
    return kpis, alarms, logs


# Function to normalize values between 0 and 1
def normalize_df(df):
    return (df - df.min()) / (df.max() - df.min())

# Function to detect anomalies using z-score
def detect_anomalies(series, threshold=3):
    z_scores = (series - series.mean()) / series.std()
    return np.abs(z_scores) > threshold

# Function to plot a single metric with anomalies highlighted
def plot_metric_with_anomalies(df, metric, vis_type, threahold=3):
    #anomalies = detect_anomalies(df[metric], threshold)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    
    if vis_type == "Line":
        ax.plot(df['timestamp'], df[metric], label=metric)
    elif vis_type == "Area":
        ax.fill_between(df['timestamp'], df[metric], alpha=0.3)
    elif vis_type == "Scatter":
        ax.scatter(df['timestamp'], df[metric], label=metric)
    
    # Highlight anomalies in red
    anomalies = df['anomaly'] == 1
    ax.scatter(df['timestamp'][anomalies], df[metric][anomalies], color='red', label="Anomaly", zorder=5)
    
    ax.set_title(f"{metric} (with Anomalies)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Normalized Value")
    ax.legend()
    ax.grid(True)
    
    return fig
    #st.pyplot(fig)

# Call ollama for RCA
def perform_rca(query, context):
    # context: string with relevant data
    # prompt = f"You are an AI operational assistant. Only output the answer, do not show your thinking process or reasoning.\nQuery: {query}\nData:\n{context}\nAnswer:"
    prompt = (
        "You are an AI operational assistant.\n"
        "Your task is to answer concisely without showing any reasoning or thought process.\n"
        "Only output the final answer.\n"
        "Do NOT explain your steps.\n"
        "Do NOT show your reasoning.\n"
        "Respond with only the answer below:\n\n"
        f"Query: {query}\n\n"
        f"Data:\n{context}\n\n"
        "Answer:"
    )

    try:
        result = subprocess.run(
            ["ollama", "run", "llama3.2:1b", prompt], #deepseek-r1:1.5b
            capture_output=True,
            text=True,
            timeout=60,
        )
        if not result.stdout.strip() and result.stderr:
            return f"Ollama error: {result.stderr}"
        return result.stdout
    except Exception as e:
        return f"Error running ollama: {e}"

def get_latest_kpi_file(data_dir):
    runs = sorted(glob.glob(os.path.join(data_dir, "run_*")), reverse=True)
    if not runs:
        raise FileNotFoundError(f"No 'run_*' folders found in {data_dir}")
    latest_run = runs[0]
    kpi_path = os.path.join(latest_run, "kpis.csv")
    if not os.path.exists(kpi_path):
        raise FileNotFoundError(f"No kpis.csv in {latest_run}")
    return kpi_path


import streamlit as st
import pandas as pd
import os
import subprocess
import datetime

from utils import read_config, get_config_value

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
def get_latest_run_folder(base_dir='../output_data'):
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
            ["ollama", "run", "llama3.2:1b", prompt], #llama3.2:1b, deepseek-r1:1.5b
            capture_output=True,
            text=True,
            timeout=60,
        )
        if not result.stdout.strip() and result.stderr:
            return f"Ollama error: {result.stderr}"
        return result.stdout
    except Exception as e:
        return f"Error running ollama: {e}"

# Streamlit UI
st.title('AI TelecomOps Conversational Assistant')

config = read_config("config.txt")
data_str = config.get("simulated_data")

run_folder = get_latest_run_folder(data_str)
kpis, alarms, logs = load_data(run_folder)

st.sidebar.header('Data Preview')
if kpis is not None:
    st.sidebar.subheader('KPIs')
    st.sidebar.dataframe(kpis.head())
if alarms is not None:
    st.sidebar.subheader('Alarms')
    st.sidebar.dataframe(alarms.head())
if logs is not None:
    st.sidebar.subheader('Logs')
    st.sidebar.dataframe(logs.head())


st.sidebar.header('Select Time Range for Data Context')

# Default time range: last 24 hours or full data range
default_start = None
default_end = None

if alarms is not None and not alarms.empty:
    # Try to parse datetime from alarms if possible
    try:
        alarms['timestamp'] = pd.to_datetime(alarms['timestamp'])
        default_start = alarms['timestamp'].min()
        default_end = alarms['timestamp'].max()
    except Exception:
        pass

start_time = st.sidebar.date_input('Start date', value=default_start.date() if default_start else datetime.date.today())
end_time = st.sidebar.date_input('End date', value=default_end.date() if default_end else datetime.date.today())

# Optional time input (hours and minutes) for more granularity
start_time_dt = st.sidebar.time_input('Start time', value=datetime.time(0, 0))
end_time_dt = st.sidebar.time_input('End time', value=datetime.time(23, 59))

# Combine date and time into full datetime objects
start_datetime = datetime.datetime.combine(start_time, start_time_dt)
end_datetime = datetime.datetime.combine(end_time, end_time_dt)


st.write('Ask a question about the network, alarms, or request a root cause analysis (RCA).')
user_input = st.text_input('Your question or RCA request:')

if st.button('Ask') and user_input:
    # For RCA, use all alarms+logs as context (could be improved)
    context = ''
    if alarms is not None and not alarms.empty:
        context += alarms.tail(10).to_string(index=False) + '\n'
        #context += alarms.to_string(index=False) + '\n'
    if logs is not None and not logs.empty:
        context += logs.tail(10).to_string(index=False) + '\n'
        #context += logs.to_string(index=False) + '\n'


    # if alarms is not None and not alarms.empty:
    #     filtered_alarms = filter_by_time(alarms, start_datetime, end_datetime)
    #     if not filtered_alarms.empty:
    #         context += filtered_alarms.to_string(index=False) + '\n'

    # print(f"Context after filtering alarms: {len(context)} ")

    # if logs is not None and not logs.empty:
    #     filtered_logs = filter_by_time(logs, start_datetime, end_datetime)
    #     if not filtered_logs.empty:
    #         context += filtered_logs.to_string(index=False) + '\n'

    print(f"Context after filtering logs: {len(context)}")

    if not context.strip():
        context = 'No recent data.'
    st.info('Performing RCA using ollama...')

    rca_result = perform_rca(user_input, context)
    if rca_result.strip():
        st.success(rca_result)
    else:
        st.warning('No RCA result returned. Please check if the ollama model is running and responding.')

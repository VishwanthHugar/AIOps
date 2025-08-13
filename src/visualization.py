import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

    st.pyplot(fig)

# Streamlit App
st.title("📊 Time Series Visualization with Anomalies")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # Ensure timestamp column exists
    if 'timestamp' not in df.columns:
        st.error("CSV must contain a 'timestamp' column.")
    else:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')

        # Date range selection
        min_date, max_date = df['timestamp'].min(), df['timestamp'].max()
        start_date = st.date_input("Start Date", min_date)
        end_date = st.date_input("End Date", max_date)
        
        if start_date and end_date:
            df = df[(df['timestamp'] >= pd.Timestamp(start_date)) & 
                    (df['timestamp'] <= pd.Timestamp(end_date))]

        # Normalize numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = normalize_df(df[numeric_cols])

        # Visualization type selection
        vis_type = st.selectbox("Select Visualization Type", ["Line", "Area", "Scatter"])

        # Plot each metric in separate row
        for metric in numeric_cols:
            plot_metric_with_anomalies(df, metric, vis_type)
else:
    st.info("Please upload a CSV file to visualize the data.")

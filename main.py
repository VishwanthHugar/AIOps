import streamlit as st
import pandas as pd
import numpy as np
import datetime
import os

# Import implementations from other files
from feature_selections import FeatureImportanceETL
from anomaly_detection import detect_time_series_anomalies_on_top_features_all_runs
from rca_genai import perform_rca, filter_by_time, get_latest_run_folder, load_data
from visualization import normalize_df, detect_anomalies, plot_metric_with_anomalies


def main():
    st.title("📡 Telecom KPI AIOps Dashboard")

    # Step 1: Feature Importance ETL
    st.sidebar.header("Step 1: Feature Importance ETL")
    if st.sidebar.button("Run Feature Importance ETL"):
        try:
            etl = FeatureImportanceETL()
            etl.run()
        except Exception as e:
            st.error(f"ETL failed: {e}")

    # Step 2: Anomaly Detection on Top Features
    st.sidebar.header("Step 2: Time-Series Anomaly Detection")
    if st.sidebar.button("Run Anomaly Detection on Top Features"):
        try:
            base_dir = "output_data"
            feature_ranking_csv = "feature_ranking.csv"
            if not os.path.exists(feature_ranking_csv):
                st.error("Feature ranking CSV not found. Run ETL first.")
            else:
                detect_time_series_anomalies_on_top_features_all_runs(
                    base_dir, feature_ranking_csv, window_size=50, contamination=0.05, top_n=7
                )
                st.success("Anomaly detection completed.")
        except Exception as e:
            st.error(f"Anomaly detection failed: {e}")

    # Step 3: RCA Query
    st.sidebar.header("Step 3: Root Cause Analysis (RCA)")
    run_folder = get_latest_run_folder()
    kpis, alarms, logs = load_data(run_folder)

    if alarms is not None and logs is not None:
        st.sidebar.subheader("RCA Data Preview")
        st.sidebar.write("Alarms sample:")
        st.sidebar.dataframe(alarms.head())
        st.sidebar.write("Logs sample:")
        st.sidebar.dataframe(logs.head())

    default_start, default_end = None, None
    if alarms is not None and not alarms.empty:
        try:
            alarms['timestamp'] = pd.to_datetime(alarms['timestamp'])
            default_start = alarms['timestamp'].min().date()
            default_end = alarms['timestamp'].max().date()
        except Exception:
            default_start = datetime.date.today()
            default_end = datetime.date.today()

    # start_date = st.sidebar.date_input("Start date", value=default_start or datetime.date.today())
    # end_date = st.sidebar.date_input("End date", value=default_end or datetime.date.today())
    start_date = st.sidebar.date_input("Start date", value=default_start or datetime.date.today(), key="start_date")
    end_date = st.sidebar.date_input("End date", value=default_end or datetime.date.today(), key="end_date")
    
    # start_time = st.sidebar.time_input("Start time", value=datetime.time(0, 0))
    # end_time = st.sidebar.time_input("End time", value=datetime.time(23, 59))

    start_time = st.sidebar.time_input("Start time", value=datetime.time(0, 0), key="start_time")
    end_time = st.sidebar.time_input("End time", value=datetime.time(12, 0), key="end_time")

    start_datetime = datetime.datetime.combine(start_date, start_time)
    end_datetime = datetime.datetime.combine(end_date, end_time)

    st.header("Ask a question or request RCA")
    # user_input = st.text_input("Your question or RCA request:")
    user_input = st.text_input("Your question or RCA request:", key="user_input")


    # if st.button("Ask") and user_input:
    if st.button("Ask", key="ask_button_1"):

        context = ""
        if alarms is not None and not alarms.empty:
            context += alarms.tail(10).to_string(index=False) + '\n'
        if logs is not None and not logs.empty:
            context += logs.tail(10).to_string(index=False) + '\n'

        # if alarms is not None and not alarms.empty:
        #     filtered_alarms = filter_by_time(alarms, start_datetime, end_datetime)
        #     if not filtered_alarms.empty:
        #         context += filtered_alarms.to_string(index=False) + "\n"
        # if logs is not None and not logs.empty:
        #     filtered_logs = filter_by_time(logs, start_datetime, end_datetime)
        #     if not filtered_logs.empty:
        #         context += filtered_logs.to_string(index=False) + "\n"
        if not context.strip():
            context = "No recent data."

        st.info("Performing RCA using Ollama...")
        rca_result = perform_rca(user_input, context)
        if rca_result:
            st.success(rca_result)
        else:
            st.warning("No RCA result returned. Check if Ollama model is running.")

    # Visualization
    st.header("Visualize Historical KPI Data with Anomalies")
    uploaded_file = st.file_uploader("Upload KPI CSV for Visualization (with 'timestamp' column)", type=["csv"])

    if uploaded_file:
        df_vis = pd.read_csv(uploaded_file)
        if 'timestamp' not in df_vis.columns:
            st.error("CSV must contain a 'timestamp' column.")
            return

        df_vis['timestamp'] = pd.to_datetime(df_vis['timestamp'])
        df_vis = df_vis.sort_values('timestamp')

        min_date, max_date = df_vis['timestamp'].min().date(), df_vis['timestamp'].max().date()
        start_viz = st.date_input("Start Date for Visualization", min_date, min_value=min_date, max_value=max_date)
        end_viz = st.date_input("End Date for Visualization", max_date, min_value=min_date, max_value=max_date)

        if start_viz > end_viz:
            st.error("Start date must be before end date.")
            return

        df_filtered = df_vis[(df_vis['timestamp'] >= pd.Timestamp(start_viz)) & (df_vis['timestamp'] <= pd.Timestamp(end_viz))]

        if df_filtered.empty:
            st.warning("No data available for selected date range.")
            return

        numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            st.error("No numeric columns found for visualization.")
            return

        df_filtered[numeric_cols] = normalize_df(df_filtered[numeric_cols])

        vis_type = st.selectbox("Select Visualization Type", ["Line", "Area", "Scatter"])
        threshold = st.slider("Anomaly Detection z-score threshold", 1.0, 5.0, 3.0, 0.1)
        selected_metrics = st.multiselect("Select metrics to visualize", options=numeric_cols, default=numeric_cols)

        if not selected_metrics:
            st.warning("Please select at least one metric to visualize.")
            return

        for metric in selected_metrics:
            plot_metric_with_anomalies(df_filtered, metric, vis_type, threshold)
    else:
        st.info("Upload a KPI CSV file to visualize historical data and anomalies.")


if __name__ == "__main__":
    main()

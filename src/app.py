import streamlit as st
import pandas as pd
import numpy as np
import datetime
import os
from datetime import datetime

# Import implementations from other files
from feature_selections import FeatureImportanceETL
from anomaly_detection import detect_anomalies_with_trained_model
from utils import perform_rca, get_latest_run_folder, load_data, get_latest_kpi_file
from utils import normalize_df, plot_metric_with_anomalies, read_config, get_config_value


def main():
    config = read_config("config.txt")

    st.title("📡 Telecom KPI AIOps Dashboard")

    #df_with_anomalies = None
    # Step 1: Feature Importance ETL
    st.sidebar.header("Step 1: Feature Importance ETL")
    if st.sidebar.button("Run Feature Importance ETL"):
        try:
            etl = FeatureImportanceETL()
            etl.run()
            st.success("Feature importance ETL completed.")
        except Exception as e:
            st.error(f"ETL failed: {e}")

    # Step 2: Anomaly Detection on Top Features
    st.sidebar.header("Step 2: Time-Series Anomaly Detection")
    if st.sidebar.button("Run Anomaly Detection on Top Features"):
        try:
            input_file_path = config.get("test_data")
            
            subfolders = [os.path.join(input_file_path, f) for f in os.listdir(input_file_path)
            if os.path.isdir(os.path.join(input_file_path, f))]
            kpi_file_folder = max(subfolders, key=os.path.getmtime) if subfolders else None

            lastes_kpi = kpi_file_folder + "/kpis.csv"
            kpi_file_path = os.path.join(os.getcwd(), lastes_kpi)
            print("Latest subfolder:", kpi_file_path)
 
            model_path = config.get("model_path")
            feature_ranking_csv = config.get("feature_ranking_csv")
            if not os.path.exists(feature_ranking_csv):
                st.error("Feature ranking CSV not found. Run ETL first.")
            else:
                df_with_anomalies = detect_anomalies_with_trained_model(
                    kpi_file_path, model_path
                )
                # Current timestamp in YYYYMMDD_HHMMSS format
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                # File path with suffix
                anomaly_file_path = os.path.join(os.getcwd(), "../result" ,f"df_with_anomalies_{timestamp}.csv")

                # Save DataFrame
                df_with_anomalies.to_csv(anomaly_file_path, index=False)

                st.success("Anomaly detection completed. Results saved to: " + anomaly_file_path)
        except Exception as e:
            st.error(f"Anomaly detection failed: {e}")

    # Step 3: RCA Query
    st.sidebar.header("Step 3: Root Cause Analysis (RCA)")
    data_str = config.get("simulated_data")

    run_folder = get_latest_run_folder(data_str)
    kpis, alarms, logs = load_data(run_folder)

    print("runfolder: ", run_folder)
    if alarms is not None and logs is not None:
        st.sidebar.subheader("RCA Data Preview")
        st.sidebar.write("Alarms sample:")
        st.sidebar.dataframe(alarms.head())
        st.sidebar.write("Logs sample:")
        st.sidebar.dataframe(logs.head())

    st.header("Ask a question or request RCA")
    user_input = st.text_input("Your question or RCA request:", key="user_input")

    if st.button("Ask", key="ask_button_1"):

        context = ""
        if alarms is not None and not alarms.empty:
            context += alarms.tail(10).to_string(index=False) + '\n'
        if logs is not None and not logs.empty:
            context += logs.tail(10).to_string(index=False) + '\n'

        if not context.strip():
            context = "No recent data."

        st.info("Performing RCA using Ollama...")
        rca_result = perform_rca(user_input, context)
        if rca_result:
            st.success(rca_result)
        else:
            st.warning("No RCA result returned. Check if Ollama model is running.")

    # 4 Visualization
    st.header("Visualize Historical KPI Data with Anomalies")
    uploaded_file = st.file_uploader("Upload KPI CSV for Visualization (with 'timestamp' column)", type=["csv"])

    if uploaded_file:
        df_vis = pd.read_csv(uploaded_file)
        if 'timestamp' not in df_vis.columns:
            st.error("CSV must contain a 'timestamp' column.")
            st.stop()

        df_vis['timestamp'] = pd.to_datetime(df_vis['timestamp'])
        df_vis = df_vis.sort_values('timestamp')

        # Removed date range selection and filtering, use full data
        df_filtered = df_vis.copy()

        if df_filtered.empty:
            st.warning("No data available in the uploaded file.")
            st.stop()

        numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            st.error("No numeric columns found for visualization.")
            st.stop()

        df_filtered[numeric_cols] = normalize_df(df_filtered[numeric_cols])

        vis_type = st.selectbox("Select Visualization Type", ["Line", "Area", "Scatter"])
        selected_metrics = st.multiselect("Select metrics to visualize", options=numeric_cols, default=numeric_cols)

        if not selected_metrics:
            st.warning("Please select at least one metric to visualize.")
            st.stop()

        for metric in selected_metrics:
            fig = plot_metric_with_anomalies(df_filtered, metric, vis_type)
            st.pyplot(fig)
    else:
        st.info("Upload a KPI CSV file to visualize historical data and anomalies.")



if __name__ == "__main__":
    main()

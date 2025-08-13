import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime

# Import implementations from other files
from feature_selections import FeatureImportanceETL
from anomaly_detection import detect_anomalies_with_trained_model
from utils import perform_rca, load_data, normalize_df, plot_metric_with_anomalies
from utils import read_config
from push_to_prometheus_click import push_metrics_batch
     
def main():
    st.title("📡 Telecom KPI AIOps Dashboard")

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
            config = read_config("config.txt")
            input_file_path = config.get("test_data")
            
            # Get latest subfolder
            subfolders = [os.path.join(input_file_path, f) for f in os.listdir(input_file_path)
                          if os.path.isdir(os.path.join(input_file_path, f))]
            kpi_file_folder = max(subfolders, key=os.path.getmtime) if subfolders else None
            if not kpi_file_folder:
                st.error("No run folders found in input path.")
                return

            # KPI file path
            kpi_file_path = os.path.join(kpi_file_folder, "kpis.csv")
            if not os.path.exists(kpi_file_path):
                st.error(f"kpis.csv not found in {kpi_file_folder}")
                return

            st.write(f"Using latest KPI file: {kpi_file_path}")

            model_path = config.get("model_path")
            feature_ranking_csv = config.get("feature_ranking_csv")
            if not os.path.exists(feature_ranking_csv):
                st.error("Feature ranking CSV not found. Run ETL first.")
                return

            # Detect anomalies
            df_with_anomalies = detect_anomalies_with_trained_model(kpi_file_path, model_path)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            anomaly_file_path = os.path.join(os.getcwd(), "../result", f"df_with_anomalies_{timestamp}.csv")
            os.makedirs(os.path.dirname(anomaly_file_path), exist_ok=True)
            df_with_anomalies.to_csv(anomaly_file_path, index=False)
            st.success(f"Anomaly detection completed. Results saved to: {anomaly_file_path}")

            # --- push to prometheus --
            pushgateway_url = "http://localhost:9091/"
            delay_sec = 1
            batch_size = 20
            job_name = "kpi_"+anomaly_file_path
            push_metrics_batch(df_with_anomalies, pushgateway_url, job_name)

            # --- Trigger RCA automatically ---
            st.header("Root Cause Analysis (RCA) after Anomaly Detection")
            run_folder = kpi_file_folder  # Use detected folder
            kpis, alarms, logs = load_data(run_folder)

            context = ""
            if alarms is not None and not alarms.empty:
                context += alarms.tail(10).to_string(index=False) + "\n"
            if logs is not None and not logs.empty:
                context += logs.tail(10).to_string(index=False) + "\n"
            if not context.strip():
                context = "No recent data."

            st.info("Performing RCA using Ollama...")
            rca_result = perform_rca("Analyze detected anomalies", context)
            if rca_result:
                st.success(rca_result)
            else:
                st.warning("No RCA result returned. Check if Ollama model is running.")

            # --- Trigger Visualization automatically ---
            st.header("Visualize KPIs with Anomalies")
            numeric_cols = df_with_anomalies.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_cols:
                st.error("No numeric columns found for visualization.")
            else:
                df_with_anomalies['timestamp'] = pd.to_datetime(df_with_anomalies['timestamp'])
                df_with_anomalies = df_with_anomalies.sort_values('timestamp')
                df_with_anomalies[numeric_cols] = normalize_df(df_with_anomalies[numeric_cols])

                vis_type = st.selectbox("Select Visualization Type", ["Line", "Area", "Scatter"])
                selected_metrics = st.multiselect(
                    "Select metrics to visualize", options=numeric_cols, default=numeric_cols
                )

                for metric in selected_metrics:
                    fig = plot_metric_with_anomalies(df_with_anomalies, metric, vis_type)
                    st.pyplot(fig)

        except Exception as e:
            st.error(f"Anomaly detection or post-processing failed: {e}")

if __name__ == "__main__":
    main()

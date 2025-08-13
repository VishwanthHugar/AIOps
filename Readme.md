# AIOps for Telecommunication Network KPI Anomaly Detection & RCA

## Overview
This project implements an **AIOps pipeline** for detecting anomalies in **telecommunication network KPI time series data**, classifying anomalies, and performing **Root Cause Analysis (RCA)** using **LLM integration (Ollama)**.

It supports:
- Collection of KPI, alarm, and log data from **multiple sources** (simulated, S3, public datasets).  Currently supported only simulated data.
- **Anomaly detection** using isolation forest based methods.
- **Visualization** of anomalies, events, and trends (via **Streamlit** & **Grafana**).
- RCA powered by **Ollama**.
- MLOps integration for continuous deployment & monitoring.

---

## Project Architecture
```plaintext
        Data Sources
   ┌───────────────────┐
   │ Simulated Data    │
   │ S3 Buckets        │
   │ Public Datasets   │
   └───────────────────┘
            │
            ▼
   ┌───────────────────┐
   │ Data Collector/   │
   │ feature extract   │
   │ (Python classes)  │
   └───────────────────┘
            │
            ▼
   ┌───────────────────┐
   │ Anomaly Detection │
   │ (ML, stats)       │
   └───────────────────┘
            │
            ▼
   ┌───────────────────┐
   │ RCA via Ollama    │
   └───────────────────┘
            │
            ▼
   ┌────────────────────┐
   │ Visualization      │
   │ (Streamlit/Grafana)│
   └────────────────────┘

---

## File Structure

.
├── src/app.py                       # Streamlit main application
├── src/app2.py                      # Alternative Streamlit app
├── src/anomaly_detection.py         # Anomaly detection logic
├── src/rca_genai.py                 # RCA integration with Ollama
├── src/Simulate.py                  # Simulated data collector
├── src/feature_selections.py        # ETL and feature selection
├── src/utils.py                     # Utility functions
├── src/config.txt                   # Configuration file
├── src/feature_ranking.csv          # Feature ranking CSV
├── src/push_to_prometheus_click.py  # Push events to Prometheus
├── src/train_inference.py           # Model training & inference
├── src/visualization.py             # Visualization logic
├── src/Generate_data.sh             # Data generation script
├── requirements.txt                 # Python dependencies
├── README.md                        # Project documentation
├── Dockerfile                       # Docker configuration
├── result/                          # Stores the result csv files
├── test_data/                       # Stores the testing runs
├── model/iforest_single.pkl         # model trained on the output_data
└── output_data/                     # Generated dataset


```
---

## Setting up project
```bash
git clone https://github.com/yourusername/aiops-telecom-kpi.git
cd aiops-telecom-kpi
pip install -r requirements.txt
install ollama, llama3.2:1b, prometheus, prometheus client, Grafana 
```
---

## Prometheus configure
```bash
Add the job in the prometheus.yml
  - job_name: 'pushgateway'
    static_configs:
      - targets: ['localhost:9091']

./prometheus --config.file=prometheus.yml
```
---
## Start the pushgateway
```bash
./pushgateway --web.listen-address=":9091"
```
---
## Start the Grafana for visualization realtime
```bash
/usr/sbin/grafana-server --homepath=/usr/share/grafana
```

---
## Start the ollama for RCA
```bash
ollama serve
```
---

## To train the model
```bash
python main.py infer \
    --kpi-csv output_data/run_001/kpis.csv \
    --model-dir model \
    --output-csv results_run001.csv
```
---

## To inference
```bash
python  train_inference.py train \
	--data-dir output_data \
	--feature-ranking-csv feature_ranking.csv 
	--model-dir model
```
---

## To run end to end
```bash
streamlit run app.py
```

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

.
├── main.py                      # Streamlit main application
├── anomaly_detection.py         # Anomaly detection logic
├── rca_genai.py                 # Streamlit application for RCA integration with Ollama
├── Simulate.py                  # Simulated/S3 data collector
├── feature_selections.py        # ETL and feature selection
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
├── push_to_prometheus.py        # push event to prometheus
├── visualization.py             # visualisation
├── Dockerfile                   # Docker file used for the project
└── output_data/                 # Example dataset
 

---

## Steps to run

git clone https://github.com/yourusername/aiops-telecom-kpi.git
cd aiops-telecom-kpi
pip install -r requirements.txt
install ollama, llama3.2:1b, prometheus, prometheus client, Grafana 

Add the job in the prometheus.yml
  - job_name: 'pushgateway'
    static_configs:
      - targets: ['localhost:9091']

./prometheus --config.file=prometheus.yml
./pushgateway --web.listen-address=":9091"
/usr/sbin/grafana-server --homepath=/usr/share/grafana
ollama serve


## To run end to end
streamlit run main.py


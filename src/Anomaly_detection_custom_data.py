import streamlit as st
import pandas as pd
import altair as alt
import threading, time
from collections import deque
#from simulator_3 import generate_kpi_data_with_trend_and_anomalies, static_analysis
from Simulate import generate_kpi_data_with_trend_and_anomalies, static_analysis
from learn_apply_hist import KPIAnalyzerModel  # anomaly-only class

# ---------------- CONFIG ----------------
TICK_SECONDS = 1
BUFFER_MAX = 5000
TREND_WINDOW = 30  # moving average window for trend

# ---------------- GENERATE KPI DATA ----------------
df_historic_data = generate_kpi_data_with_trend_and_anomalies()
df_historic_data['timestamp'] = pd.to_datetime(df_historic_data['timestamp'])
kpis = [col for col in df_historic_data.columns if col != 'timestamp']

# ---------------- INITIALIZE ANALYZER ----------------
analyzer = KPIAnalyzerModel()
analyzer.train_historical(df_historic_data)

# ---------------- BUFFER & SIMULATOR ----------------
buffer = deque(maxlen=BUFFER_MAX)

df_full = generate_kpi_data_with_trend_and_anomalies(start_date="2025-07-02")
df_full['timestamp'] = pd.to_datetime(df_full['timestamp'])
kpis = [col for col in df_full.columns if col != 'timestamp']
static_analysis(df_full, time_col='timestamp', output_dir="./plots")

def simulator():
    for _, row in df_full.iterrows():
        buffer.append(row.to_dict())
        time.sleep(TICK_SECONDS)

threading.Thread(target=simulator, daemon=True).start()

# ---------------- STREAMLIT UI ----------------
st.title("Real-Time KPI with Interval Anomalies, Trend Bands & PoP Analysis")

# ---------------- SIDEBAR SETTINGS ----------------
st.sidebar.header("Settings")
kpi_sel = st.sidebar.selectbox("Select KPI", kpis)
pop_minutes = st.sidebar.number_input(
    "PoP Comparison Window (minutes)",
    min_value=1, max_value=120, value=60, step=1,
    help="Number of minutes to look back for PoP calculation"
)
window_points = st.sidebar.number_input(
    "Visible Window Size (points)",
    min_value=5, max_value=500, value=30, step=5,
    help="Number of recent points to show in the chart"
)

# ---------------- PLACEHOLDERS ----------------
placeholder_main = st.empty()
placeholder_pop = st.empty()

# ---------------- STREAM PROCESSING ----------------
while True:
    if not buffer:
        time.sleep(1)
        continue

    # Convert buffer to DataFrame
    data = pd.DataFrame(list(buffer))
    if data.empty:
        print("Data is empty\n")
        continue

    df_processed = analyzer.apply_to_new_data(data)

    # ---------------- Period-over-Period Calculation ----------------
    prev_col = f"{kpi_sel}_prev"
    df_processed[prev_col] = df_processed[kpi_sel].shift(pop_minutes)
    df_processed['PoP_%'] = ((df_processed[kpi_sel] - df_processed[prev_col]) / df_processed[prev_col])
    df_processed['PoP_%'] = df_processed['PoP_%'].fillna(0) * 100  # fill NaN with 0 and convert to %

    # ---------------- Trend Calculation ----------------
    df_processed['trend'] = df_processed[kpi_sel].rolling(window=TREND_WINDOW, min_periods=1).mean()

    # ---------------- Interval Stats ----------------
    # Small epsilon to avoid equal lower/upper
    epsilon = 1e-6

    # 1 Compute interval_mean and interval_std based on analyzer stats
    df_processed['interval_mean'] = df_processed['interval'].map(
        lambda x: analyzer.interval_stats.get(x, {}).get('mean', df_processed[kpi_sel].mean())
    )

    df_processed['interval_std'] = df_processed['interval'].map(
        lambda x: analyzer.interval_stats.get(x, {}).get('std', df_processed[kpi_sel].std())
    )

    # 2️ Fill missing interval_std with default value
    df_processed.loc[df_processed['interval_std'].isna(), 'interval_std'] = 2.0

    # 3️ Compute lower and upper bounds
    df_processed['lower'] = df_processed['interval_mean'] - df_processed['interval_std']
    df_processed['upper'] = df_processed['interval_mean'] + df_processed['interval_std']

    # 4️ Handle NaNs in lower
    df_processed.loc[df_processed['lower'].isna(), 'lower'] = df_processed['interval_mean'] - df_processed['interval_std']

    # 5️ Handle NaNs in upper
    df_processed['upper'] = df_processed['upper'].fillna(df_processed[kpi_sel].mean())

    # 6️ Ensure upper != lower
    df_processed['upper'] = df_processed['upper'].mask(df_processed['upper'] == df_processed['lower'], df_processed['upper'] + epsilon)

    # ---------------- Select rolling window slice ----------------
    if len(df_processed) == 0:
        print ("Data processed is empty \n")
        # Show all rows and columns
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 200)
        pd.set_option('display.colheader_justify', 'center')
        pd.set_option('display.float_format', '{:.2f}'.format)

        print(df_processed)
        continue  # skip if not enough points

    visible_data = df_processed.iloc[-window_points:].copy()

    # ---------------- KPI Chart ----------------
    base = alt.Chart(visible_data).mark_line(color='blue').encode(
        x='timestamp:T', y=kpi_sel, tooltip=[kpi_sel, 'timestamp', 'interval', 'PoP_%']
    )
    trend_line = alt.Chart(visible_data).mark_line(color='green').encode(
        x='timestamp:T', y='trend'
    )
    interval_band = alt.Chart(visible_data).mark_area(opacity=0.3, color='yellow').encode(
        x='timestamp:T', y='lower', y2='upper'
    )
    anomaly_col = f"{kpi_sel}_anomaly"
    z_points = visible_data[visible_data[anomaly_col] == 1]
    anomaly_points = alt.Chart(z_points).mark_point(color='red', size=50).encode(
        x='timestamp:T', y=kpi_sel, tooltip=['timestamp', kpi_sel, 'interval', 'PoP_%']
    ) if not z_points.empty else alt.Chart(pd.DataFrame()).mark_point()

    chart_main = (interval_band + base + trend_line + anomaly_points).interactive()
    placeholder_main.altair_chart(chart_main, use_container_width=True)

    # ---------------- PoP Chart ----------------
    pop_chart = alt.Chart(visible_data).mark_line(color='purple').encode(
        x='timestamp:T', y='PoP_%', tooltip=['timestamp', 'PoP_%']
    ).properties(title=f"PoP (%) Change for {kpi_sel}").interactive()
    placeholder_pop.altair_chart(pop_chart, use_container_width=True)

    time.sleep(TICK_SECONDS)

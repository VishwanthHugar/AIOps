import pandas as pd

class KPIAnalyzerModel:
    def __init__(self, kpi_col=None, timestamp_col="timestamp"):
        """
        kpi_col: str or list of KPI columns to analyze. If None, will take all non-timestamp columns.
        timestamp_col: name of the timestamp column
        """
        self.kpi_col = kpi_col
        self.timestamp_col = timestamp_col
        self.zscore_threshold = 3
        self.interval_stats = {}  # mean/std per interval per KPI

    # ---------------- TRAINING ----------------
    def train_historical(self, df):
        # Auto-detect KPI columns if not set
        if self.kpi_col is None:
            self.kpi_col = [c for c in df.columns if c != self.timestamp_col]
        elif isinstance(self.kpi_col, str):
            self.kpi_col = [self.kpi_col]

        df = self.assign_interval(df)

        # Compute stats for each KPI column
        for kpi in self.kpi_col:
            self.compute_stats(df, kpi)
        return df

    def assign_interval(self, df):
        def interval(hour):
            if 6 <= hour < 12: return "morning"
            elif 12 <= hour < 18: return "afternoon"
            elif 18 <= hour < 24: return "evening"
            else: return "night"
        df['interval'] = df[self.timestamp_col].dt.hour.apply(interval)
        return df

    def compute_stats(self, df, kpi):
        stats = df.groupby("interval")[kpi].agg(['mean','std']).reset_index()
        self.interval_stats[kpi] = {row['interval']: {'mean': row['mean'], 'std': row['std']} 
                                    for _, row in stats.iterrows()}

    # ---------------- INFERENCE ----------------
    def apply_to_new_data(self, df):
        df = self.assign_interval(df)
        df = self.detect_anomalies(df)
        return df

    def detect_anomalies(self, df):
        for kpi in self.kpi_col:
            df[f'{kpi}_z_score'] = 0.0
            df[f'{kpi}_anomaly'] = 0

            for interval, stats in self.interval_stats[kpi].items():
                idx = df['interval'] == interval
                # Z-score anomaly detection
                df.loc[idx, f'{kpi}_z_score'] = (df.loc[idx, kpi] - stats['mean']) / stats['std']
                df.loc[idx, f'{kpi}_anomaly'] = (df.loc[idx, f'{kpi}_z_score'].abs() > self.zscore_threshold).astype(int)
        return df

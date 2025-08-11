import pandas as pd
import glob
import os
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import IsolationForest, RandomForestClassifier

class FeatureImportanceETL:
    def __init__(self, input_folder="output_data", output_file="feature_ranking.csv"):
        self.input_folder = os.path.join(os.getcwd(), input_folder)
        self.output_file = os.path.join(os.getcwd(), output_file)

    def extract(self):
        """Read all CSV files recursively from the input folder and merge them."""
        all_files = glob.glob(os.path.join(self.input_folder, "**", "*.csv"), recursive=True)
        if not all_files:
            raise FileNotFoundError(f"No CSV files found in {self.input_folder}")
        
        dfs = []
        for file in all_files:
            try:
                dfs.append(pd.read_csv(file))
            except Exception as e:
                print(f"[Extract] Error reading {file}: {e}")
        
        if not dfs:
            raise ValueError("No CSV files could be loaded successfully.")
        
        df = pd.concat(dfs, ignore_index=True)
        print(f"[Extract] Loaded {len(all_files)} files, total rows: {len(df)}")
        return df

    def transform(self, df):
        """Select important features using variance and model-based ranking."""
        features = [
            'latency_ms', 'throughput_mbps', 'packet_loss_percent', 'jitter_ms', 'rtt_ms',
            'call_setup_success_rate', 'data_session_success_rate', 'handover_success_rate', 'call_drop_rate',
            'cpu_utilization_percent', 'memory_utilization_percent', 'bandwidth_utilization_percent', 'disk_utilization_percent'
        ]
        
        # Keep only available columns
        available_features = [f for f in features if f in df.columns]
        if not available_features:
            raise ValueError("None of the expected features are in the dataset.")
        
        # Handle missing data
        #df = df[available_features].dropna()
        df = df[available_features].fillna(df[available_features].mean())

        # Step 1: Variance threshold
        selector = VarianceThreshold(threshold=0.01)
        selector.fit(df)
        high_variance_features = [available_features[i] for i in selector.get_support(indices=True)]
        print(f"[Transform] High variance features: {high_variance_features}")

        if not high_variance_features:
            raise ValueError("No features passed the variance threshold.")

        # Step 2: Isolation Forest for anomaly labels
        iso = IsolationForest(random_state=42)
        iso.fit(df[high_variance_features])
        labels = (iso.predict(df[high_variance_features]) == -1).astype(int)

        # Step 3: RandomForest for feature ranking
        rf = RandomForestClassifier(random_state=42)
        rf.fit(df[high_variance_features], labels)
        importance_scores = rf.feature_importances_

        ranked_features = sorted(
            zip(high_variance_features, importance_scores),
            key=lambda x: x[1],
            reverse=True
        )
        return ranked_features

    def load(self, ranked_features):
        """Save ranked features to CSV."""
        df_ranked = pd.DataFrame(ranked_features, columns=["feature", "importance"])
        df_ranked.to_csv(self.output_file, index=False)
        print(f"[Load] Saved ranked features to {self.output_file}")

    def run(self):
        df = self.extract()
        ranked_features = self.transform(df)
        self.load(ranked_features)
        print("[ETL] Done.")

# if __name__ == "__main__":
#     etl = FeatureImportanceETL(input_folder="output_data", output_file="feature_ranking.csv")
#     etl.run()

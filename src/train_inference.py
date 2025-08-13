import os
import click
from anomaly_detection import train_time_series_anomaly_model, detect_anomalies_with_trained_model


@click.group()
def cli():
    """CLI for training and inference of time-series anomaly detection."""
    pass


@cli.command()
@click.option("--data-dir", required=True, type=click.Path(exists=True, file_okay=False),
              help="Directory containing run_* folders with KPI CSVs")
@click.option("--feature-ranking-csv", required=True, type=click.Path(exists=True, dir_okay=False),
              help="Path to feature ranking CSV")
@click.option("--model-dir", default="model", type=click.Path(file_okay=False),
              help="Directory to save the trained model")
@click.option("--contamination", default=0.05, show_default=True, type=float,
              help="Anomaly contamination ratio")
@click.option("--top-n", default=7, show_default=True, type=int,
              help="Top N features to use")
def train(data_dir, feature_ranking_csv, model_dir, contamination, top_n):
    """Train the single IsolationForest anomaly detection model."""
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "iforest_single.pkl")

    train_time_series_anomaly_model(
        base_dir=data_dir,
        feature_ranking_csv=feature_ranking_csv,
        model_path=model_path,
        contamination=contamination,
        top_n=top_n
    )


@cli.command()
@click.option("--kpi-csv", required=True, type=click.Path(exists=True, dir_okay=False),
              help="Path to KPI CSV for inference")
@click.option("--model-dir", default="model", type=click.Path(exists=True, file_okay=False),
              help="Directory containing the trained model")
@click.option("--output-csv", required=True, type=click.Path(dir_okay=False),
              help="Path to save the anomaly detection output CSV")
def infer(kpi_csv, model_dir, output_csv):
    """Run inference on new KPI data using trained model."""
    model_path = os.path.join(model_dir, "iforest_single.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    detect_anomalies_with_trained_model(
        kpi_csv_path=kpi_csv,
        model_path=model_path,
        output_csv=output_csv
    )


if __name__ == "__main__":
    cli()

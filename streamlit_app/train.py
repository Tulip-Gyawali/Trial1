from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .models.training import TrainingOutputs, train_model
from .utils.datasets import load_eew_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the PGA prediction pipeline.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/EEW_features_2024-10-21.csv"),
        help="Path to the EEW features CSV file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("streamlit_app/models"),
        help="Directory to store the trained model and preprocessing artifacts.",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("streamlit_app/models/metrics_summary.parquet"),
        help="Optional path to write evaluation metrics as a parquet file.",
    )
    args = parser.parse_args()

    df = load_eew_dataset(args.data)
    outputs: TrainingOutputs = train_model(df, save_dir=args.output)

    if args.metrics:
        metrics_df = []
        for label, values in outputs.metrics_log.items():
            row = {"split": label, "scale": "log", **values}
            metrics_df.append(row)
        for label, values in outputs.metrics_raw.items():
            row = {"split": label, "scale": "raw", **values}
            metrics_df.append(row)
        metrics_table = pd.DataFrame(metrics_df)
        args.metrics.parent.mkdir(parents=True, exist_ok=True)
        metrics_table.to_parquet(args.metrics, index=False)

    print("Training complete. Artifacts saved to", args.output)


if __name__ == "__main__":
    main()

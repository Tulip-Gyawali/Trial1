from __future__ import annotations

from typing import Dict

import pandas as pd


def build_metrics_table(metrics: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    columns = sorted(next(iter(metrics.values())).keys()) if metrics else []
    data = []
    for split in ["train", "val", "test"]:
        if split in metrics:
            row = [metrics[split][col] for col in columns]
            data.append([split.capitalize(), *row])
    return pd.DataFrame(data, columns=["Split", *columns])

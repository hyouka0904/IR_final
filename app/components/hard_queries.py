"""Failure analysis：讀 results/{query_set}/per_query/{config}.csv，依指標由低到高列出 query。"""
import os

import pandas as pd


def load_hard_queries(results_root: str, query_set: str, config_id: str, metric: str = "map") -> pd.DataFrame | None:
    path = os.path.join(results_root, query_set, "per_query", f"{config_id}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, dtype={"query": str})
    df.insert(0, "qidx", range(len(df)))
    return df.dropna(subset=[metric]).sort_values(metric, kind="stable").reset_index(drop=True)


def available_configs(results_root: str, query_set: str) -> list[str]:
    d = os.path.join(results_root, query_set, "per_query")
    return sorted(f[:-4] for f in os.listdir(d) if f.endswith(".csv")) if os.path.isdir(d) else []

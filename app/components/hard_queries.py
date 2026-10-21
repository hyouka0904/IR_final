"""讀取 results/{dataset}/per_query/*.csv，依 AP 由低到高列出困難 query。"""
import os

import pandas as pd


def load_hard_queries(results_root: str, dataset: str, config_id: str, protocol: str = "M") -> pd.DataFrame | None:
    """回傳欄位 [qidx, query, AP] 的表格；找不到檔案（05_run_grid.py 尚未執行）時回傳 None。"""
    path = os.path.join(results_root, dataset, "per_query", f"{config_id}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    col = f"{protocol}_AP"
    # 沒有 positive 的 query AP 為 NaN，不列入
    return (df.dropna(subset=[col]).sort_values(col)[["qidx", "query", col]]
            .rename(columns={col: "AP"}).reset_index(drop=True))

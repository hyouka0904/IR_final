"""冒煙測試：以假資料依序執行 02 → (假 03) → 04 → 05 → 05 --speed → 06 → 05 phone → check_data。

每支腳本以子行程執行，與實際使用方式相同；只確認流程能跑完、輸出檔案齊全且數值合理，不檢查檢索品質。
約需數十秒；只想跑單元測試時用 pytest -m "not smoke"。
"""
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from src.config import ROOT, load_config
from tests.smoke import fake_data

pytestmark = pytest.mark.smoke


def run(script: str, *args: str, config: str) -> str:
    env = {**os.environ, "PYTHONUTF8": "1"}
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / script), *args, "--config", config],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, cwd=ROOT)
    assert r.returncode == 0, f"{script} {' '.join(args)} 失敗：\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}"
    return r.stdout


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    work = tmp_path_factory.mktemp("smoke")
    config = fake_data.make_data(str(work))
    cfg = load_config(config)
    run("02_prepare_data.py", "--no-download", config=config)
    fake_data.make_features(cfg)
    run("04_build_indexes.py", config=config)
    run("05_run_eval.py", config=config)
    run("05_run_eval.py", "--speed", config=config)
    return config, cfg


def test_prepare_outputs(pipeline):
    _, cfg = pipeline
    sets = os.listdir(os.path.join(cfg["paths"]["cache_root"], "sets"))
    assert {"coco40k", "coco_semantic", "coco_semantic_box", "coco_semantic_box_full", "coco_similar",
            "holidays", "holidays_db", "phone"} <= set(sets)


def test_eval_summaries(pipeline):
    _, cfg = pipeline
    results = cfg["paths"]["results_root"]
    for qset in ("coco_semantic", "coco_semantic_box", "coco_semantic_box_full", "coco_similar", "holidays"):
        df = pd.read_csv(os.path.join(results, qset, "summary.csv"))
        assert set(df["config_id"]) == set(cfg["eval"]["experiments"][qset]), qset
        metrics = df[["ndcg_cut_10", "map", "recip_rank"]].to_numpy()
        assert np.isfinite(metrics).all() and (metrics >= 0).all() and (metrics <= 1).all(), qset
        per_query = set(os.listdir(os.path.join(results, qset, "per_query")))   # --speed 的結果也寫在這裡
        assert {f"{c}.csv" for c in df["config_id"]} <= per_query, qset
    by_edit = pd.read_csv(os.path.join(results, "coco_similar", "summary_by_edit.csv"))
    assert set(by_edit["edit"]) == set(cfg["data"]["edits"]["types"])
    # 假特徵中 edit query = 來源圖向量 + 少量雜訊，向量模型的 flat 搜尋應該幾乎都找得到來源圖
    sim = pd.read_csv(os.path.join(results, "coco_similar", "summary.csv")).set_index("config_id")
    assert sim.loc["dinov2-flat-none", "recall_10"] > 0.9


def test_speed_summary(pipeline):
    _, cfg = pipeline
    df = pd.read_csv(os.path.join(cfg["paths"]["results_root"], "coco_semantic", "summary_speed.csv"))
    assert set(df["config_id"]) == set(cfg["eval"]["experiments"]["speed"])
    assert (df.filter(like="lat_").fillna(0).to_numpy() >= 0).all()


def test_human_judgments(pipeline):
    """06：pool → 模擬兩位評審填分數 → kappa → qrels → 05 --queries phone。"""
    config, cfg = pipeline
    run("06_human_judgments.py", "pool", config=config)
    path = os.path.join(cfg["paths"]["results_root"], "phone", "judgments.csv")
    df = pd.read_csv(path, dtype=str)
    rng = np.random.default_rng(0)
    a, b = cfg["judging"]["judges"]
    df[a] = rng.integers(0, 3, len(df))
    df[b] = np.where(rng.random(len(df)) < 0.8, df[a], rng.integers(0, 3, len(df)))
    df.to_csv(path, index=False, encoding="utf-8-sig")                    # 模擬 Excel 存成「CSV UTF-8」（有 BOM）
    run("06_human_judgments.py", "kappa", config=config)
    run("06_human_judgments.py", "qrels", config=config)
    run("05_run_eval.py", "--queries", "phone", config=config)
    assert os.path.exists(os.path.join(cfg["paths"]["results_root"], "phone", "summary.csv"))


def test_check_data(pipeline):
    """02 產生的 cache/sets 要通過完整性檢查（假 COCO 的張數與真實 COCO 不同，所以不檢查 coco 標註）。"""
    config, _ = pipeline
    out = run("check_data.py", "--only", "partials", "sets", config=config)
    assert "失敗 0" in out

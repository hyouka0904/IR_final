"""圖庫（gallery）與 query set 的存檔格式，以及 qrels。

每個 image set 存在 cache/sets/{name}/：
  manifest.csv   idx, id, path（相對 repo 根目錄），以及各 set 自己的欄位
  meta.json      {"kind": "gallery" | "queries", "gallery": ..., "task": ...}
  qrels.npz      CategoryQrels（semantic：由類別矩陣即時算分級）
  qrels.txt      TREC 格式 qrels（similar / holidays / phone）

manifest 的選用欄位：
  box          JSON [x1, y1, x2, y2]，query 先用此框裁切（走與 demo 框選相同的程式）
  exclude_idx  評估時從結果中移除的圖庫索引（Holidays 的 query 本身也在圖庫裡）
  source_idx   edit 前的原圖在圖庫中的索引
  edit, edit_params, category ...
"""
import csv
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.config import ROOT


def set_dir(cache_root: str, name: str) -> str:
    return os.path.join(cache_root, "sets", name)


@dataclass
class ImageSet:
    name: str
    rows: list[dict]                      # manifest 的每一列（值皆為字串，選用欄位已解析）
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def ids(self) -> list[str]:
        return [r["id"] for r in self.rows]

    @property
    def paths(self) -> list[str]:
        return [str(ROOT / r["path"]) for r in self.rows]

    def box(self, i: int):
        return self.rows[i].get("box")

    def exclude(self, i: int) -> int | None:
        v = self.rows[i].get("exclude_idx")
        return None if v in (None, "") else int(v)

    def column(self, key: str) -> list:
        return [r.get(key) for r in self.rows]


def rel_path(path: str | Path) -> str:
    p = Path(path).resolve()
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def save_set(cache_root: str, name: str, rows: list[dict], meta: dict) -> str:
    d = set_dir(cache_root, name)
    os.makedirs(d, exist_ok=True)
    keys = ["idx", "id", "path"] + sorted({k for r in rows for k in r} - {"idx", "id", "path"})
    with open(os.path.join(d, "manifest.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        for i, r in enumerate(rows):
            out = {k: r.get(k, "") for k in keys}
            out["idx"] = i
            for k in ("box", "edit_params"):
                if isinstance(out.get(k), (list, tuple, dict)):
                    out[k] = json.dumps(out[k])
            w.writerow(out)
    with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"name": name, **meta}, f, ensure_ascii=False, indent=2)
    return d


def load_set(cache_root: str, name: str) -> ImageSet:
    d = set_dir(cache_root, name)
    with open(os.path.join(d, "manifest.csv"), newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("box", "edit_params"):
            if r.get(k):
                r[k] = json.loads(r[k])
            elif k in r:
                r[k] = None
    with open(os.path.join(d, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    return ImageSet(name, rows, meta)


def list_sets(cache_root: str, kind: str | None = None) -> list[str]:
    root = os.path.join(cache_root, "sets")
    if not os.path.isdir(root):
        return []
    out = []
    for name in sorted(os.listdir(root)):
        meta = os.path.join(root, name, "meta.json")
        if os.path.exists(meta):
            with open(meta, encoding="utf-8") as f:
                if kind is None or json.load(f).get("kind") == kind:
                    out.append(name)
    return out


# ---------------- qrels ----------------

class Qrels:
    """grades(qi) 回傳長度為圖庫大小的整數陣列（0 = 不相關）。"""

    n_queries: int
    n_docs: int

    def grades(self, qi: int) -> np.ndarray:
        raise NotImplementedError


class CategoryQrels(Qrels):
    """semantic：分級 = query 與圖庫圖片共享的類別數，上限 max_grade。

    1,000 個 query × 40k 張圖的 qrels 若展開成 TREC 檔會有上千萬列（person 類別幾乎到處都是），
    因此只存類別矩陣，需要時即時計算。
    """

    def __init__(self, q_cats: np.ndarray, g_cats: np.ndarray, max_grade: int):
        self.q_cats = q_cats.astype(np.int16)    # (Q, C) 0/1
        self.g_cats = g_cats.astype(np.int16)    # (N, C) 0/1
        self.max_grade = int(max_grade)
        self.n_queries, self.n_docs = len(q_cats), len(g_cats)

    def grades(self, qi: int) -> np.ndarray:
        return np.minimum(self.g_cats @ self.q_cats[qi], self.max_grade).astype(np.int8)

    def save(self, path: str) -> None:
        np.savez_compressed(path, q_cats=self.q_cats.astype(np.uint8), g_cats=self.g_cats.astype(np.uint8),
                            max_grade=self.max_grade)

    @classmethod
    def load(cls, path: str, max_grade: int | None = None) -> "CategoryQrels":
        z = np.load(path)
        return cls(z["q_cats"], z["g_cats"], int(z["max_grade"]) if max_grade is None else max_grade)


class SparseQrels(Qrels):
    """一般的 qrels：每個 query 一個 {圖庫索引: 分級}。與 trec_eval 相同，未判斷的文件視為不相關。"""

    def __init__(self, rels: list[dict[int, int]], n_docs: int):
        self.rels = rels
        self.n_queries, self.n_docs = len(rels), n_docs

    def grades(self, qi: int) -> np.ndarray:
        g = np.zeros(self.n_docs, dtype=np.int8)
        for d, r in self.rels[qi].items():
            g[d] = r
        return g

    def save(self, path: str, q_ids: list[str], doc_ids: list[str]) -> None:
        """TREC 格式：qid 0 docid rel，只寫 rel > 0 的列。"""
        with open(path, "w", encoding="utf-8") as f:
            for qi, rel in enumerate(self.rels):
                for d in sorted(rel):
                    f.write(f"{q_ids[qi]} 0 {doc_ids[d]} {rel[d]}\n")

    @classmethod
    def load(cls, path: str, q_ids: list[str], doc_ids: list[str]) -> "SparseQrels":
        q_pos = {q: i for i, q in enumerate(q_ids)}
        d_pos = {d: i for i, d in enumerate(doc_ids)}
        rels = [dict() for _ in q_ids]
        with open(path, encoding="utf-8") as f:
            for line in f:
                qid, _, did, r = line.split()
                if qid in q_pos and did in d_pos and int(r) > 0:
                    rels[q_pos[qid]][d_pos[did]] = int(r)
        return cls(rels, len(doc_ids))


def load_qrels(cache_root: str, queries: ImageSet, gallery: ImageSet, cfg: dict) -> Qrels | None:
    src = queries.meta.get("qrels_from", queries.name)   # box_full 與 box 共用 qrels
    d = set_dir(cache_root, src)
    if os.path.exists(os.path.join(d, "qrels.npz")):
        max_grade = queries.meta.get("max_grade", cfg["eval"]["semantic_max_grade"])
        return CategoryQrels.load(os.path.join(d, "qrels.npz"), max_grade)
    if os.path.exists(os.path.join(d, "qrels.txt")):
        return SparseQrels.load(os.path.join(d, "qrels.txt"), queries.ids, gallery.ids)
    return None

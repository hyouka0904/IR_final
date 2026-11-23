"""抽樣圖庫與 query、下載 COCO 圖片、產生 edit query，並寫出 manifest 與 qrels。

產出（cache/sets/{name}/）：
  coco40k                 圖庫：train2017 中有標註的圖，隨機 40,000 張
  coco_semantic           1,000 張 val2017；qrels = 共享類別數（上限 eval.semantic_max_grade）
  coco_semantic_box       從 coco_semantic 中含 >= 2 類別的 query，框出其中一個物件；qrels = 是否含該類別
  coco_semantic_box_full  同上但不裁切（對照組，共用 qrels）
  coco_similar            1,000 張圖庫圖片各套一種 edit；qrels = 原圖
  holidays_db / holidays  INRIA Holidays 圖庫與 500 個 query（data/holidays 存在時）
  phone                   data/phone 裡的手機照片（存在時；qrels 由 06_human_judgments.py 產生）
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image
from tqdm import tqdm

from src.config import ROOT, load_config
from src.data import coco as coco_mod
from src.data import edits, holidays
from src.data.sets import CategoryQrels, SparseQrels, rel_path, save_set, set_dir

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")  # HEIC 請先轉成 JPEG


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--no-download", action="store_true", help="不下載 COCO 圖片（已自行放好 train2017 / val2017）")
    return p.parse_args()


def prepare_coco(cfg: dict, download: bool) -> None:
    dcfg, data_root, cache_root = cfg["data"], cfg["paths"]["data_root"], cfg["paths"]["cache_root"]
    rng = np.random.default_rng(cfg["seed"])
    train = coco_mod.Coco(data_root, dcfg["coco"]["gallery_split"])
    val = coco_mod.Coco(data_root, dcfg["coco"]["query_split"])

    g_ids = coco_mod.sample(train.annotated_ids(), dcfg["coco"]["gallery_size"], rng)
    q_ids = coco_mod.sample(val.annotated_ids(), dcfg["coco"]["n_semantic_queries"], rng)
    if download:
        for c, ids in ((train, g_ids), (val, q_ids)):
            failed = coco_mod.download_images(c, ids)
            if failed:
                sys.exit(f"!! {c.split}: {failed} 張下載失敗，請重跑（已下載的會跳過）")
    missing = [i for i in g_ids if not os.path.exists(train.path(i))] + [i for i in q_ids if not os.path.exists(val.path(i))]
    if missing:
        sys.exit(f"!! 缺少 {len(missing)} 張 COCO 圖片，例如 {missing[:3]}")

    # 圖庫
    gallery = [{"id": str(i), "path": rel_path(train.path(i))} for i in g_ids]
    save_set(cache_root, "coco40k", gallery, {"kind": "gallery", "categories": train.cat_names})
    g_cats = train.category_matrix(g_ids)

    # Semantic
    q_cats = val.category_matrix(q_ids)
    rows = [{"id": str(i), "path": rel_path(val.path(i)), "n_categories": int(q_cats[k].sum())} for k, i in enumerate(q_ids)]
    d = save_set(cache_root, "coco_semantic", rows, {"kind": "queries", "gallery": "coco40k", "task": "semantic"})
    CategoryQrels(q_cats, g_cats, cfg["eval"]["semantic_max_grade"]).save(os.path.join(d, "qrels.npz"))
    print(f">> coco_semantic: {len(rows)} queries, 平均 {q_cats.sum(1).mean():.2f} 個類別")

    # Box queries（意圖消歧）
    bcfg = dcfg["box_queries"]
    box_rows, box_cats = [], []
    for k, iid in enumerate(q_ids):
        if q_cats[k].sum() < bcfg["min_categories"]:
            continue
        picked = val.pick_box(iid, rng, bcfg["min_area_frac"], bcfg["pad_frac"])
        if picked is None:
            continue
        col, box = picked
        box_rows.append({"id": f"{iid}_c{col}", "path": rel_path(val.path(iid)), "box": box,
                         "category": val.cat_names[col]})
        one_hot = np.zeros(len(val.cat_names), dtype=np.uint8)
        one_hot[col] = 1
        box_cats.append(one_hot)
    meta = {"kind": "queries", "gallery": "coco40k", "task": "semantic", "max_grade": 1}
    d = save_set(cache_root, "coco_semantic_box", box_rows, meta)
    CategoryQrels(np.stack(box_cats), g_cats, 1).save(os.path.join(d, "qrels.npz"))
    full_rows = [{k: v for k, v in r.items() if k != "box"} for r in box_rows]
    save_set(cache_root, "coco_semantic_box_full", full_rows, {**meta, "qrels_from": "coco_semantic_box"})
    print(f">> coco_semantic_box(_full): {len(box_rows)} queries")

    # Similar：edit query
    n_sim = dcfg["coco"]["n_similar_queries"]
    src_idx = sorted(int(i) for i in rng.choice(len(g_ids), size=n_sim, replace=False))
    types = edits.assign_edits(n_sim, dcfg["edits"]["types"], rng)
    out_dir = os.path.join(set_dir(cache_root, "coco_similar"), "images")
    os.makedirs(out_dir, exist_ok=True)
    rows, rels = [], []
    for gi, edit in tqdm(list(zip(src_idx, types)), desc="edits"):
        params = edits.sample_params(edit, dcfg["edits"], rng)
        qid = f"{edit}_{g_ids[gi]}"
        with Image.open(train.path(g_ids[gi])) as im:
            path = edits.save_edited(im.convert("RGB"), edit, params, os.path.join(out_dir, qid))
        rows.append({"id": qid, "path": rel_path(path), "source_idx": gi, "edit": edit, "edit_params": params})
        rels.append({gi: 1})
    d = save_set(cache_root, "coco_similar", rows, {"kind": "queries", "gallery": "coco40k", "task": "similar"})
    SparseQrels(rels, len(g_ids)).save(os.path.join(d, "qrels.txt"), [r["id"] for r in rows], [r["id"] for r in gallery])
    print(f">> coco_similar: {len(rows)} queries")


def prepare_holidays(cfg: dict) -> None:
    data_root, cache_root = cfg["paths"]["data_root"], cfg["paths"]["cache_root"]
    names = holidays.image_names(data_root)
    if not names:
        print("-- 找不到 data/holidays/jpg，略過 Holidays")
        return
    gallery = [{"id": n[:-4], "path": rel_path(holidays.image_path(data_root, n)), "group": holidays.group_of(n)} for n in names]
    save_set(cache_root, "holidays_db", gallery, {"kind": "gallery"})
    rows, rels = [], []
    for gi, n in enumerate(names):
        if not holidays.is_query(n):
            continue
        rows.append({"id": n[:-4], "path": gallery[gi]["path"], "exclude_idx": gi})
        rels.append({j: 1 for j, m in enumerate(names) if j != gi and holidays.group_of(m) == holidays.group_of(n)})
    d = save_set(cache_root, "holidays", rows, {"kind": "queries", "gallery": "holidays_db", "task": "similar"})
    SparseQrels(rels, len(names)).save(os.path.join(d, "qrels.txt"), [r["id"] for r in rows], [g["id"] for g in gallery])
    print(f">> holidays: {len(names)} 張圖庫、{len(rows)} 個 query")


def prepare_phone(cfg: dict) -> None:
    d = cfg["data"]["phone"]["dir"]
    d = d if os.path.isabs(d) else str(ROOT / d)
    names = sorted(f for f in os.listdir(d) if f.lower().endswith(IMAGE_EXTS)) if os.path.isdir(d) else []
    if not names:
        print(f"-- {d} 沒有照片，略過 phone")
        return
    rows = [{"id": os.path.splitext(n)[0], "path": rel_path(os.path.join(d, n))} for n in names]
    save_set(cfg["paths"]["cache_root"], "phone", rows, {"kind": "queries", "gallery": "coco40k", "task": "semantic"})
    print(f">> phone: {len(rows)} 張（qrels 待人工標註後由 06_human_judgments.py 產生）")


def main():
    args = parse_args()
    cfg = load_config(args.config)
    ann = os.path.join(cfg["paths"]["data_root"], "coco", "annotations", f"instances_{cfg['data']['coco']['gallery_split']}.json")
    if not os.path.exists(ann):
        sys.exit(f"!! 找不到 {ann}，請先執行 python scripts/01_download_data.py")
    prepare_coco(cfg, download=not args.no_download)
    prepare_holidays(cfg)
    prepare_phone(cfg)


if __name__ == "__main__":
    main()

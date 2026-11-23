"""資料完整性檢查：原始資料集（01 下載的）與 cache/sets（02 產生的）。

每個 check_* 回傳 Section，裡面是 (等級, 訊息)：
  ok    通過
  skip  資料集還沒下載 / 還沒準備（選用的資料集，例如 Holidays、ROxford）
  fail  檔案缺少、數量不符、圖片損毀、qrels 對不上
scripts/check_data.py 是命令列入口；tests/test_data_integrity.py 用合成資料測這些檢查本身，
並在真實資料存在時直接檢查真實資料。
"""
import json
import os
import pickle
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from src.data import holidays as holidays_mod
from src.data.coco import load_instances
from src.data import revisited
from src.data.sets import CategoryQrels, list_sets, load_set, set_dir

COCO_N_IMAGES = {"train2017": 118287, "val2017": 5000}
COCO_N_CATEGORIES = 80
HOLIDAYS_N, HOLIDAYS_N_QUERIES = 1491, 500
REVISITED_N = {"roxford5k": (4993, 70), "rparis6k": (6322, 70)}       # (圖庫, query)
JPEG_EOI = b"\xff\xd9"


@dataclass
class Section:
    name: str
    items: list[tuple[str, str]] = field(default_factory=list)

    def ok(self, msg: str) -> None:
        self.items.append(("ok", msg))

    def skip(self, msg: str) -> None:
        self.items.append(("skip", msg))

    def fail(self, msg: str) -> None:
        self.items.append(("fail", msg))

    def expect(self, cond: bool, msg: str) -> bool:
        self.items.append(("ok" if cond else "fail", msg))
        return cond

    @property
    def failed(self) -> bool:
        return any(level == "fail" for level, _ in self.items)


# ---------------- 圖片 ----------------

def image_problem(path: str, full: bool = False) -> str | None:
    """回傳問題描述；沒問題回傳 None。

    預設的快速檢查：檔案存在且非空、PIL 能讀出格式與尺寸；JPEG 另外檢查結尾的 EOI 標記
    （下載或解壓被截斷的 JPEG 會缺這個標記），缺少時再完整解碼確認。full=True 則每張都完整解碼。
    """
    if not os.path.exists(path):
        return "不存在"
    if os.path.getsize(path) == 0:
        return "大小為 0"
    try:
        with Image.open(path) as im:
            fmt = im.format
            if im.width == 0 or im.height == 0:
                return "尺寸為 0"
        need_decode = full
        if fmt == "JPEG" and not full:
            with open(path, "rb") as f:
                f.seek(max(0, os.path.getsize(path) - 64))
                tail = f.read().rstrip(b"\x00\r\n ")
            need_decode = not tail.endswith(JPEG_EOI)      # 少數正常 JPEG 結尾有其他資料，解碼確認再判定
        if need_decode:
            with Image.open(path) as im:
                im.load()
    except Exception as e:                                  # PIL 對各種損毀會丟出不同的例外
        return f"無法讀取（{type(e).__name__}: {e}）"
    return None


def check_images(sec: Section, label: str, paths: list[str], full: bool = False, workers: int = 8,
                 progress=None) -> None:
    """檢查一批圖片，列出前幾個有問題的檔案。progress 可傳入 tqdm 之類的包裝函式。"""
    with ThreadPoolExecutor(workers) as ex:
        it = ex.map(lambda p: image_problem(p, full), paths)
        problems = list(progress(it, total=len(paths), desc=label) if progress else it)
    bad = [(p, msg) for p, msg in zip(paths, problems) if msg]
    if not bad:
        sec.ok(f"{label}：{len(paths)} 張圖片都可讀取")
        return
    examples = "；".join(f"{os.path.basename(p)} {msg}" for p, msg in bad[:5])
    sec.fail(f"{label}：{len(bad)} / {len(paths)} 張有問題，例如 {examples}")


# ---------------- 原始資料集 ----------------

def check_partials(data_root: str) -> Section:
    sec = Section("未完成的下載")
    parts = sorted(Path(data_root).rglob("*.part")) if os.path.isdir(data_root) else []
    if parts:
        names = "、".join(str(p.relative_to(data_root)) for p in parts[:5])
        sec.fail(f"有 {len(parts)} 個未完成的下載（{names}）：重跑 01（會接續下載）或 02（COCO 圖片）")
    else:
        sec.ok("沒有未完成的下載（.part）")
    return sec


def check_coco_annotations(data_root: str) -> Section:
    sec = Section("COCO 標註")
    for split, n_expected in COCO_N_IMAGES.items():
        path = os.path.join(data_root, "coco", "annotations", f"instances_{split}.json")
        slim = path[:-len(".json")] + ".slim.json"
        if not os.path.exists(path) and not os.path.exists(slim):
            sec.fail(f"{split}：找不到 {path}，請執行 01_download_data.py")
            continue
        try:
            # load_instances 有精簡版就讀精簡版，沒有就以不保留 segmentation 的方式讀原始檔並產生精簡版
            if os.path.exists(path):
                inst = load_instances(path)
            else:
                with open(slim, encoding="utf-8") as f:
                    inst = json.load(f)
        except (ValueError, OSError) as e:
            sec.fail(f"{split}：標註檔無法解析（{type(e).__name__}: {e}），請刪除後重跑 01")
            continue
        sec.expect(len(inst["images"]) == n_expected, f"{split}：{len(inst['images'])} 張圖（應為 {n_expected}）")
        sec.expect(len(inst["categories"]) == COCO_N_CATEGORIES,
                   f"{split}：{len(inst['categories'])} 個類別（應為 {COCO_N_CATEGORIES}）")
        sec.expect(len(inst["annotations"]) > 0, f"{split}：{len(inst['annotations'])} 個物件標註")
    return sec


def check_holidays(data_root: str, full: bool = False, progress=None) -> Section:
    sec = Section("INRIA Holidays")
    names = holidays_mod.image_names(data_root)
    if not names:
        sec.skip("尚未下載（選用；similar search 的補充評估）")
        return sec
    n_q = sum(holidays_mod.is_query(n) for n in names)
    sec.expect(len(names) == HOLIDAYS_N, f"{len(names)} 張圖（應為 {HOLIDAYS_N}）")
    sec.expect(n_q == HOLIDAYS_N_QUERIES, f"{n_q} 個 query（應為 {HOLIDAYS_N_QUERIES}）")
    groups = {holidays_mod.group_of(n) for n in names}
    lonely = [g for g in groups if sum(holidays_mod.group_of(n) == g for n in names) < 2]
    sec.expect(not lonely, f"每組至少 2 張（只有 1 張的組：{len(lonely)}）")
    check_images(sec, "Holidays 圖片", [holidays_mod.image_path(data_root, n) for n in names], full, progress=progress)
    return sec


def check_revisited(data_root: str, dataset: str, full: bool = False, progress=None) -> Section:
    sec = Section(dataset)
    gnd_path = os.path.join(revisited.dataset_dir(dataset, data_root), f"gnd_{dataset}.pkl")
    if not os.path.exists(gnd_path):
        sec.skip("尚未下載（選用；01_download_data.py --with-roxford）")
        return sec
    try:
        with open(gnd_path, "rb") as f:
            gnd = pickle.load(f)
    except Exception as e:
        sec.fail(f"gnd 檔無法讀取（{type(e).__name__}: {e}），請刪除後重跑 01")
        return sec
    n_db, n_q = REVISITED_N[dataset]
    sec.expect(len(gnd["imlist"]) == n_db, f"圖庫 {len(gnd['imlist'])} 張（應為 {n_db}）")
    sec.expect(len(gnd["qimlist"]) == n_q and len(gnd["gnd"]) == n_q, f"{len(gnd['qimlist'])} 個 query（應為 {n_q}）")
    # 只檢查 gnd 用到的圖：RParis 原始封存檔中有 20 張損毀的圖，本來就不在 imlist 裡
    paths = revisited.db_image_paths(gnd, dataset, data_root) + revisited.query_image_paths(gnd, dataset, data_root)
    check_images(sec, f"{dataset} 圖片", paths, full, progress=progress)
    return sec


# ---------------- cache/sets ----------------

EXPECTED_SIZE = {  # set 名稱 → config 中的大小
    "coco40k": ("coco", "gallery_size"),
    "coco_semantic": ("coco", "n_semantic_queries"),
    "coco_similar": ("coco", "n_similar_queries"),
}


def check_sets(cfg: dict, full: bool = False, progress=None) -> list[Section]:
    """02 產生的每個 set：大小、圖片、qrels 與圖庫是否一致。"""
    cache = cfg["paths"]["cache_root"]
    names = list_sets(cache)
    if not names:
        sec = Section("cache/sets")
        sec.skip("尚未產生（請執行 02_prepare_data.py）")
        return [sec]
    sets = {n: load_set(cache, n) for n in names}
    out = []
    for name, s in sets.items():
        sec = Section(f"set {name}")
        out.append(sec)
        if name in EXPECTED_SIZE:
            group, key = EXPECTED_SIZE[name]
            n_expected = cfg["data"][group][key]
            sec.expect(len(s) == n_expected, f"{len(s)} 筆（config 為 {n_expected}）")
        else:
            sec.expect(len(s) > 0, f"{len(s)} 筆")
        sec.expect(len(set(s.ids)) == len(s), "id 沒有重複")
        if s.meta.get("kind") == "queries":
            gallery = sets.get(s.meta.get("gallery"))
            if not sec.expect(gallery is not None, f"對應的圖庫 {s.meta.get('gallery')} 存在"):
                continue
            check_query_refs(sec, s, gallery)
            check_qrels(sec, cache, s, gallery)
        # 同一張圖可能被多個 set 共用（例如 coco_semantic 與 box），各自檢查一次也只是重複讀檔頭
        check_images(sec, f"{name} 圖片", sorted(set(s.paths)), full, progress=progress)
    return out


def check_query_refs(sec: Section, s, gallery) -> None:
    for col in ("exclude_idx", "source_idx"):
        vals = [v for v in s.column(col) if v not in (None, "")]
        if vals:
            bad = [v for v in vals if not 0 <= int(v) < len(gallery)]
            sec.expect(not bad, f"{col} 都在圖庫範圍內（超出：{len(bad)}）")
    boxes = [b for b in s.column("box") if b]
    if boxes:
        bad = [b for b in boxes if not (len(b) == 4 and b[0] < b[2] and b[1] < b[3])]
        sec.expect(not bad, f"box 格式正確（x1 < x2、y1 < y2；錯誤：{len(bad)}）")


def check_qrels(sec: Section, cache: str, s, gallery) -> None:
    src = s.meta.get("qrels_from", s.name)
    d = set_dir(cache, src)
    npz, txt = os.path.join(d, "qrels.npz"), os.path.join(d, "qrels.txt")
    if os.path.exists(npz):
        q = CategoryQrels.load(npz)
        sec.expect(q.n_queries == len(s) and q.n_docs == len(gallery),
                   f"qrels {q.n_queries}×{q.n_docs}（應為 {len(s)}×{len(gallery)}）")
        no_rel = sum(1 for i in range(min(q.n_queries, len(s))) if not (q.g_cats @ q.q_cats[i]).any())
        sec.expect(no_rel == 0, f"每個 query 至少有一張相關圖（沒有的：{no_rel}）")
    elif os.path.exists(txt):
        q_ids, g_ids = set(s.ids), set(gallery.ids)
        unknown_q, unknown_d, judged = set(), set(), set()
        with open(txt, encoding="utf-8") as f:
            for line in f:
                qid, _, did, rel = line.split()
                if qid not in q_ids:
                    unknown_q.add(qid)
                if did not in g_ids:
                    unknown_d.add(did)
                if int(rel) > 0:
                    judged.add(qid)
        sec.expect(not unknown_q, f"qrels 的 query id 都在 set 中（未知：{len(unknown_q)}）")
        sec.expect(not unknown_d, f"qrels 的 doc id 都在圖庫中（未知：{len(unknown_d)}）")
        missing = len(q_ids - judged)
        sec.expect(missing == 0, f"每個 query 至少有一張相關圖（沒有的：{missing}）")
    elif s.name == "phone":
        sec.skip("phone 的 qrels 尚未產生（人工標註後由 06_human_judgments.py qrels 產生）")
    else:
        sec.fail("找不到 qrels（qrels.npz / qrels.txt），請重跑 02")


def summarize(sections: list[Section]) -> tuple[int, int, int]:
    counts = {"ok": 0, "skip": 0, "fail": 0}
    for sec in sections:
        for level, _ in sec.items:
            counts[level] += 1
    return counts["ok"], counts["skip"], counts["fail"]


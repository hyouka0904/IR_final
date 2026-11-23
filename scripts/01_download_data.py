"""下載 COCO 2017 標註、INRIA Holidays，以及（選用）supplementary 用的 ROxford / RParis。

只用 Python 標準函式庫，Windows / macOS / Linux 都能跑（不需要 bash、curl、wget、unzip、tar）。
COCO 圖片不在這裡下載：02_prepare_data.py 抽樣後只下載用得到的約 4.1 萬張。
已下載完成的部分會跳過，中斷後重跑即可。

用法：python scripts/01_download_data.py [--with-roxford]
"""
import argparse
import os
import shutil
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config

COCO_ANN = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
HOLIDAYS = [f"ftp://ftp.inrialpes.fr/pub/lear/douze/data/{p}.tar.gz" for p in ("jpg1", "jpg2")]
HOLIDAYS_N = 1491
REVISITOP_ZIP = "https://github.com/filipradenovic/revisitop/archive/refs/heads/master.zip"
REVISITED = {  # 與 revisitop download.py 相同的來源
    "roxford5k": ("https://www.robots.ox.ac.uk/~vgg/data/oxbuildings", ["oxbuild_images-v1.tgz"]),
    "rparis6k": ("https://www.robots.ox.ac.uk/~vgg/data/parisbuildings", ["paris_1-v1.tgz", "paris_2-v1.tgz"]),
}
GND_URL = "http://cmp.felk.cvut.cz/revisitop/data/datasets/{0}/gnd_{0}.pkl"


def fetch(url: str, dst: Path) -> Path:
    """下載到 dst；已存在且非空就跳過。先寫到 .part，完整下載後才改名。"""
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_name(dst.name + ".part")
    print(f">> 下載 {url}")
    with urllib.request.urlopen(url, timeout=60) as r, open(part, "wb") as f:
        total, done, last = int(r.headers.get("Content-Length") or 0), 0, -1
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            pct = done * 100 // total if total else -1
            if pct != last and pct % 5 == 0:
                print(f"   {done / 2**20:,.0f} MB" + (f" / {total / 2**20:,.0f} MB ({pct}%)" if total else ""), flush=True)
                last = pct
    os.replace(part, dst)
    return dst


def count_jpg(d: Path) -> int:
    return sum(1 for _ in d.rglob("*.jpg")) if d.is_dir() else 0


def extract_flat(archive: Path, dst: Path) -> None:
    """解壓 tar 並把所有 .jpg 攤平到 dst（revisitop 的封存檔內有子資料夾）。"""
    dst.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        for m in tar:
            if m.isfile() and m.name.lower().endswith(".jpg"):
                with tar.extractfile(m) as src, open(dst / Path(m.name).name, "wb") as out:
                    shutil.copyfileobj(src, out)


def coco(data: Path) -> None:
    ann = data / "coco" / "annotations"
    names = [f"instances_{s}2017.json" for s in ("train", "val")]
    if not all((ann / n).exists() for n in names):
        z = fetch(COCO_ANN, data / "coco" / "annotations_trainval2017.zip")
        with zipfile.ZipFile(z) as zf:
            for n in names:
                zf.extract(f"annotations/{n}", data / "coco")
    print(">> COCO annotations OK")


def holidays(data: Path) -> None:
    d = data / "holidays"
    if count_jpg(d / "jpg") < HOLIDAYS_N:
        for url in HOLIDAYS:
            try:
                archive = fetch(url, d / url.rsplit("/", 1)[1])
            except OSError as e:
                sys.exit(f"!! Holidays 下載失敗（{e}）。請到 https://lear.inrialpes.fr/~jegou/data.php "
                         f"手動下載 jpg1.tar.gz、jpg2.tar.gz 放到 {d} 後重跑")
            extract_flat(archive, d / "jpg")
    print(f">> Holidays: {count_jpg(d / 'jpg')} 張")


def revisitop_code(revisitop_path: Path) -> None:
    """07 用 revisitop 的 compute_map；下載 repo 的 zip（不需要 git）。"""
    repo = revisitop_path.parent
    if (revisitop_path / "evaluate.py").exists():
        return
    z = fetch(REVISITOP_ZIP, repo.parent / "revisitop-master.zip")
    with zipfile.ZipFile(z) as zf:
        zf.extractall(repo.parent)
    if repo.exists():
        shutil.rmtree(repo)
    os.replace(repo.parent / "revisitop-master", repo)
    z.unlink()
    print(f">> revisitop 程式碼 → {repo}")


def revisited(data: Path) -> None:
    for ds, (base, files) in REVISITED.items():
        d = data / "datasets" / ds
        if count_jpg(d / "jpg") == 0:
            for name in files:
                archive = fetch(f"{base}/{name}", d / name)
                extract_flat(archive, d / "jpg")
                archive.unlink()
        fetch(GND_URL.format(ds), d / f"gnd_{ds}.pkl")
        n = count_jpg(d / "jpg")
        if n == 0:
            sys.exit(f"!! {ds} 下載不完整（jpg: 0 張）。若原始連結失效，請到 revisitop repo 的 issues 查詢替代來源。")
        print(f">> {ds}: {n} 張 jpg")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--with-roxford", action="store_true", help="另外下載 supplementary 用的 ROxford / RParis")
    args = p.parse_args()
    cfg = load_config(args.config)
    data = ROOT / cfg["paths"]["data_root"]
    coco(data)
    holidays(data)
    if args.with_roxford:
        revisitop_code(ROOT / cfg["paths"]["revisitop"])
        revisited(data)


if __name__ == "__main__":
    main()

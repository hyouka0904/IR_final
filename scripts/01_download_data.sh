#!/usr/bin/env bash
# 下載 COCO 2017 標註、INRIA Holidays，以及（選用）supplementary 用的 ROxford / RParis。
# COCO 圖片不在這裡下載：02_prepare_data.py 抽樣後只下載用得到的約 4.1 萬張。
#
# 用法：bash scripts/01_download_data.sh [--with-roxford]
set -euo pipefail
cd "$(dirname "$0")/.."

fetch() {  # fetch URL 目的檔；已存在就跳過
  if [ ! -s "$2" ]; then
    echo ">> 下載 $1"
    curl -fL --retry 3 -o "$2.part" "$1" && mv "$2.part" "$2"
  fi
}

# ---------- COCO 2017 annotations（約 241 MB） ----------
mkdir -p data/coco
if [ ! -s data/coco/annotations/instances_train2017.json ]; then
  fetch http://images.cocodataset.org/annotations/annotations_trainval2017.zip data/coco/annotations_trainval2017.zip
  (cd data/coco && unzip -q -o annotations_trainval2017.zip 'annotations/instances_*2017.json')
fi
echo ">> COCO annotations OK"

# ---------- INRIA Holidays（約 2.5 GB） ----------
mkdir -p data/holidays
if [ ! -d data/holidays/jpg ] || [ "$(find data/holidays/jpg -name '*.jpg' | wc -l)" -lt 1491 ]; then
  for part in jpg1 jpg2; do
    if ! fetch "ftp://ftp.inrialpes.fr/pub/lear/douze/data/$part.tar.gz" "data/holidays/$part.tar.gz"; then
      echo "!! Holidays 下載失敗。請到 https://lear.inrialpes.fr/~jegou/data.php 手動下載 jpg1/jpg2.tar.gz 放到 data/holidays/ 後重跑" >&2
      exit 1
    fi
    tar -xzf "data/holidays/$part.tar.gz" -C data/holidays
  done
fi
echo ">> Holidays: $(find data/holidays/jpg -name '*.jpg' | wc -l) 張"

# ---------- ROxford / RParis（supplementary，選用） ----------
if [ "${1:-}" = "--with-roxford" ]; then
  if [ ! -d third_party/revisitop ]; then
    mkdir -p third_party
    git clone https://github.com/filipradenovic/revisitop.git third_party/revisitop
  fi
  # download_datasets 遇到已存在的 jpg/ 與 gnd 檔會跳過；失敗時不會報錯，所以下面逐一檢查
  (cd third_party/revisitop/python && python -c "from download import download_datasets; download_datasets('../../../data')")
  for ds in roxford5k rparis6k; do
    n=$(find "data/datasets/$ds/jpg" -name '*.jpg' 2>/dev/null | wc -l)
    if [ ! -s "data/datasets/$ds/gnd_$ds.pkl" ] || [ "$n" -eq 0 ]; then
      echo "!! $ds 下載不完整（jpg: $n 張）。若原始連結失效，請到 revisitop repo 的 issues 查詢替代來源。" >&2
      exit 1
    fi
    echo ">> $ds: $n 張 jpg"
  done
fi

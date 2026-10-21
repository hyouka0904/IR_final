#!/usr/bin/env bash
# 用官方 revisitop 工具下載 ROxford / RParis 圖片與 ground truth（README 第 2 節）
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -d third_party/revisitop ]; then
  mkdir -p third_party
  git clone https://github.com/filipradenovic/revisitop.git third_party/revisitop
fi

mkdir -p data
# download_datasets 遇到已存在的 jpg/ 與 gnd 檔會跳過
(cd third_party/revisitop/python && python -c "from download import download_datasets; download_datasets('../../../data')")

# 下載失敗時 revisitop 不會報錯（內部用 os.system），這裡逐一檢查
for ds in roxford5k rparis6k; do
  gnd="data/datasets/$ds/gnd_$ds.pkl"
  n=$(find "data/datasets/$ds/jpg" -name '*.jpg' 2>/dev/null | wc -l)
  if [ ! -s "$gnd" ] || [ "$n" -eq 0 ]; then
    echo "!! $ds 下載不完整（gnd: $gnd, jpg: $n 張）。若原始連結失效，請到 revisitop repo 的 issues 查詢替代來源。" >&2
    exit 1
  fi
  echo ">> $ds: $n 張 jpg"
done

# Landmark Image Retrieval: Failure Mode Analysis & Diagnostic Demo

這是一套以 ROxford / RParis 為圖庫的以圖搜圖系統。我們比較三種預訓練模型（CLIP、SigLIP、DINOv2）、三種 FAISS 索引，以及兩種 re-ranking 手段（alpha-QE、SIFT + RANSAC）的效果與成本，並提供一個能逐個 query 診斷結果的互動 demo。

---

## 1. 環境建置

### 1.1 系統需求

- Linux（以 Ubuntu 22.04 測試）
- Python 3.11
- NVIDIA GPU：只有抽 embedding 時需要；評估與 demo 都可以在 CPU 上執行
- 硬碟空間：資料集約 5 GB；embedding、索引、SIFT 特徵與 inlier 快取另外需要數 GB

### 1.2 建立環境

```bash
conda create -n lmret python=3.11 -y
conda activate lmret

# PyTorch（CUDA 版本請依本機調整）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# FAISS：有 GPU 就用 conda 版，沒有 GPU 就用 CPU 版
conda install -c pytorch -c nvidia faiss-gpu -y
# pip install faiss-cpu

pip install -r requirements.txt
```

`requirements.txt` 的內容：

```
open_clip_torch      # CLIP, SigLIP
timm
opencv-python>=4.5   # SIFT（4.4 版起已併入主套件）
gradio
numpy
scipy
pandas
h5py                 # SIFT 特徵與 inlier 快取的存檔格式
pyyaml
tqdm
matplotlib
pillow
```

DINOv2 透過 `torch.hub` 載入，第一次執行時會自動下載權重。

### 1.3 驗證安裝

```bash
python -c "import torch, faiss, cv2, open_clip; print(torch.cuda.is_available(), faiss.__version__, cv2.SIFT_create())"
```

### 1.4 注意事項

- **延遲一律分階段量測**，分成 embedding 抽取、索引搜尋、re-ranking 三段。其中「索引搜尋」這一段一律在 CPU 上量，原因是 FAISS 的 HNSW 只支援 CPU，三種索引都在 CPU 上量才公平。
- 模型版本、K、alpha、ensemble 權重、索引參數與尺度，統一寫在 `configs/default.yaml`，不要寫死在程式裡。
- IVF-PQ 的分群與量化 codebook 直接以圖庫本身訓練。這一步不屬於模型訓練。

---

## 2. 資料集下載

用官方 revisitop 工具下載圖片與 ground truth：

```bash
mkdir -p third_party
git clone https://github.com/filipradenovic/revisitop.git third_party/revisitop

cd third_party/revisitop/python
python -c "from download import download_datasets; download_datasets('../../../data')"
cd ../../..
```

下載完成後的結構如下：

```
data/datasets/
├── roxford5k/
│   ├── jpg/                 # 圖片（包含 query 圖）
│   └── gnd_roxford5k.pkl    # 內含 imlist, qimlist, gnd（gnd 含 bbx, easy, hard, junk）
└── rparis6k/
    ├── jpg/
    └── gnd_rparis6k.pkl
```

### 使用規則

| 項目 | 規則 |
|---|---|
| 資料庫範圍 | **只使用 `gnd['imlist']` 列出的圖片**：ROxford 4,993 張、RParis 6,322 張。`jpg/` 資料夾裡另外還有 70 張 query 圖，這些圖不屬於資料庫 |
| Query | 每個資料集 70 個，名單在 `gnd['qimlist']` |
| Query 裁切 | 官方協定要求 query 必須用 `gnd['gnd'][i]['bbx']` 裁切，**所有正式實驗（包含 baseline）都要裁切**。唯一例外是「不裁切」的補充實驗，結果會單獨標示 |
| 評估 | 使用 revisitop 的 `compute_map`，Easy / Medium / Hard 各自計算 mAP 與 mP@k（k = 1, 5, 10） |
| 不使用 | R1M distractor |

如果原始圖片的下載連結失效，請到 revisitop repo 的 issues 查詢替代來源。

---

## 3. 檔案目錄規劃

```
landmark-retrieval/
├── README.md
├── requirements.txt
├── .gitignore                  # 排除 data/, cache/, third_party/
├── configs/
│   └── default.yaml            # 模型版本、K、alpha、QE 的 n、ensemble 權重、索引參數、尺度
│
├── data/                       # [不進 git] 原始資料集
│   └── datasets/{roxford5k, rparis6k}/
│
├── cache/                      # [不進 git] 離線階段產物，demo 啟動時直接載入
│   ├── embeddings/{dataset}/
│   │   ├── {model}_{ss|ms}_db.npy      # 圖庫 embedding，順序與 imlist 一致
│   │   └── {model}_{ss|ms}_q.npy       # 官方 query embedding（已用 bbx 裁切）
│   ├── indexes/{dataset}/
│   │   └── {model}_{ss|ms}_{flat|hnsw|ivfpq}.index
│   ├── sift/{dataset}/
│   │   └── db_sift.h5                  # 圖庫的 SIFT keypoints + descriptors
│   └── inliers/{dataset}/
│       └── inliers.h5                  # (query, 圖庫圖片) → inlier 數，所有設定共用
│
├── results/                    # 實驗輸出（CSV 進 git，大圖可不進）
│   ├── {dataset}/
│   │   ├── summary.csv         # 每個設定的 mAP / mP@k / 分階段延遲 / 索引大小 / recall 損失
│   │   └── per_query/
│   │       └── {config_id}.csv # 每個 query 的 AP，是困難 query 清單的資料來源
│   └── figures/
│
├── src/
│   ├── data/
│   │   └── revisited.py        # 讀取 gnd、imlist、bbx；解析圖片路徑
│   ├── models/
│   │   ├── encoders.py         # CLIP / SigLIP / DINOv2 的統一介面，輸出 L2-normalized 向量
│   │   └── ensemble.py         # 候選取聯集 → 精確重算 → 加權融合
│   ├── preprocess/             # 線上與離線共用同一份
│   │   ├── crop.py             # bbx 裁切、使用者框選裁切
│   │   └── multiscale.py       # 多尺度抽取與合併
│   ├── index/
│   │   └── faiss_index.py      # Flat / HNSW / IVF-PQ 的建立、存檔與載入
│   ├── rerank/
│   │   ├── aqe.py              # alpha-QE（每個模型各自執行）
│   │   └── geometric.py        # SIFT 比對 + RANSAC；先查 inlier 快取，沒有才計算
│   ├── eval/
│   │   └── metrics.py          # 包裝 revisitop compute_map，並計算 recall 損失與分階段延遲
│   ├── schema.py               # 組員 A / B 共同定義的資料格式（SearchRequest, SearchResult）
│   └── pipeline.py             # 線上檢索流程，評估腳本與 demo 共用同一份
│
├── scripts/                    # 離線階段，依編號順序執行
│   ├── 01_download_data.sh
│   ├── 02_extract_embeddings.py
│   ├── 03_build_indexes.py
│   ├── 04_extract_sift.py
│   └── 05_run_grid.py          # 跑完整網格，產生 summary.csv 與 per_query/*.csv
│
├── app/                        # Gradio demo
│   ├── app.py
│   └── components/
│       ├── query_input.py      # 上傳 / 選官方 query / 框選
│       ├── compare_view.py     # 並排比較
│       ├── inlier_view.py      # 畫出 SIFT inlier 匹配連線
│       └── hard_queries.py     # 讀取 per_query/*.csv，依 AP 由低到高排序
│
└── third_party/
    └── revisitop/              # [不進 git] 官方評估與下載工具
```

### 3.1 設定命名（config_id）

```
{model}-{scale}-{index}-{rerank}
model : clip | siglip | dinov2 | ens
scale : ss | ms
index : flat | hnsw | ivfpq
rerank: none | aqe | gv | aqe+gv
範例  : dinov2-ms-flat-aqe+gv
```

- 每個資料集共 4 × 2 × 3 × 4 = **96 組**設定。
- `ens` 表示三個模型同時使用相同的 scale 與 index 設定；換索引時，三個模型的索引一起換。

### 3.2 線上流程（`src/pipeline.py`）

1. **前處理**：依 query 來源做 bbx 或框選裁切，再依設定決定是否使用 multi-scale。
2. **抽取 embedding**：每個模型分別抽取 query embedding。
3. **alpha-QE（可選）**：每個模型各自用初次檢索的前 n 名擴展 query 向量。
4. **檢索**：每個模型在自己的索引上取 top-K。
5. **Ensemble（`ens` 時）**：先取三個模型候選的聯集，再用 `cache/embeddings` 精確重算每個模型的 cosine 分數，加權融合後取 top-K。
6. **幾何驗證（可選）**：query 現場抽 SIFT，圖庫則讀取 `db_sift.h5`，以 RANSAC 的 inlier 數重新排序 top-K。官方 query 的 inlier 數優先從 `inliers.h5` 讀取。
7. **輸出**：排序結果、分數、分階段延遲，以及 inlier 匹配點（有啟用幾何驗證時才有）。

K 的初始值為 100。

### 3.3 指標定義

- **mAP、mP@k**：使用官方評估程式計算。
- **recall 損失**：ANN 索引回傳的 top-K 與 Flat 回傳的 top-K 的重疊比例，再拿 1 減去它。
- **索引大小**：`.index` 檔案的大小。
- **延遲**：分 embedding 抽取、索引搜尋、re-ranking 三段記錄；索引搜尋一段一律在 CPU 上量。

---

## 4. 執行順序

```bash
bash scripts/01_download_data.sh
python scripts/02_extract_embeddings.py --dataset roxford5k --model all --scale ss ms
python scripts/03_build_indexes.py      --dataset roxford5k
python scripts/04_extract_sift.py       --dataset roxford5k
python scripts/05_run_grid.py           --dataset roxford5k
# RParis 請以 --dataset rparis6k 再跑一次

python app/app.py
```

- `05_run_grid.py` 必須在 demo 之前執行，因為「困難 query 清單」需要讀取 `results/{dataset}/per_query/`。
- 第一次跑含幾何驗證的設定時，會建立 inlier 快取，之後的設定可以直接重用。報告中的消融路徑直接從 `summary.csv` 擷取，不需要另外重跑。

---

## 5. 分工對應

| 組員 | 負責目錄 |
|---|---|
| A（檢索核心） | `src/models`, `src/index`, `src/rerank`, `src/eval`, `scripts/` |
| B（系統層） | `app/`, `src/preprocess`（A 在離線階段呼叫同一份模組） |
| 共同 | `src/schema.py`, `src/pipeline.py`, `configs/` |

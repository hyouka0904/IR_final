# Zero-Shot Image-to-Image Search Engine

使用者上傳任意一張照片，系統就從約 4 萬張日常照片（MS-COCO 2017）中排序出結果。系統支援兩種任務：

- **Semantic search**：找同一類東西，例如拿貓的照片找其他貓。
- **Similar-image search**：找同一張圖或同一個場景，包括裁切、編輯過或重拍的版本。

研究問題：**哪一種影像表示法適合哪一種任務？** 我們比較 CLIP、SigLIP、DINOv2、pHash，以及以 CLIP 概念標籤（tag）為文件的 BM25。全部只用預訓練模型（zero-shot），並以 trec_eval 的指標評估。題目與分工見 [`proposal.md`](proposal.md)。

---

## 1. 環境建置

### 1.1 系統需求

- Linux（以 Ubuntu 22.04 測試）或 Windows 10 / 11（見 [1.4](#14-windows)），Python 3.11
- NVIDIA GPU：只有抽特徵（`03`、`07`）時需要；評估與 demo 都可以只用 CPU
- 硬碟：COCO 抽樣圖片約 7 GB、Holidays 約 2.5 GB；特徵與索引另外約 2 GB

### 1.2 建立環境

```bash
conda create -n zsis python=3.11 -y
conda activate zsis

# PyTorch（CUDA 版本請依本機調整）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# FAISS：一律用 CPU 版（HNSW 只支援 CPU，所有索引都在 CPU 上量延遲才公平）
pip install faiss-cpu

pip install -r requirements.txt
pip install -r requirements-dev.txt   # 單元測試用（選用）
```

DINOv2 透過 `torch.hub` 載入，第一次執行時會自動下載權重；CLIP / SigLIP 權重由 open_clip 下載。

### 1.3 驗證安裝

```bash
python -c "import torch, faiss, open_clip, imagehash; print(torch.cuda.is_available(), faiss.__version__)"
python -m pytest -q tests        # 單元測試，不需要 GPU、資料集與模型權重
```

### 1.4 Windows

所有腳本都是 Python，不需要 bash、WSL、curl、wget、unzip 或 tar。以下指令在 **PowerShell**（或 Anaconda PowerShell Prompt）執行。

**1. 事前準備**

- 安裝 [Miniconda](https://docs.conda.io/en/latest/miniconda.html)。若要在一般 PowerShell 用 conda，先執行一次 `conda init powershell` 再重開視窗。
- 有 NVIDIA 顯卡時，先更新驅動，再用 `nvidia-smi` 查看右上角的 `CUDA Version`，決定下面 PyTorch 要裝的版本。
- **repo 的位置**：路徑只用英數字（FAISS 讀寫索引時不支援中文路徑），也不要放在 OneDrive 同步的資料夾。「桌面」常被 OneDrive 同步，而資料加上 cache 超過 10 GB。建議放在 `C:\work\IR_final`、`D:\IR_final` 這類位置。
- 不想把資料放在 repo 裡時，可以改 `configs/default.yaml` 的 `paths.data_root`、`paths.cache_root`。路徑請用斜線，例如 `D:/zsis/data`。

**2. 建立環境**

```powershell
conda create -n zsis python=3.11 -y
conda activate zsis

# Windows 的 Python 預設用系統編碼（繁中是 cp950）讀寫檔案；改成 UTF-8，之後每次 activate 自動生效
conda env config vars set PYTHONUTF8=1 HF_HUB_DISABLE_SYMLINKS_WARNING=1
conda deactivate; conda activate zsis

# PyTorch：cu121 / cu124 依 nvidia-smi 的 CUDA Version 選，不能高於它；沒有 NVIDIA 顯卡就拿掉 --index-url 那段，裝 CPU 版
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install faiss-cpu      # PyPI 有 Windows wheel；若安裝失敗改用 conda install -c pytorch faiss-cpu
pip install -r requirements.txt
pip install -r requirements-dev.txt   # 選用；pytrec_eval-terrier 若需要編譯而失敗，可以略過，不影響主流程
```

**3. 驗證**

```powershell
python -c "import torch, faiss, open_clip, imagehash; print(torch.__version__, torch.cuda.is_available(), faiss.__version__)"
python -m pytest -q tests
```

有 NVIDIA 顯卡但第一個值印出 `False`，代表裝到 CPU 版的 torch。先 `pip uninstall torch torchvision`，再用正確的 `--index-url` 重裝。

**4. 執行**

與 [第 4 節](#4-執行順序) 相同，`scripts/` 用斜線或反斜線都可以：

```powershell
python scripts\01_download_data.py
python scripts\02_prepare_data.py
# ……03 → 05，與第 4 節相同
python app\app.py
```

**常見問題**

| 狀況 | 處理方式 |
|---|---|
| 01 的 Holidays 下載失敗 | 官方來源是 FTP，有些網路（例如校園網路）會擋。到 https://lear.inrialpes.fr/~jegou/data.php 手動下載 `jpg1.tar.gz`、`jpg2.tar.gz`，放到 `data\holidays\` 後重跑 01，會跳過下載直接解壓 |
| 下載中斷 | 直接重跑，已完成的檔案會跳過（下載中的檔案以 `.part` 結尾，完成才改名） |
| demo 啟動後打不開 `http://0.0.0.0:7860` | 瀏覽器請開 `http://localhost:7860`。Windows 防火牆跳出詢問時，只想本機使用可以按取消 |
| 第一次跑 03 卡很久 | 正在下載 CLIP / SigLIP / DINOv2 權重（共數 GB），存在 `%USERPROFILE%\.cache` |
| GPU 記憶體不足 | `python scripts\03_extract_features.py --batch-size 16`（或更小） |
| 用 Excel 填 `judgments.csv` | 存檔時選「CSV UTF-8」或「CSV（逗號分隔）」都可以，06 都讀得到。不要改欄位名稱與順序 |

---

## 2. 資料

| 名稱（`cache/sets/`） | 角色 | 內容 | 相關性（qrels） |
|---|---|---|---|
| `coco40k` | 圖庫 | COCO train2017 中至少有一個標註物件的圖，隨機抽 40,000 張 | — |
| `coco_semantic` | semantic query | COCO val2017 中有標註的圖，隨機抽 1,000 張 | 等級 = 與 query 共享的 COCO 類別數，上限 2（0 / 1 / 2） |
| `coco_semantic_box` | 意圖消歧 query | 從 `coco_semantic` 中含 ≥ 2 個類別的 query，選一個面積 ≥ 5% 的物件框起來 | 圖中是否有該類別（0 / 1） |
| `coco_semantic_box_full` | 對照組 | 與上面相同的 query，但不裁切 | 同上（共用 qrels） |
| `coco_similar` | similar query | 1,000 張圖庫圖片，平均分配 5 種 edit（crop / rotation / color / JPEG / blur，各 200 張） | edit 前的原圖（1 張） |
| `holidays_db` / `holidays` | 圖庫 / similar query | INRIA Holidays 1,491 張、500 個 query | 同組的其他照片，排除 query 本身 |
| `phone` | semantic query | 組員自己拍的 100 張手機照片（`data/phone/`） | pooled human judgments（見 4.2） |
| ROxford / RParis | supplementary | revisitop 官方資料 | 官方 Easy / Medium / Hard |

- 抽樣都使用固定的 `seed`（`configs/default.yaml`），重跑結果相同。
- 只抽有標註的圖，因為沒有任何標註的圖無法定義 semantic 分級。
- edit 的參數範圍在 `data.edits`，每張 query 實際用的參數記在 manifest 的 `edit_params` 欄。JPEG 以外的 edit 都存成 PNG，避免壓縮失真混進其他 edit 類型。
- **Tag 詞彙表（待決定）**：proposal 寫的是 "a fixed external class list"。不能用 COCO 80 類，因為 semantic 的 qrels 就是依 COCO 類別分級，用同一份清單會造成洩漏。目前預設為 open_clip 內建的 ImageNet-1k 類別名稱，可在 `tags.vocab` 換成任意文字檔（一行一個類別）。

---

## 3. 系統設計

### 3.1 設定命名（config_id）

```
{method}-{index}-{post}
method : clip | siglip | dinov2 | phash | bm25 | clip+bm25 | siglip+bm25 | dinov2+bm25
index  : flat | hnsw | ivfpq | cp      （向量部分；bm25 單獨使用時為 inv；phash 只有 flat）
post   : none，或 aqe / rocchio / mmr 以 + 串接（依此順序執行，只能用在向量模型）
範例   : siglip+bm25-flat-aqe+mmr
```

這次不跑完整網格。每個 query set 要跑哪些設定，列在 `eval.experiments`。

### 3.2 線上流程（`src/pipeline.py`，評估與 demo 共用）

1. **前處理**：有框選時先裁切（box cropping）。demo 的框選與 `coco_semantic_box` 走同一段程式。
2. **抽 query 特徵**：視設定抽向量、pHash，或 CLIP top-15 tag。
3. **向量檢索**：在 FAISS / cluster pruning 索引上取前 K 名。
   - **alpha-QE**：用前 n 名擴展 query 向量後再搜一次。
   - **Rocchio**：`q' = α·q + β·mean(相關) − γ·mean(不相關)`，擴展後再搜一次。demo 中由使用者勾選相關與不相關的結果。
4. **Tag BM25**：query 的 tag 當作查詢，在圖庫 tag 的 inverted index 上算 BM25。
5. **RRF**：向量排序與 BM25 排序以 `Σ 1/(60 + rank)` 融合。
6. **MMR**：`λ·rel − (1−λ)·max sim(已選)` 多樣化重排，讓前幾名涵蓋不同意圖（the cat or the sofa?）。
7. **輸出**：排序、分數、分階段延遲（embed / search / rerank）、query tag，以及每個結果命中的 tag（explainability）。

### 3.3 各元件

| 元件 | 檔案 | 說明 |
|---|---|---|
| CLIP / SigLIP / DINOv2 | `src/models/encoders.py` | 使用各模型官方的前處理，輸出 L2-normalized 向量 |
| pHash | `src/models/phash.py` | imagehash 64 bits、FAISS `IndexBinaryFlat`；分數 = 1 − hamming / 64 |
| CLIP tagger | `src/models/tagger.py` | 詞彙表 × 4 個 prompt template 取平均；每張圖取 top-15 tag |
| Inverted index（HW1） | `src/text/inverted_index.py` | 依 doc 排序的 postings、布林 AND / OR，以及給 explainability 用的 forward index |
| BM25（HW2） | `src/text/bm25.py` | Okapi BM25（k1 = 1.2、b = 0.75，query 端 tf 以 k3 飽和），term-at-a-time 計分 |
| Cluster pruning（HW2） | `src/index/cluster_pruning.py` | 隨機選 √N 個 leader，b1 = 1、b2 = 4 |
| FAISS 索引 | `src/index/faiss_index.py` | Flat / HNSW / IVF-PQ，一律用 inner product |
| alpha-QE / Rocchio / MMR / RRF | `src/rerank/` | |
| 指標 | `src/eval/metrics.py` | 與 trec_eval 定義相同（見 3.4） |

BM25 文件的 tf：`tags.tf_mode: weighted` 時，每個 tag 依 CLIP 機率重複 `max(1, round(10 × p))` 次，讓高信心的 tag 權重較高，文件長度也有差異。若用 `binary`，15 個 tag 的文件長度全部相同，BM25 會退化成 IDF 加權的交集。

### 3.4 指標

| 指標 | trec_eval 名稱 | 用途 |
|---|---|---|
| nDCG@10 | `ndcg_cut_10` | 主要指標；gain 為 qrels 等級本身 |
| mAP | `map` | run 深度 1,000（`retrieval.depth`） |
| Recall@1、Recall@10 | `recall_1`、`recall_10` | similar 任務只有 1 張相關圖時最直觀 |
| MRR | `recip_rank` | |
| cat_coverage_10 | — | 前 10 名涵蓋了 query 幾成類別；用來看 MMR 的多樣性，nDCG 看不出這一點 |
| recall 損失 | — | 1 − ANN 前 K 名與 Flat 前 K 名的重疊比例 |
| 延遲、索引大小 | — | 延遲分 embed / search / rerank 三段；search 一律在 CPU、單執行緒量測 |

semantic 的 qrels 若展開成 TREC 檔，會有上千萬列（person 類別出現在將近一半的 COCO 圖片中），因此只存類別矩陣，指標直接在 numpy 上計算。`tests/test_metrics.py` 以 pytrec_eval（trec_eval 的 Python 綁定）逐 query 驗證兩者結果一致。`05_run_eval.py --write-runs` 也會輸出 TREC run 檔，可以再用官方 trec_eval 交叉檢查。

---

## 4. 執行順序

```bash
python scripts/01_download_data.py          # COCO 標註、Holidays（加 --with-roxford 另外下載 ROxford / RParis）
python scripts/02_prepare_data.py           # 抽樣、只下載用得到的 COCO 圖片、產生 edit / box query、寫 qrels
python scripts/03_extract_features.py       # CLIP / SigLIP / DINOv2 / pHash / tag（需要 GPU 比較快）
python scripts/04_build_indexes.py          # flat / hnsw / ivfpq / cp、pHash、tag inverted index
python scripts/05_run_eval.py               # 每個 query set 跑 eval.experiments
python scripts/05_run_eval.py --speed       # 索引比較（recall 損失、延遲、大小）
python scripts/07_roxford_supplementary.py  # supplementary（需要 --with-roxford）

python app/app.py                           # demo
```

輸出在 `results/{query_set}/`：

- `summary.csv`：每個設定的整體指標
- `summary_by_edit.csv`：只有 `coco_similar` 有，是逐 edit 類型的 robustness 分析
- `per_query/{config_id}.csv`：逐 query 指標，供 failure analysis 使用
- `summary_speed.csv`：`--speed` 的索引比較結果

### 4.1 Rocchio 的評估協定

1. 先跑去掉 rocchio 的設定，取得初次排序。
2. 以 qrels 模擬使用者，判斷前 10 名（等級 ≥ 1 視為相關）。
3. 帶著回饋重新搜尋。
4. 使用 **frozen ranking**：已判斷的前 10 名維持原位，只重排其後的結果。這是為了不把使用者已經看過的答案算成 Rocchio 的進步。

延遲包含兩輪搜尋。

### 4.2 人工標註（phone）

```bash
# 1. 把 100 張手機照片（JPEG / PNG；HEIC 請先轉檔）放進 data/phone/，再跑 02、03
python scripts/06_human_judgments.py pool    # 各系統前 10 名的聯集，打亂順序 → results/phone/judgments.csv
# 2. 兩位組員各自填 judge_a / judge_b 欄（0 不相關、1 部分相關、2 相關），可對照 judgments.html 看圖
python scripts/06_human_judgments.py kappa   # Cohen's kappa（一般、線性加權、二元）→ results/phone/kappa.json
python scripts/06_human_judgments.py qrels   # 依 judging.aggregate（預設 min）合併 → cache/sets/phone/qrels.txt
python scripts/05_run_eval.py --queries phone
```

- 標註時不要打開 `pool_systems.csv`（每張圖是哪些系統找到的），避免影響判斷。
- `pool` 不會覆寫已存在的 `judgments.csv`，避免蓋掉已完成的人工標註。
- 未進 pool 的圖視為不相關，與 trec_eval 的慣例相同。

### 4.3 Demo

- **Query**：上傳任意照片（也可以用剪貼簿或 webcam），或從評估用的 query set 選一張。圖上點兩下即可框選。
- **任務預設**：semantic 帶入 `siglip+bm25-flat-none`，similar 帶入 `dinov2-flat-none`（`demo.presets`）。
- **並排比較**：同一張 query 以兩組設定同時搜尋。
- **Explainability**：顯示 query 的 tag，以及每個結果與 query 共有的 tag。
- **Rocchio 回饋**：勾選結果中相關與不相關的圖後，重新搜尋。
- **Query set 的 query**：結果標示 qrels 等級，並顯示該 query 的 nDCG@10、AP、RR。
- **Failure analysis 分頁**：依指標由低到高列出 query，點一下就載入到搜尋頁。

---

## 5. 檔案目錄

```
├── configs/default.yaml        # 所有超參數與 eval.experiments
├── data/                       # [不進 git] coco/, holidays/, phone/, datasets/（ROxford）
├── cache/                      # [不進 git] sets/, features/, indexes/, text/, tags/
├── results/                    # 評估輸出（CSV 進 git）
├── src/
│   ├── data/                   # sets.py（manifest、qrels）, coco.py, holidays.py, edits.py, revisited.py
│   ├── models/                 # encoders.py, phash.py, tagger.py, names.py
│   ├── text/                   # inverted_index.py（HW1）, bm25.py（HW2）
│   ├── index/                  # faiss_index.py, cluster_pruning.py（HW2）
│   ├── rerank/                 # aqe.py, rocchio.py, mmr.py, rrf.py
│   ├── eval/                   # metrics.py, protocol.py, trec.py, revisited_eval.py
│   ├── preprocess/crop.py      # 框選裁切（線上與離線共用）
│   ├── features.py             # 特徵檔案位置、抽取與載入
│   ├── schema.py               # SearchRequest, SearchResult
│   └── pipeline.py             # 線上檢索流程（評估與 demo 共用）
├── scripts/01–07
├── app/                        # Gradio demo
└── tests/                      # 單元測試（合成資料，不需要 GPU 與模型權重）
```

---

## 6. 實作狀態

- 單元測試全部通過；`02 → 04 → 05 → 06` 已用假資料（隨機特徵）跑通整個流程。
- **尚未在真實資料與模型上執行過**：開發環境沒有 torch、模型權重與資料集，所以 `03`、`07` 與 demo 的線上抽特徵還沒有實際跑過。第一次執行時請依第 4 節的順序逐步確認。
- 已知風險：
  - Holidays 的官方下載位置（INRIA FTP）可能失效，`01` 會提示手動下載。
  - COCO 圖片是依 URL 逐張下載，中斷後重跑會跳過已下載的檔案。

---

## 7. 分工

| 組員 | 負責 |
|---|---|
| 陳長壻 | semantic search：tag + BM25（`src/text`、`src/models/tagger.py`）、RRF、Rocchio、alpha-QE、MMR（`src/rerank`）、評估 |
| 鄧存正 | similar search：DINOv2、pHash、edit（`src/data/edits.py`）、FAISS 與 cluster pruning（`src/index`）、demo（`app/`）、評估 |
| 共同 | 資料準備（`02`）、共用 pipeline（`src/pipeline.py`、`src/schema.py`）、人工標註（`06`）、期末報告 |

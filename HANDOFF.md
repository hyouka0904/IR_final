# HANDOFF

> 交接文件：每完成一個重要階段就更新。最後更新：2026-11-23（01 改為跨平台 Python 腳本）

## 任務目標

依新版一頁 proposal（`proposal.md`，Zero-Shot Image-to-Image Search Engine）改寫整個 repo：
README、proposal、source code。舊題目（ROxford 地標檢索的失敗模式分析）只保留 ROxford 當 supplementary run。

- 工作位置：worktree `../ir-final-zeroshot`，branch `feat/zero-shot-search`（從 `main` 的 `acae01b` 分出）
- 不在 `main` 上 commit
- sandbox 是 clone mode：commit 留在 sandbox 內，host 需要 `git fetch sandbox-<name>` 取回

## 使用者已確認的決定

1. worktree `../ir-final-zeroshot`、branch `feat/zero-shot-search`
2. 舊程式：保留 ROxford 資料讀取與 revisitop 評估當 supplementary；刪除 SIFT/RANSAC、inlier 快取、multi-scale、96 組 grid
3. `proposal.md` 直接替換成 PDF 內容（不擴寫）
4. HW1 inverted index、HW2 BM25、HW2 cluster pruning 在 repo 內重新實作，完整程度依設計需求決定
5. tag 用的 external class list：**未指定，列為待解決問題**（暫用可替換的預設）
6. 資料切分照預設（見 README）
7. 人工標註：proposal 有寫（pooled human judgments on 100 phone photos, kappa）。只提供 pooling 與 kappa／qrels 腳本，標註由組員用 CSV 人工完成，不做標註介面
8. README 改新題目；本機資料夾名稱不改

## 已完成

- [x] 階段 0：建 worktree、替換 `proposal.md`、建立本文件
- [x] 階段 1：核心模組＋單元測試
  - `configs/default.yaml`：新題目的全部超參數與各 query set 的實驗清單
  - 資料層：`src/data/sets.py`（manifest、CategoryQrels、SparseQrels）、`coco.py`、`holidays.py`、`edits.py`；`scripts/01_download_data.py`、`02_prepare_data.py`
  - 表示法：`src/models/encoders.py`（改用各模型官方前處理）、`phash.py`、`tagger.py`、`names.py`；`src/features.py`
  - 文字檢索：`src/text/inverted_index.py`（HW1，含布林 AND/OR 與 forward index）、`bm25.py`（HW2）
  - 索引：`src/index/cluster_pruning.py`（HW2）、`faiss_index.py`（加 cp 與 pHash 二元索引）
  - 後處理：`src/rerank/rocchio.py`、`mmr.py`、`rrf.py`（`aqe.py` 沿用）
  - `src/pipeline.py`、`src/schema.py` 改寫；`src/eval/metrics.py`（trec_eval 定義）、`trec.py`、`protocol.py`、`revisited_eval.py`
  - 刪除：SIFT/RANSAC、multi-scale、舊的 02–05 腳本、inlier 視覺化、舊困難 query 元件
- [x] 階段 2：離線腳本 03–07
  - `03_extract_features.py`：向量（圖庫 batch、query 逐張計時）、pHash、CLIP tag
  - `04_build_indexes.py`：各圖庫的 flat/hnsw/ivfpq/cp、pHash flat、tag inverted index，並記錄建立時間
  - `05_run_eval.py`：各 query set 的 summary / summary_by_edit / per_query / TREC run；`--speed` 跑索引比較（recall 損失、延遲、大小）
  - `06_human_judgments.py`：pool（盲標 CSV + HTML 對照表 + 另存系統來源）→ kappa → qrels
  - `07_roxford_supplementary.py`：bbx 裁切、revisitop compute_map
- [x] 階段 3：demo、README、requirements
  - `app/`：上傳（含剪貼簿 / webcam）或選 query set、框選、任務預設、並排比較、query tag 與命中 tag、Rocchio 勾選回饋、qrels 標記與單一 query 指標、failure analysis 分頁
  - README 全面改寫成新題目（資料、config_id、流程、指標、執行順序、Rocchio 協定、人工標註流程、demo、目錄、狀態、分工）
  - requirements：加 imagehash、pytest、pytrec_eval-terrier；移除 opencv、h5py、matplotlib

## 未完成

需要真實環境（torch、模型權重、資料集）才能做的：

- [ ] 實際跑 `01 → 07`，確認 03 / 07 / demo 的線上抽特徵在真實模型上正常
- [ ] 決定 tag 詞彙表（見待解決問題），換掉後要重跑 03（tags）、04、05
- [ ] 拍 100 張手機照片放 `data/phone/`，跑 06 pool → 兩人標註 → kappa → qrels → `05 --queries phone`
- [ ] 依實際結果調整超參數（IVF-PQ nlist/nprobe、cp b2、MMR λ、Rocchio β/γ、tf_scale）
- [ ] 期末報告

## 重要決策

- **Demo 預設只綁 127.0.0.1**（2026-11-23）：原本 `server_name="0.0.0.0"`，Gradio 印出 `http://0.0.0.0:7860`，在 Windows 瀏覽器開會出現 ERR_ADDRESS_INVALID。改成 `--host`（預設 127.0.0.1）、`--port`、`--share` 參數。
- **Windows 建置流程**（2026-11-23）：使用者在 Windows 上跑。README 新增 1.4（conda、PYTHONUTF8、torch CUDA 版本、faiss-cpu、路徑限制、常見問題）。程式面的修正：manifest 一律存 `/` 分隔的路徑（`rel_path` 用 `as_posix()`，cache 可在 Windows / Linux 間搬移），json 讀寫都指定 utf-8，judgments.html 的圖片路徑改用 `/`，讀 `judgments.csv` 用 `utf-8-sig`（Excel「CSV UTF-8」會加 BOM）。pytest、pytrec_eval-terrier 移到 `requirements-dev.txt`，在 Windows 若需要編譯而失敗，不會擋住主要套件的安裝。使用者照舊 README 用 `cu121` 裝到 torch 2.5.1，與 open_clip 4.x（需要 torch>=2.6）衝突；README 預設改成 `cu126`，並加上 nvidia-smi CUDA 版本 → wheel index 的對照表。
- **01 下載改用 Python**（2026-11-23）：使用者在 Windows PowerShell 執行 `bash scripts/01_download_data.sh` 失敗（`set: pipefail` 無效，CRLF 換行造成），且 Windows 通常沒有 curl/wget/unzip/tar 的 bash 環境。改成 `scripts/01_download_data.py`，只用標準函式庫（urllib、zipfile、tarfile）；ROxford / RParis 也自己下載攤平（revisitop 的 download.py 用 `os.system('wget …')`，Windows 不能用），revisitop 程式碼改下載 GitHub zip，不需要 git。02 / 04 / 05 / 07 缺前置資料時改成印出明確訊息並結束（原本 04 / 05 什麼都不印、07 直接 traceback）。

- **config_id = `{method}-{index}-{post}`**：method ∈ clip / siglip / dinov2 / phash / bm25 / {向量模型}+bm25；index ∈ flat / hnsw / ivfpq / cp（bm25 單獨時為 inv，phash 只有 flat）；post 為 none 或 aqe / rocchio / mmr 以 + 串接。不跑完整網格，每個 query set 跑哪些設定寫在 `eval.experiments`。
- **指標自己算、與 trec_eval 定義一致**：semantic qrels 展開後會有上千萬列（person 出現在約一半的 COCO 圖），所以只存類別矩陣（`CategoryQrels`）並在 numpy 上計算。`tests/test_metrics.py` 以 pytrec_eval 逐 query 驗證一致；`src/eval/trec.py` 可輸出 run / qrels 給官方 trec_eval 交叉驗證。
- **Semantic 分級**：共享類別數，上限 2（0/1/2），作為 nDCG 的 gain。只抽有標註的圖。
- **Box query（意圖消歧）**：從含 ≥2 類別的 semantic query 中選一個面積 ≥5% 的物件框起來，qrels 只看該類別；另有不裁切的 `coco_semantic_box_full` 對照組共用 qrels。裁切走與 demo 框選相同的 `crop_user_box`。
- **MMR 的多樣性指標**：nDCG 看不出多樣性，另外算 `cat_coverage_10`（前 10 名涵蓋 query 幾成類別）。
- **Rocchio 評估**：以 qrels 模擬使用者判斷前 10 名，並用 frozen ranking（已判斷的前 10 名不動，只重排其後），避免把使用者已看過的答案算成進步。
- **Similar query**：1,000 張圖庫圖片平均分配 5 種 edit（各 200 張），每張只套一種；JPEG 以外都存 PNG，避免壓縮失真混入。rotation 不擴大畫布、四角補黑。
- **Holidays**：獨立圖庫（`holidays_db`），500 個 query 在圖庫內，評估時以 `exclude_idx` 排除 query 本身。
- **BM25 文件**：一個 tag 就是一個 term；`tf_mode: weighted` 時 tf = max(1, round(10 × CLIP 機率))。用 binary 的話所有文件長度相同，BM25 會退化成 IDF 加權的交集。query 端 tf 以 k3 飽和。
- **pHash**：imagehash 64 bits，FAISS `IndexBinaryFlat`，分數 = 1 − hamming/64。
- **Cluster pruning**：隨機 √N 個 leader、b1=1、b2=4，與 IVF 的差別是不做 k-means。
- **延遲**：FAISS 執行緒數設為 1、全部 CPU；embed 延遲取自 03 腳本逐張量測。
- **人工標註盲標**：`judgments.csv` 不含系統來源，來源另存 `pool_systems.csv`；pool 順序打亂；`pool` 不覆寫已存在的標註檔。合併方式預設 min（兩人都認為相關才算）。
- **Demo 的 query 一律線上抽特徵**（即使來自 query set），延遲才是真實值；query set 的 query 會顯示 qrels 等級與前 K 名的 nDCG@10 / AP / RR（深度與評估的 1,000 不同，數值不能直接對照 summary）。
- **Demo 事件一律用 `.input`**，避免程式更新元件時連鎖觸發重設（例如從 failure analysis 載入 query）。

## 待解決問題

- **Tag 詞彙表**：proposal 只說 "a fixed external class list"。不能用 COCO 80 類（會跟 semantic qrels 的評分依據重疊，造成洩漏）。暫用 open_clip 內建的 ImageNet-1k 類別名稱，可在 config 換掉。

## 測試指令與結果

開發環境沒有 torch、模型權重與資料集，只能用合成資料跑單元測試（不需要 torch）。
測試用套件裝在 scratchpad：`pip install --target <dir> numpy faiss-cpu pillow pyyaml pandas scipy imagehash pytrec_eval-terrier pytest tqdm`

```bash
PYTHONPATH=<dir> python3 -m pytest -q -p no:cacheprovider tests
```

- 階段 3：單元測試 **43 passed**；demo 以假特徵（stub torch、替換線上抽特徵）逐一呼叫事件處理函式：semantic+並排、Rocchio 回饋、box query＋使用者框選、Holidays（排除 query 本身）、上傳圖片、failure analysis、不合法設定的錯誤訊息都正常；實際 `launch()` 後 `/` 與 `/config` 回 200。
- 階段 2：煙霧測試（假的 COCO / Holidays / phone 資料＋隨機特徵，代替需要 torch 的 03）：
  `02 → 04 → 05 → 05 --speed → 06 pool → （模擬兩位評審）→ 06 kappa → 06 qrels → 05 --queries phone` 全部跑通，
  輸出欄位正確；similar 的 query 向量 = 原圖 + 雜訊時，向量模型 mAP = 1，IVF-PQ / cp 有 recall 損失。產生假資料的程式在 scratchpad，未進 repo。
  03、07 需要 torch / open_clip / 模型權重，**尚未執行過**。
- 階段 1：**43 passed**（指標 vs pytrec_eval、BM25 公式、cluster pruning、各索引、pHash、alpha-QE/Rocchio/MMR/RRF、manifest/qrels、edit、pipeline 各設定、評估流程含 Rocchio frozen ranking）

## 下一步

程式部分已完成。下一步在有 GPU 的機器上依 README 第 4 節實際執行 01–07 與 demo，並處理「未完成」中列的項目。
本 branch 尚未 push；sandbox 為 clone mode，host 端用 `git fetch sandbox-<name> feat/zero-shot-search` 取回。

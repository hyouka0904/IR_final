# HANDOFF

最後更新：2026-11-23（階段 2：資料）

## 任務目標

使用者提供組員的 `ir_image_search-20261123.zip`，要求：

- **以這包為主軸重新開發**：功能重疊的部分以它為準，UI 以它為準。
- **我們多做的功能保留**（來自上一版 `feat/zero-shot-search`）。
- 一樣要能在 **Windows、Linux** 上建置網頁；**CI**；**檢查資料完整性**。
- **把功能都實作出來**：zip 的 README / proposal 中列為 planned 的項目全部完成。

## 工作位置

- worktree：`/home/user/projects/ir-final-upstream`，branch `feat/ir-image-search`（從 `feat/zero-shot-search` 分出，第一個 commit 以 zip 內容取代舊程式）。
- 舊版程式在 `feat/zero-shot-search`（worktree `../ir-final-zeroshot`），移植功能時參考。使用者的 Linux 伺服器目前用的是舊 branch（正在下載資料）。
- 程式碼流向：sandbox → Linux 伺服器（`git fetch sandbox-ir-final`，建置、開 demo）→ GitHub → Windows。

## zip 的架構（以它為準）

- `src/pipeline.py` `SearchEngine`：representation（clip / dinov2 / phash / tags / fusion / tags_nococo / fusion_nococo）、`tasks.<task>.allowed`、`add_image`（demo 加圖）。
- `src/sparse/`：HW1 postings 格式 + HW2 BM25（與 HW2 `weighting.py` parity test）。
- `src/tagging.py`：COCO + Open Images + Places365 詞彙表、softmax tf、`_nococo` 洩漏消融。
- `src/demo.py` + `app/app.py`：Gradio（ImageEditor 裁切、task、representation、k、query tags、加入圖庫、hard queries）。
- `src/pairs.py` / `scripts/06_pairs.py`：同主題照片對。
- `scripts/makezip.py`：打包給組員。
- 資料：`data/coco/{gallery,queries_semantic}.json` manifest，`cache/<rep>/{vectors,ids}.npy`、`cache/<rep>/flat/`、`cache/tags*/`。
- 程式風格：英文註解、README 有英文與繁中兩版。

## 要移植 / 實作的功能（計畫）

- [x] 階段 1：引擎擴充
  - SigLIP：`ClipEncoder(name="siglip")`（open_clip ViT-B-16-SigLIP / webli），兩個任務都允許
  - `src/index.py`：`FlatIndex`（原本的）、`ClusterPruningIndex`（HW2，√N 隨機 leader、b1 / b2）、`FaissANNIndex`（ivfpq / hnsw），共用介面（ntotal、ids、kind、add、search、save、load）；`build_index` / `load_index`（舊的 flat meta.json 沒有 kind 也能讀）；存在 `cache/<rep>/<kind>/`
  - `src/rerank.py`：`alpha_qe`、`rocchio`、`mmr`
  - `SearchRequest.index` / `rerank` / `feedback`；`SearchResponse.index` / `rerank` / `notes`
  - `SearchEngine`：`index_kinds(rep)`、`index_for(rep, kind)`（非 flat 的索引第一次用到才載入，載入時補上之前 demo 加入的圖）、`vectors_of`（從 flat 索引 reconstruct，re-ranker 用）、`_rerank`（aqe → rocchio → mmr，時間計入 rerank；aqe / rocchio 會重新檢索）；fusion 的 ANN 與 re-rank 作用在 clip 那條線；tags / phash 用 ANN 或 re-rank 會丟 ValueError
  - `scripts/04_build_index.py --index flat cluster ivfpq hnsw`，寫 `cache/<rep>/build.json`（建置時間、大小）；`scripts/search.py --index --rerank`
  - `resolve_device` 在沒有 torch 時回傳 cpu（CI、只跑搜尋的環境）
- [x] 階段 2：資料
  - `src/download.py`：從舊 branch 移植的接續下載 / 檢查修復（ensure_archive、extract_with_retry、image_problem），訊息改英文
  - `scripts/01_download.py` 重寫（沿用 zip 的抽樣邏輯與 manifest 格式）：自動下載標註 zip（zip 的版本要手動 curl）、COCO 子集（下載前先檢查已有圖片，壞圖刪掉重下）、Holidays（FTP，失敗不中斷）、`--with-roxford`（ROxford / RParis + revisitop 程式碼）、`--check-only`、`--skip-holidays`；寫 `gallery.json`（COCO + Holidays）、`queries_semantic.json`、`queries_box.json`、`queries_holidays.json`
  - `src/data.py`：`load_coco_instances` 解析時丟掉 segmentation（避免 train 標註吃掉數 GB，Windows 當機的原因）、`with_boxes`；`gallery_path`（COCO / Holidays）；`holidays_records`（id = 500,000,000 + 檔名數字）；`box_queries`
  - `src/integrity.py` + `scripts/check_data.py`：.part、標註數量、manifest（數量、id 唯一、query 不在圖庫、box 類別、Holidays 每組 ≥ 2）、所有圖片可讀、ROxford gnd、cache / 索引 / tag 索引是否涵蓋整個圖庫、手機照片
  - `IR_CONFIG` 環境變數可指定 config（冒煙測試、CI 用）；預期的資料集大小放在 config 的 `integrity:`
  - 02_embed / demo / search.py 改用 `gallery_path`
- [ ] 階段 3：評估
  - 設定 = representation × index × rerank；Rocchio 模擬回饋＋frozen ranking；ANN 的 recall loss、索引大小
  - Holidays、box query、手機照片人工標註（pool / kappa / qrels）、ROxford supplementary
- [ ] 階段 4：demo（zip 的 UI 為主）＋索引、重排、Rocchio 標記、並排比較、`--host` / `--auth`
- [ ] 階段 5：Windows / Linux 建置說明（README 兩語）、CI、冒煙測試

## 重要決策

- **data/ 結構與舊 branch 相容**：`data/coco/annotations`、`data/coco/{train2017,val2017}`、`data/holidays/jpg`、`data/datasets/<ds>` 都和舊 branch 一樣，伺服器上已下載的資料可以直接用；舊位置的 `data/coco/annotations_trainval2017.zip` 會自動搬到 zip 版本的位置 `data/annotations_trainval2017.zip`。
- **Holidays 併入圖庫**（依 zip 的 README）：`holidays.merge: true`。預設 2,000 張的小圖庫時 Holidays 佔比很高（1,491 張），擴大到 40k 時約 3.6%。Holidays 圖沒有 COCO 類別，semantic 評估中一律不相關。
- **使用者要求「download 先檢查 data 完不完整，再決定接續或重下」**（2026-11-23）：舊 branch 已先加上（commit b3344f6，讓伺服器馬上能用），新 branch 的 01 用同一套邏輯。
- **IVF-PQ 參數隨圖庫大小調整**：`nlist: null` = round(√N)（與 cluster pruning 的 leader 數同量級），nbits 不超過 log2(N)，m=32 可整除 384 / 512 / 768。預設圖庫只有 2,000 張，固定 nlist=256 會訓練不足。
- **ANN 索引延遲載入**：40k 張時每個 HNSW 約 100 MB，demo 啟動只載 flat；`_added` 紀錄 demo 加入的圖，延遲載入的索引會補上，結果才和 flat 一致。
- **Rocchio 沒有回饋時不報錯**：結果不變，`SearchResponse.notes` 說明。回饋中不在圖庫的 id 忽略。
- **dark mode**（使用者 2026-11-23 要求）：階段 4 在 demo 加切換。
- **新 branch**：伺服器正在用 `feat/zero-shot-search` 下載資料，避免中途改到它；新 branch 完成後再讓使用者切換。
- **cache / data 不進 git**：沿用 zip 的 `.gitignore`。

## 測試指令與結果

```bash
SP=/tmp/claude-1000/-home-user-projects-landmark-retrieval/f04e3a8f-3702-474e-8cc3-e040267ca451/scratchpad
PYTHONPATH=$SP/pylib:$SP/pylib_gr python3 -m pytest -q -p no:cacheprovider -m "not models"
```

- 階段 0（匯入後未修改）：132 passed、1 failed（`test_resolve_device_cpu_passthrough` 需要 torch，sandbox 沒有）。
- 階段 2：181 passed（新增 `test_download.py`（本機 HTTP server：接續、過小 / 過大檔案、check-only、全 0 封存檔重試、圖片檢查）、`test_integrity.py`、test_data 的 boxes / Holidays / box query / gallery_path）。另以假的 HTTP server 跑完整的 `01_download.py --with-roxford`：全新下載 → 製造損壞（截斷的圖庫圖、空的 Holidays 圖、刪掉的 ROxford 圖、被截斷且已改名的標註 zip、截斷的 val JSON）→ `--check-only` 不改任何檔案 → 修復 → 再 check 全部完整、沒有 .part；`check_data.py` passed 22 / skipped 7 / failed 0。
- 階段 1：162 passed（新增 `test_index_kinds.py`、`test_engine_rerank.py`、re-ranker 與 SearchRequest 的測試）。zip 附的真實 cache（2,000 張 CLIP / DINOv2 / pHash）跑 `04_build_index.py --index flat cluster ivfpq hnsw` 成功（IVF-PQ 0.68 MB vs flat 3.9 MB）。

## 下一步

階段 3：評估（設定組合、Rocchio 協定、ANN recall loss、Holidays / box / phone / ROxford）。

# HANDOFF

最後更新：2026-11-23（階段 1：引擎擴充）

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
- [ ] 階段 2：資料
  - 01 自動下載標註（接續下載、大小檢查）、Holidays（併入圖庫）、ROxford / RParis
  - box query（意圖消歧）
  - 資料完整性檢查（`scripts/check_data.py`）
- [ ] 階段 3：評估
  - 設定 = representation × index × rerank；Rocchio 模擬回饋＋frozen ranking；ANN 的 recall loss、索引大小
  - Holidays、box query、手機照片人工標註（pool / kappa / qrels）、ROxford supplementary
- [ ] 階段 4：demo（zip 的 UI 為主）＋索引、重排、Rocchio 標記、並排比較、`--host` / `--auth`
- [ ] 階段 5：Windows / Linux 建置說明（README 兩語）、CI、冒煙測試

## 重要決策

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
- 階段 1：162 passed（新增 `test_index_kinds.py`、`test_engine_rerank.py`、re-ranker 與 SearchRequest 的測試）。zip 附的真實 cache（2,000 張 CLIP / DINOv2 / pHash）跑 `04_build_index.py --index flat cluster ivfpq hnsw` 成功（IVF-PQ 0.68 MB vs flat 3.9 MB）。

## 下一步

階段 2：資料（01 下載 + 檢查修復、Holidays 併入圖庫、box query、check_data）。

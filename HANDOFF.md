# HANDOFF

> 交接文件：每完成一個重要階段就更新。最後更新：2026-11-18（階段 0）

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

## 未完成

- [ ] README 改寫（設計文件）
- [ ] 資料層：COCO 抽樣、edit 產生、box query、Holidays、phone、qrels
- [ ] 表示法：CLIP / SigLIP / DINOv2 encoder、pHash、CLIP tagger
- [ ] 文字檢索：inverted index、BM25
- [ ] 索引：Flat / HNSW / IVF-PQ / cluster pruning
- [ ] 後處理：alpha-QE、Rocchio、MMR、RRF
- [ ] 共用 pipeline、評估指標、評估腳本
- [ ] 人工標註腳本（pool / kappa / qrels）
- [ ] ROxford supplementary 腳本
- [ ] Gradio demo
- [ ] 單元測試

## 重要決策

（隨階段補上）

## 待解決問題

- **Tag 詞彙表**：proposal 只說 "a fixed external class list"。不能用 COCO 80 類（會跟 semantic qrels 的評分依據重疊，造成洩漏）。暫用 open_clip 內建的 ImageNet-1k 類別名稱，可在 config 換掉。

## 測試指令與結果

（隨階段補上）

## 下一步

改寫 README 作為設計文件，然後依模組實作。

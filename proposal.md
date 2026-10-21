# 期末專題 Proposal（修正版）

## 題目

**地標圖搜圖系統的失敗模式分析：從 global embedding 到 re-ranking 的逐層補救與互動式診斷工具**

## 1. 摘要

本專題實作一套以圖搜圖（image-to-image retrieval）軟體，以公開 benchmark ROxford / RParis 作為圖庫，系統化比較以下元件的組合：

- 三種預訓練視覺模型：CLIP、SigLIP、DINOv2
- 三種 FAISS 索引：Flat、HNSW、IVF-PQ
- 兩種 re-ranking：alpha-QE、SIFT + RANSAC 幾何驗證

本專題要回答的問題是：**global embedding 在地標檢索中會因什麼原因失敗？每一種補救手段修好了哪些 query，又付出多少延遲與記憶體成本？** 成果包含一份依官方協定評估的實驗報告，以及一個能逐個 query 診斷結果的互動 demo。

## 2. 背景與動機

圖像檢索（CBIR）長期面對兩個核心難題：intention gap 與 semantic gap。近年的主流做法是直接拿預訓練 foundation model 的 image embedding 做 nearest-neighbor search，但不同模型的特性並不相同：

- Self-supervised 模型（如 DINOv2）通常更適合 near-duplicate 與 instance retrieval，因為它著重像素與幾何，而非概念。
- CLIP 有「CLIP-blind pairs」問題：明顯不同的圖會被編成幾乎一樣的向量，而且缺乏空間結構編碼。
- 有研究把 CLIP、SigLIP、DINOv2 的相似度分數合併，結果比單一模型更穩健。

此外，兩階段檢索是常見流程：先用全域向量取得候選，再做 re-ranking。常用的 re-ranking 手段包含 geometric verification（RANSAC）與 query expansion，但它們的計算成本在大規模集合上是一項挑戰。因此「準確度、延遲、記憶體」三者的取捨值得實際量測。

## 3. 相關工作與參考的現有專案

我們參考了以下開源專案，並從中取用設計點子：

| 專案 | 我們參考的部分 |
|---|---|
| CLIPImageFinder | OpenCLIP + FAISS 的本地檢索架構，以及索引持久化（index 存檔）的設計 |
| CBIR_System（課程專題） | 以 Oxford5k / Paris6k 為資料集的網頁系統，以及讓使用者框選區域當 query 的功能 |
| naver/deep-image-retrieval | ROxford / RParis 的評估流程，以及 alpha-query expansion 與 multi-scale 的參數設計 |

在此基礎上，本專題另外加入四項內容：三個模型與 ensemble、多種 ANN 索引、幾何驗證，以及以失敗模式為主軸的逐 query 診斷介面。

## 4. 研究問題

主問題：global embedding 檢索在地標資料上有哪些典型失敗模式？各種補救手段的效益與成本為何？

| 失敗模式 | 對應的補救手段 | 待驗證的假設 |
|---|---|---|
| 語意型模型缺乏細節與幾何辨識力 | 改用 DINOv2、ensemble | H1：DINOv2 在 Hard 協定上的 mAP 優於 CLIP / SigLIP |
| 官方 query 常是建築的局部，放大倍率極端 | multi-scale | H2：multi-scale 能提升 Hard 協定的 mAP |
| 單一全域向量沒有空間結構 | SIFT + RANSAC 幾何驗證 | H3：幾何驗證能修正 top-K 中視覺相似、但不是同一地標的結果，提升 mP@k |
| 單一 query 資訊不足 | alpha-QE | H4：alpha-QE 能提升 mAP，且對 Hard 協定的幫助最大 |
| 補救後是否仍可用 | Flat / HNSW / IVF-PQ | H5：HNSW 以更多記憶體換取搜尋速度；IVF-PQ 以壓縮大幅降低記憶體；兩者相對 Flat 的 recall 損失都很小 |

以上皆為待驗證的假設。若實驗結果不支持，同樣會如實寫入報告。

## 5. 資料集

使用 **ROxford 與 RParis**（Revisited Oxford / Paris）：

- 任務是：給定一張 query，找出資料庫中拍到同一地標的所有圖片。
- 原始 Oxford 與 Paris 分別有 5,063 與 6,392 張高解析度圖片。Revisited 版本把 70 個 query 從資料庫中分出來，因此官方資料庫（`imlist`）為 **ROxford 4,993 張、RParis 6,322 張**。
- 兩個資料集各有 70 個 query，並附重新標註的 ground truth，提供 Easy / Medium / Hard 三種評估協定。
- 官方提供標註檔、query 的 bounding box 與評估程式。
- 官方協定要求 query 必須以 bounding box 裁切，本專題所有正式實驗都遵守此規則。

評估與 demo 使用同一個圖庫。不使用 R1M distractor，因為其規模超出本專題範圍。

## 6. 系統設計

### 離線階段

所有耗時運算都在離線階段完成，demo 啟動時直接載入結果。

1. 對圖庫與官方 query（已用 bbox 裁切），以三個模型各抽取兩份 embedding：單一尺度與 multi-scale。
2. 對每份圖庫 embedding 建立 Flat、HNSW、IVF-PQ 三種索引並存檔。IVF-PQ 的分群與量化 codebook 以圖庫本身訓練，不屬於模型訓練。
3. 抽取圖庫所有圖片的 SIFT 特徵並存檔，供幾何驗證使用。
4. 跑完所有設定組合的評估，存下每個設定的整體指標與逐 query AP。逐 query AP 是困難 query 清單的資料來源。

### 線上階段

1. **Query 輸入**：上傳圖片、選擇官方 query，或在圖上框選區域。
2. **前處理**：依來源做 bbox 或框選裁切，再依設定決定是否使用 multi-scale。
3. **alpha-QE（可選）**：每個模型各自以初次檢索的前 n 名擴展 query 向量。
4. **全域檢索**：每個模型在自己的索引上取 top-K。
5. **Ensemble（可選）**：取三個模型候選的聯集，用存下的 embedding 精確重算每個模型的 cosine 分數，加權融合後取 top-K。這樣可避免某個候選在其他模型的索引中沒有分數。
6. **幾何驗證（可選）**：對 top-K 候選做 SIFT 比對 + RANSAC，依 inlier 數重新排序。圖庫的 SIFT 特徵在離線階段已經算好。
7. **輸出**：排序結果、分數、各階段延遲。

K 為超參數，初始值設 100。

### 確定實作的功能（共 15 項）

**檢索核心**

1. 三個模型（CLIP、SigLIP、DINOv2），以及三者分數加權融合的 ensemble
2. 三種 FAISS 索引：Flat、HNSW、IVF-PQ
3. Query 前處理：官方 bounding box 裁切、使用者框選裁切、multi-scale
4. Re-ranking：alpha-QE、SIFT + RANSAC 幾何驗證

**評估**

5. 依官方協定，在 ROxford 與 RParis 上分別計算 Easy / Medium / Hard 的 mAP 與 mP@k
6. 延遲、索引大小，以及 ANN 索引相對 Flat 的 recall 損失。recall 損失定義為：ANN 回傳的 top-K 與 Flat 回傳的 top-K 的重疊比例
7. 完整組合網格：模型（含 ensemble）× multi-scale 開關 × 索引 × re-ranking，每個資料集共 96 組
8. 失敗案例分析：整理各設定下 AP 最低的 query

**Demo（Gradio）**

9. Query 來源：上傳圖片、選擇官方 query、在圖上框選區域
10. 設定面板：模型、索引、re-ranking、multi-scale
11. 結果顯示相似度分數與各階段延遲。選擇官方 query 時，另外標示 ground truth（positive / junk / negative）與該次查詢的 AP
12. 並排比較：同一個 query 以兩組設定同時搜尋，左右並列
13. 幾何驗證視覺化：畫出 query 與結果圖之間的 SIFT inlier 匹配連線
14. 困難 query 清單：依選定設定列出 AP 最低的官方 query，點選後直接載入搜尋介面
15. 圖庫切換：ROxford 與 RParis

### 確定不做

R1M、模型訓練或 fine-tuning、DINOv3、以文搜圖、增量新增圖片、線上部署、embedding 降維視覺化。

## 7. 實驗設計

完整網格（功能 7）全部跑完，報告中以逐層疊加的消融路徑呈現。每一步都在 Easy / Medium / Hard 三種協定下報告：

1. **Baseline**：三個模型各自搭配 Flat 索引，query 以官方 bbox 裁切，單一尺度，無 re-ranking
2. **+ Multi-scale**
3. **+ Ensemble**：三模型分數融合
4. **+ alpha-QE**
5. **+ 幾何驗證**
6. **索引比較**：在最佳設定上，將 Flat 換成 HNSW 與 IVF-PQ，比較 mAP、recall 損失、延遲與索引大小。若最佳設定包含 ensemble，三個模型的索引同時替換

**補充實驗**：不做 bbox 裁切的 baseline，用來說明裁切的影響。此設定不符合官方協定，報告中會單獨標示。

**指標**

- 檢索品質：mAP、mP@k（k = 1, 5, 10，沿用官方評估程式）
- 成本：索引大小，以及分階段量測的延遲（embedding 抽取、索引搜尋、re-ranking）
- FAISS 的 HNSW 只支援 CPU，因此索引搜尋延遲一律在 CPU 上量測，以確保三種索引的比較公平

**分析**：逐 query 比較各設定的 AP 變化，整理兩類 query，並據此歸納失敗模式：

- 被某個補救手段救回的 query，以及救回它的是哪個手段
- 所有設定都失敗的 query

**成本控制**：幾何驗證的 inlier 數只取決於（query, 圖庫圖片）這一對，與模型和索引無關。因此每一對只算一次並快取，供所有設定共用。

## 8. Demo 展示規劃

Demo 的定位是逐個 query 的診斷工具。展示流程如下：

1. 從困難 query 清單選一個 query
2. 在 baseline 設定下看到錯誤的結果
3. 用並排比較切換成補救後的設定
4. 以 inlier 連線說明幾何驗證為何修正了排序
5. 顯示該 query 的 AP 變化

## 9. 預期產出

1. 可執行的 demo 軟體與使用說明
2. 完整的實驗報告，包含上述消融實驗與失敗案例分析
3. 預先建立好的 embedding、索引、SIFT 特徵與逐 query 評估結果，供 demo 直接載入

## 10. 分工

- **組員 A（檢索核心）**：embedding 抽取（含 multi-scale）、FAISS 索引、ensemble、re-ranking、評估腳本與實驗
- **組員 B（系統層）**：Gradio 介面、query 前處理模組（裁切、框選、multi-scale）、並排比較、inlier 視覺化、困難 query 清單

前處理模組由 B 實作；A 在離線階段直接呼叫同一份模組，確保線上與離線的前處理一致。雙方共同定義介面串接處的資料格式。報告由各自負責的部分撰寫，實驗分析共同完成。（組員姓名待填）

## 11. 限制與風險

- ROxford / RParis 各只有約五、六千張圖。在此規模下，HNSW 相對 Flat 的延遲優勢會很小，因此索引比較將以 recall 損失與記憶體為主，並在報告中如實說明。
- 預訓練模型未針對地標領域微調，結果受限於模型本身的能力。
- 幾何驗證只對 top-K 候選執行，K 值會同時影響成本與效果，將作為超參數在實驗中報告。
- 完整網格中含幾何驗證的設定較耗時；透過 inlier 快取控制成本，若仍不足，會優先保留消融路徑上的設定。

## 12. 參考資料

- Radenović et al., *Revisiting Oxford and Paris: Large-Scale Image Retrieval Benchmarking*, CVPR 2018（arXiv:1803.11285）
- Zhou, Li, Tian, *Recent Advance in Content-based Image Retrieval: A Literature Survey*（arXiv:1706.06064）
- *ZSE-Cap: A Zero-Shot Ensemble for Image Retrieval and Prompt-Guided Captioning*（arXiv:2507.20564）
- *ILIAS: Instance-Level Image retrieval At Scale*（arXiv:2502.11748）
- GitHub：sHIRo-da/CLIPImageFinder、ShenggKai/CBIR_System、naver/deep-image-retrieval

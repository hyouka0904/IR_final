"""組員 A / B 共同定義的資料格式（SearchRequest, SearchResult）。"""
from dataclasses import dataclass, field

import numpy as np
from PIL import Image


@dataclass
class SearchRequest:
    dataset: str                                    # roxford5k | rparis6k
    config_id: str                                  # 例如 dinov2-ms-flat-aqe+gv
    image: Image.Image | None = None                # 上傳的圖；選官方 query 時可留空
    qidx: int | None = None                         # 官方 query 編號；有值時讀官方圖並可使用 inlier 快取
    crop: bool = True                               # 官方 query 是否用 bbx 裁切；False 只用於補充實驗
    user_box: tuple[int, int, int, int] | None = None  # 使用者框選區域（原圖座標）
    k: int | None = None                            # None 表示使用 config 的 retrieval.k
    full_ranking: bool = False                      # 評估用：附上整個圖庫的排序


@dataclass
class SearchResult:
    config_id: str
    ids: np.ndarray                                 # 圖庫索引（對應 gnd['imlist']），長度 K
    scores: np.ndarray                              # 全域相似度分數（ens 時為加權融合分數）
    latency_ms: dict[str, float]                    # embed / search / rerank
    query_image: Image.Image | None = None          # 實際送進模型的 query（已裁切）
    inliers: np.ndarray | None = None               # 有啟用幾何驗證時，每個結果的 inlier 數
    full_ranking: np.ndarray | None = None          # (n_db,)，前 K 名同 ids，其後依精確分數排序
    extra: dict = field(default_factory=dict)

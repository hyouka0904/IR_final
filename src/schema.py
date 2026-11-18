"""組員共同定義的資料格式（SearchRequest, SearchResult）；評估腳本與 demo 都透過這兩個類別呼叫 pipeline。"""
from dataclasses import dataclass, field

import numpy as np
from PIL import Image


@dataclass
class SearchRequest:
    config_id: str                                  # 例如 siglip+bm25-flat-aqe+mmr（格式見 README 3.1）
    image: Image.Image | None = None                # 上傳的圖；評估時可改傳快取的特徵
    user_box: tuple[float, float, float, float] | None = None   # 框選區域（原圖座標）；box cropping
    k: int | None = None                            # 回傳幾名；None 表示 retrieval.k，評估時用 retrieval.depth
    feedback: tuple[list[int], list[int]] | None = None         # Rocchio：(相關, 不相關) 的圖庫索引
    exclude: list[int] = field(default_factory=list)            # 不得出現在結果中的圖庫索引（Holidays 的 query 本身）


@dataclass
class SearchResult:
    config_id: str
    ids: np.ndarray                                 # 圖庫索引，依名次排序
    scores: np.ndarray                              # 最後一個階段的分數（cosine / BM25 / RRF / MMR）
    latency_ms: dict[str, float]                    # embed / search / rerank
    query_image: Image.Image | None = None          # 實際送進模型的 query（已裁切）
    query_tags: list[tuple[str, float]] | None = None           # 有用到 BM25 時：query 的 CLIP tag 與機率
    extra: dict = field(default_factory=dict)

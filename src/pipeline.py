"""線上檢索流程（README 3.2），評估腳本與 demo 共用同一份。"""
from src.schema import SearchRequest, SearchResult


def parse_config_id(config_id: str) -> dict:
    ...


class Pipeline:
    def __init__(self, dataset: str, cfg: dict):
        ...

    def search(self, request: SearchRequest) -> SearchResult:
        ...

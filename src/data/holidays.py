"""INRIA Holidays：1,491 張、500 組；每組第一張（檔名以 00.jpg 結尾）是 query。

標準協定是 query 對全部 1,491 張檢索並排除 query 本身，同組的其他圖為 relevant。
目錄：data/holidays/jpg/*.jpg
"""
import os


def image_names(data_root: str) -> list[str]:
    d = os.path.join(data_root, "holidays", "jpg")
    if not os.path.isdir(d):
        return []
    return sorted(f for f in os.listdir(d) if f.lower().endswith(".jpg"))


def image_path(data_root: str, name: str) -> str:
    return os.path.join(data_root, "holidays", "jpg", name)


def group_of(name: str) -> str:
    return name[:4]


def is_query(name: str) -> bool:
    return name[4:6] == "00"

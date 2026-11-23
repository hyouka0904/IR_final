"""Load configs/default.yaml; every tunable parameter comes from here."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = ROOT / path
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(cfg: dict, key: str) -> Path:
    """Absolute path for cfg['paths'][key], relative paths taken from the repo root."""
    p = Path(cfg["paths"][key])
    return p if p.is_absolute() else ROOT / p

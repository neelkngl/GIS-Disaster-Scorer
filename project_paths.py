from pathlib import Path
from typing import Optional


def find_repo_root(start: Optional[Path] = None) -> Path:
    start = Path(start or Path(__file__).resolve().parent)
    p = start
    # Look for common repository markers
    while True:
        if (p / "README.md").exists() or (p / "datasets").exists() or (p / "data").exists():
            return p
        if p == p.parent:
            return start
        p = p.parent


def data_dir(root: Optional[Path] = None) -> Path:
    root = Path(find_repo_root(root))
    if (root / "datasets").exists():
        return root / "datasets"
    return root / "data"


def raw_dir(root: Optional[Path] = None) -> Path:
    d = data_dir(root)
    # support both data/raw and datasets/raw
    if (d / "raw").exists():
        return d / "raw"
    return d


def processed_dir(root: Optional[Path] = None) -> Path:
    d = data_dir(root)
    if (d / "processed").exists():
        return d / "processed"
    # fallback to data/processed
    return d / "processed"


def out_dir(root: Optional[Path] = None) -> Path:
    # prefer data/out, datasets/out, else data/processed/out
    d = data_dir(root)
    if (d / "out").exists():
        return d / "out"
    return d / "out"

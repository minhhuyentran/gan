import os
import yaml
from dataclasses import dataclass

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

def ensure_dirs(*dirs: str):
    for d in dirs:
        os.makedirs(d, exist_ok=True)

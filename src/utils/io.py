import json
import joblib

def save_json(obj, path: str):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)

def save_pkl(obj, path: str):
    joblib.dump(obj, path)

def load_pkl(path: str):
    return joblib.load(path)

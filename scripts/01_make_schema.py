import os
from src.utils.config import load_config, ensure_dirs
from src.utils.io import save_json
from src.data.load import load_all_days
from src.data.schema import infer_schema

def main():
    cfg = load_config("configs/base.yaml")
    out_dir = cfg["paths"]["out_dir"]
    art_dir = cfg["paths"]["artifacts_dir"]
    ensure_dirs(out_dir, art_dir)

    df = load_all_days(cfg["paths"]["raw_dir"])

    schema = infer_schema(
        df,
        label_col=cfg["data"]["label_col"],
        timestamp_col=cfg["data"]["timestamp_col"],
        protocol_col=cfg["data"]["protocol_col"],
        drop_cols=cfg["data"]["drop_cols"]
    )
    save_json(schema, os.path.join(art_dir, "schema.json"))
    print("Saved schema:", os.path.join(art_dir, "schema.json"))

if __name__ == "__main__":
    main()

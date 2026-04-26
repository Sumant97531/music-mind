"""
app/models/model_loader.py
Loads all pipeline artifacts once at startup and caches them.
Works both locally and on Streamlit Cloud.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import load_npz

# Resolve data dir relative to THIS file:
# This file lives at  <repo_root>/app/models/model_loader.py
# So parent.parent.parent == repo root
BASE_DIR      = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

_cache: dict = {}


def load_all() -> dict:
    """
    Load all pipeline outputs from data/processed/.
    Cached in memory after the first call.
    """
    global _cache
    if _cache:
        return _cache

    required = [
        "cleaned_data.csv",
        "collab_filtered_data.csv",
        "transformed_hybrid_data.npz",
        "interaction_matrix.npz",
        "track_ids.npy",
    ]
    for fname in required:
        path = PROCESSED_DIR / fname
        if not path.exists():
            raise FileNotFoundError(
                f"Missing processed file: {path}\n"
                "Run  python scripts/build_pipeline.py  first."
            )

    _cache = {
        "songs_data":         pd.read_csv(PROCESSED_DIR / "cleaned_data.csv"),
        "filtered_data":      pd.read_csv(PROCESSED_DIR / "collab_filtered_data.csv"),
        "transformed_matrix": load_npz(str(PROCESSED_DIR / "transformed_hybrid_data.npz")),
        "interaction_matrix": load_npz(str(PROCESSED_DIR / "interaction_matrix.npz")),
        "track_ids":          np.load(str(PROCESSED_DIR / "track_ids.npy"), allow_pickle=True),
    }

    print(f"[model_loader] songs_data        : {len(_cache['songs_data'])} rows")
    print(f"[model_loader] filtered_data     : {len(_cache['filtered_data'])} rows")
    print(f"[model_loader] transformed_matrix: {_cache['transformed_matrix'].shape}")
    print(f"[model_loader] interaction_matrix: {_cache['interaction_matrix'].shape}")
    print(f"[model_loader] track_ids         : {len(_cache['track_ids'])}")
    return _cache
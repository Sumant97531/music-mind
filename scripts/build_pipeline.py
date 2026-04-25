"""
scripts/build_pipeline.py
─────────────────────────
Runs the full data pipeline from raw CSVs → ready-to-serve .npz files.

Steps
-----
1. Clean raw Music Info.csv → data/processed/cleaned_data.csv
2. Train transformer + transform cleaned data → models/transformer.joblib
                                              → data/processed/transformed_data.npz
3. Build interaction matrix from User Listening History.csv
        → data/processed/track_ids.npy
        → data/processed/collab_filtered_data.csv
        → data/processed/interaction_matrix.npz
4. Transform the collab-filtered data (used by hybrid recommender at runtime)
        → data/processed/transformed_hybrid_data.npz

Run once before starting the app:
    python scripts/build_pipeline.py
"""
import sys
from pathlib import Path

# Make project root importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.services.data_cleaning  import run as step1_clean
from app.services.content_based  import run as step2_content
from app.services.collaborative  import run as step3_collab
from app.services.data_cleaning  import data_for_content_filtering
from app.services.content_based  import transform_data, save_transformed_data

import pandas as pd

PROCESSED_DIR = ROOT / "data" / "processed"
HYBRID_NPZ    = PROCESSED_DIR / "transformed_hybrid_data.npz"


def step4_transform_filtered():
    print("\n── Step 4: Transform collab-filtered data ───────────────────────")
    filtered = pd.read_csv(PROCESSED_DIR / "collab_filtered_data.csv")
    print(f"  Loaded collab_filtered_data: {len(filtered)} rows")
    content_ready = data_for_content_filtering(filtered)
    transformed   = transform_data(content_ready)
    save_transformed_data(transformed, HYBRID_NPZ)


def main():
    print("═" * 60)
    print("  Music Mind — Pipeline Builder")
    print("═" * 60)

    print("\n── Step 1: Clean raw data ───────────────────────────────────────")
    step1_clean()

    print("\n── Step 2: Train transformer + transform full data ──────────────")
    step2_content()

    print("\n── Step 3: Build interaction matrix ─────────────────────────────")
    step3_collab()

    step4_transform_filtered()

    print("\n═" * 60)
    print("  Pipeline complete. Run the app:")
    print("  Streamlit → streamlit run app/streamlit_app.py")
    print("  FastAPI   → uvicorn app.main:app --reload")
    print("═" * 60)


if __name__ == "__main__":
    main()
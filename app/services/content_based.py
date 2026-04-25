#%%

"""
app/services/content_based.py
Trains the ColumnTransformer on cleaned_data and saves:
  - models/transformer.joblib
  - data/processed/transformed_data.npz
Run via:  python scripts/build_pipeline.py  (step 2)
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from category_encoders.count import CountEncoder
from scipy.sparse import save_npz, load_npz
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

BASE_DIR              = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR         = BASE_DIR / "data" / "processed"
MODELS_DIR            = BASE_DIR / "models"

CLEANED_DATA_PATH     = PROCESSED_DIR / "cleaned_data.csv"
TRANSFORMED_DATA_PATH = PROCESSED_DIR / "transformed_data.npz"
TRANSFORMER_PATH      = MODELS_DIR    / "transformer.joblib"

# ── Column groups ─────────────────────────────────────────────────────────────
FREQUENCY_ENCODE_COLS = ["year"]
OHE_COLS              = ["artist", "time_signature", "key"]
TFIDF_COL             = "tags"
STANDARD_SCALE_COLS   = ["duration_ms", "loudness", "tempo"]
MIN_MAX_SCALE_COLS    = [
    "danceability", "energy", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence",
]


def train_transformer(data: pd.DataFrame) -> None:
    """Fit ColumnTransformer on content features and persist to disk."""
    MODELS_DIR.mkdir(exist_ok=True)
    transformer = ColumnTransformer(
        transformers=[
            ("freq_encode",   CountEncoder(normalize=True, return_df=True), FREQUENCY_ENCODE_COLS),
            ("ohe",           OneHotEncoder(handle_unknown="ignore"),        OHE_COLS),
            ("tfidf",         TfidfVectorizer(max_features=85),              TFIDF_COL),
            ("std_scale",     StandardScaler(),                              STANDARD_SCALE_COLS),
            ("minmax_scale",  MinMaxScaler(),                                MIN_MAX_SCALE_COLS),
        ],
        remainder="passthrough",
        n_jobs=-1,
        force_int_remainder_cols=False,
    )
    transformer.fit(data)
    joblib.dump(transformer, TRANSFORMER_PATH)
    print(f"Transformer saved → {TRANSFORMER_PATH}")


def transform_data(data: pd.DataFrame):
    """Load persisted transformer and return sparse feature matrix."""
    transformer = joblib.load(TRANSFORMER_PATH)
    return transformer.transform(data)


def save_transformed_data(transformed, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_npz(str(save_path), transformed)
    print(f"Transformed data saved → {save_path}  shape={transformed.shape}")


def content_recommendation(
    song_name: str,
    artist_name: str,
    songs_data: pd.DataFrame,
    transformed_data,
    k: int = 10,
) -> pd.DataFrame:
    song_name   = song_name.lower()
    artist_name = artist_name.lower()

    mask = (songs_data["name"] == song_name) & (songs_data["artist"] == artist_name)
    matches = songs_data.loc[mask]
    if matches.empty:
        raise ValueError(f"Song '{song_name}' by '{artist_name}' not found.")

    idx    = matches.index[0]
    vec    = transformed_data[idx].reshape(1, -1)
    sims   = cosine_similarity(vec, transformed_data)
    top_k  = np.argsort(sims.ravel())[-k - 1:][::-1]

    return (
        songs_data.iloc[top_k][["name", "artist", "spotify_preview_url"]]
        .reset_index(drop=True)
    )


def run() -> None:
    from app.services.data_cleaning import data_for_content_filtering
    df      = pd.read_csv(CLEANED_DATA_PATH)
    content = data_for_content_filtering(df)
    train_transformer(content)
    transformed = transform_data(content)
    save_transformed_data(transformed, TRANSFORMED_DATA_PATH)


if __name__ == "__main__":
    run()
#%%

"""
app/services/collaborative.py
Builds the track×user interaction matrix from User Listening History.csv.
Saves:
  - data/processed/track_ids.npy
  - data/processed/collab_filtered_data.csv
  - data/processed/interaction_matrix.npz
Run via:  python scripts/build_pipeline.py  (step 3)
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR              = Path(__file__).resolve().parent.parent.parent
RAW_DIR               = BASE_DIR / "data" / "raw"
PROCESSED_DIR         = BASE_DIR / "data" / "processed"

SONGS_DATA_PATH        = PROCESSED_DIR / "cleaned_data.csv"
USER_HISTORY_PATH      = RAW_DIR       / "User Listening History.csv"

TRACK_IDS_PATH         = PROCESSED_DIR / "track_ids.npy"
FILTERED_DATA_PATH     = PROCESSED_DIR / "collab_filtered_data.csv"
INTERACTION_MATRIX_PATH = PROCESSED_DIR / "interaction_matrix.npz"


def filter_songs_data(
    songs_data: pd.DataFrame,
    track_ids: list,
    save_path: Path,
) -> pd.DataFrame:
    """Keep only songs that appear in the listening history."""
    filtered = (
        songs_data[songs_data["track_id"].isin(track_ids)]
        .sort_values("track_id")
        .reset_index(drop=True)
    )
    filtered.to_csv(save_path, index=False)
    print(f"Filtered songs saved → {save_path}  ({len(filtered)} rows)")
    return filtered


def create_interaction_matrix(
    history_data: pd.DataFrame,
    track_ids_save_path: Path,
    matrix_save_path: Path,
) -> csr_matrix:
    """Build a sparse track×user playcount matrix."""
    df = history_data.copy()
    df["playcount"] = df["playcount"].astype(np.float64)
    df["user_id"]   = df["user_id"].astype("category")
    df["track_id"]  = df["track_id"].astype("category")

    user_codes  = df["user_id"].cat.codes
    track_codes = df["track_id"].cat.codes
    track_ids   = df["track_id"].cat.categories.values

    np.save(str(track_ids_save_path), track_ids, allow_pickle=True)
    print(f"Track IDs saved → {track_ids_save_path}")

    df = df.assign(user_idx=user_codes, track_idx=track_codes)

    agg = (
        df.groupby(["track_idx", "user_idx"])["playcount"]
        .sum()
        .reset_index()
    )

    matrix = csr_matrix(
        (agg["playcount"], (agg["track_idx"], agg["user_idx"])),
        shape=(df["track_idx"].nunique(), df["user_idx"].nunique()),
    )
    save_npz(str(matrix_save_path), matrix)
    print(f"Interaction matrix saved → {matrix_save_path}  shape={matrix.shape}")
    return matrix


def collaborative_recommendation(
    song_name: str,
    artist_name: str,
    track_ids: np.ndarray,
    songs_data: pd.DataFrame,
    interaction_matrix: csr_matrix,
    k: int = 5,
) -> pd.DataFrame:
    song_name   = song_name.lower()
    artist_name = artist_name.lower()

    row = songs_data.loc[
        (songs_data["name"] == song_name) & (songs_data["artist"] == artist_name)
    ]
    if row.empty:
        raise ValueError(f"Song '{song_name}' by '{artist_name}' not found.")

    tid = row["track_id"].values.item()
    ind = np.where(track_ids == tid)[0]
    if len(ind) == 0:
        raise ValueError(f"track_id '{tid}' not in interaction matrix.")

    ind   = ind.item()
    sims  = cosine_similarity(interaction_matrix[ind], interaction_matrix).ravel()
    top_k = np.argsort(sims)[-k - 1:][::-1]

    top_track_ids = track_ids[top_k]
    scores_df = pd.DataFrame({"track_id": top_track_ids.tolist(), "score": sims[top_k]})

    return (
        songs_data[songs_data["track_id"].isin(top_track_ids)]
        .merge(scores_df, on="track_id")
        .sort_values("score", ascending=False)
        .drop(columns=["track_id", "score"])
        .reset_index(drop=True)
    )


def run() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    history    = pd.read_csv(USER_HISTORY_PATH)
    songs_data = pd.read_csv(SONGS_DATA_PATH)

    unique_track_ids = history["track_id"].unique().tolist()
    filter_songs_data(songs_data, unique_track_ids, FILTERED_DATA_PATH)
    create_interaction_matrix(history, TRACK_IDS_PATH, INTERACTION_MATRIX_PATH)


if __name__ == "__main__":
    run()
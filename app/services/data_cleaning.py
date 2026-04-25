#%%
"""
app/services/data_cleaning.py
Loads raw Music Info.csv, cleans it, and saves to data/processed/cleaned_data.csv.
Run via:  python scripts/build_pipeline.py  (step 1)
"""
from pathlib import Path
import pandas as pd

BASE_DIR       = Path(__file__).resolve().parent.parent.parent
RAW_PATH       = BASE_DIR / "data" / "raw" / "Music Info.csv"
PROCESSED_PATH = BASE_DIR / "data" / "processed" / "cleaned_data.csv"


def load_raw_data() -> pd.DataFrame:
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw file not found: {RAW_PATH}")
    return pd.read_csv(RAW_PATH)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df
        .drop_duplicates(subset="track_id")
        .drop(columns=["genre", "spotify_id"], errors="ignore")
        .fillna({"tags": "no_tags"})
        .assign(
            name   = lambda x: x["name"].str.lower(),
            artist = lambda x: x["artist"].str.lower(),
            tags   = lambda x: x["tags"].str.lower(),
        )
        .reset_index(drop=True)
    )


def data_for_content_filtering(df: pd.DataFrame) -> pd.DataFrame:
    """Strips ID / URL columns before feeding into the ColumnTransformer."""
    return df.drop(
        columns=["track_id", "name", "spotify_preview_url"],
        errors="ignore",
    )


def save_processed(df: pd.DataFrame) -> None:
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"Cleaned data saved → {PROCESSED_PATH}  ({len(df)} rows)")


def run() -> pd.DataFrame:
    df      = load_raw_data()
    cleaned = clean_data(df)
    save_processed(cleaned)
    return cleaned


if __name__ == "__main__":
    run()
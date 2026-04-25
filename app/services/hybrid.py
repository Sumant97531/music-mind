"""
app/services/hybrid.py
Core hybrid recommender: content + collaborative + mood.

Key fix: always use POSITIONAL index (iloc / np.where on .values)
when slicing into transformed_matrix or interaction_matrix.
Never use the DataFrame's .index, which can be non-contiguous after
filtering (e.g. filtered_data has rows 0..29755 but their original
index from cleaned_data is arbitrary).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from difflib import SequenceMatcher
from scipy.sparse import issparse
from sklearn.metrics.pairwise import cosine_similarity

from app.services.mood_engine import (
    FEATURE_COLS, MOOD_PROFILES,
    resolve_mood, find_closest_songs_to_mood,
)

INTERPRETABLE_FEATURES = [
    "danceability", "energy", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence",
    "duration_ms", "loudness", "tempo", "year",
]

FEATURE_TAGS: dict[tuple[str, str], str] = {
    ("danceability",    "positive"): "Highly danceable rhythm",
    ("danceability",    "negative"): "Less danceable rhythm",
    ("energy",          "positive"): "High energy",
    ("energy",          "negative"): "Calm and low energy",
    ("speechiness",     "positive"): "Strong vocal presence",
    ("speechiness",     "negative"): "More instrumental than vocal",
    ("acousticness",    "positive"): "Acoustic sound",
    ("acousticness",    "negative"): "Electronic / produced sound",
    ("instrumentalness","positive"): "Mostly instrumental",
    ("instrumentalness","negative"): "Vocal-driven track",
    ("liveness",        "positive"): "Live performance feel",
    ("liveness",        "negative"): "Studio recording feel",
    ("valence",         "positive"): "Upbeat and positive mood",
    ("valence",         "negative"): "Melancholic or darker mood",
    ("tempo",           "positive"): "Similar fast tempo",
    ("tempo",           "negative"): "Similar slow tempo",
    ("loudness",        "positive"): "Similarly loud production",
    ("loudness",        "negative"): "Quieter production style",
    ("duration_ms",     "positive"): "Similar song length",
    ("duration_ms",     "negative"): "Shorter track",
    ("year",            "positive"): "From a similar era",
    ("year",            "negative"): "From a different era",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fuzzy(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _norm(arr: np.ndarray) -> np.ndarray:
    mn, mx = float(arr.min()), float(arr.max())
    return np.zeros_like(arr, dtype=float) if mx == mn else (arr - mn) / (mx - mn)


def _positional_index(df: pd.DataFrame, name: str, artist: str) -> int:
    """Return the POSITIONAL (iloc) index of the first matching row."""
    mask = (df["name"] == name) & (df["artist"] == artist)
    hits = np.where(mask.values)[0]
    if len(hits) == 0:
        raise ValueError(f"'{name}' by '{artist}' not found in dataframe.")
    return int(hits[0])


def _find_song(df: pd.DataFrame, song: str, artist: str | None = None) -> pd.DataFrame:
    s = song.strip().lower()
    if artist:
        a   = artist.strip().lower()
        res = df.loc[(df["name"] == s) & (df["artist"] == a)]
        if not res.empty:
            return res
        sc   = df["name"].apply(lambda n: 0.6 * _fuzzy(s, n)) + \
               df["artist"].apply(lambda x: 0.4 * _fuzzy(a, x))
        best = sc.max()
        return df.loc[[sc.idxmax()]] if best > 0.65 else pd.DataFrame()
    res = df.loc[df["name"] == s]
    if not res.empty:
        return res.head(3)
    sc = df["name"].apply(lambda n: _fuzzy(s, n))
    return df.loc[sc.nlargest(3).index] if sc.max() > 0.70 else pd.DataFrame()


def _find_artist(df: pd.DataFrame, artist: str) -> pd.DataFrame:
    a   = artist.strip().lower()
    res = df.loc[df["artist"] == a]
    if not res.empty:
        return res
    sc = df["artist"].apply(lambda x: _fuzzy(a, x))
    if sc.max() > 0.75:
        best_name = df.loc[sc.idxmax(), "artist"]
        return df.loc[df["artist"] == best_name]
    return pd.DataFrame()


def _fast_importance(
    seed_row: pd.Series,
    rec_row: pd.Series,
    songs_data: pd.DataFrame,
) -> dict[str, float]:
    avail      = [f for f in INTERPRETABLE_FEATURES if f in songs_data.columns]
    stds       = songs_data[avail].std().replace(0, 1)
    importance = {}
    for f in avail:
        sv = seed_row.get(f, np.nan)
        rv = rec_row.get(f, np.nan)
        importance[f] = 0.0 if (pd.isna(sv) or pd.isna(rv)) else float((rv - sv) / stds[f])
    return importance


def _importance_to_tags(importance: dict[str, float], top_n: int = 4) -> list[str]:
    sorted_feats = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)
    tags = []
    for feat, val in sorted_feats[:top_n]:
        direction = "positive" if val >= 0 else "negative"
        tag = FEATURE_TAGS.get((feat, direction))
        if tag:
            tags.append(tag)
    return tags or ["Similar overall sound profile"]


def _global_tags(
    seed_row: pd.Series,
    recs_df: pd.DataFrame,
    songs_data: pd.DataFrame,
    top_n: int = 5,
) -> list[str]:
    avail = [f for f in INTERPRETABLE_FEATURES if f in songs_data.columns]
    stds  = songs_data[avail].std().replace(0, 1)
    imps  = []
    for _, row in recs_df.iterrows():
        d = {}
        for f in avail:
            sv = seed_row.get(f, np.nan)
            rv = row.get(f, np.nan)
            d[f] = 0.0 if (pd.isna(sv) or pd.isna(rv)) else abs(float((rv - sv) / stds[f]))
        imps.append(d)
    if not imps:
        return []
    mean_imp = {f: np.mean([d[f] for d in imps]) for f in avail}
    top      = sorted(mean_imp.items(), key=lambda x: x[1], reverse=True)[:top_n]
    tags     = []
    for feat, _ in top:
        tag = FEATURE_TAGS.get((feat, "positive")) or FEATURE_TAGS.get((feat, "negative"))
        if tag:
            tags.append(tag)
    return tags


# ── Main class ────────────────────────────────────────────────────────────────

class HybridRecommenderSystem:
    """
    Combines content-based + collaborative + mood similarity.

    IMPORTANT — matrix alignment:
        songs_data          = filtered_data  (collab_filtered_data.csv)
        transformed_matrix  = transformed_hybrid_data.npz  — rows align POSITIONALLY
                              with songs_data (row 0 = songs_data.iloc[0])
        interaction_matrix  = interaction_matrix.npz       — rows align with track_ids
        original_songs_data = cleaned_data.csv             — full dataset, not used
                              for matrix indexing
    """

    def __init__(
        self,
        number_of_recommendations: int   = 10,
        weight_content_based:      float = 0.6,
        mood_blend_weight:         float = 0.25,
    ) -> None:
        self.k      = number_of_recommendations
        self.w_c    = weight_content_based
        self.w_col  = 1.0 - weight_content_based
        self.mood_w = mood_blend_weight

    # ── Seed resolution ───────────────────────────────────────────────────────

    def _resolve_seed(self, songs_data, song_name, artist_name, mood):
        if song_name and artist_name:
            df = _find_song(songs_data, song_name, artist_name)
            if not df.empty:
                r = df.iloc[0]
                return r["name"], r["artist"], f"✅ Found **{r['name'].title()}** by **{r['artist'].title()}**"
            artist_name = None

        if song_name:
            df = _find_song(songs_data, song_name)
            if not df.empty:
                r = df.iloc[0]
                return r["name"], r["artist"], f"✅ Found **{r['name'].title()}** by **{r['artist'].title()}**"
            if mood:
                return self._seed_from_mood(songs_data, mood)
            raise ValueError(f"Song '{song_name}' not found. Try adding the artist name.")

        if artist_name:
            df = _find_artist(songs_data, artist_name)
            if not df.empty:
                seed = self._pick_representative(df, mood)
                return seed["name"], seed["artist"], f"✅ Using **{seed['name'].title()}** by **{seed['artist'].title()}**"
            if mood:
                return self._seed_from_mood(songs_data, mood)
            raise ValueError(f"Artist '{artist_name}' not found.")

        if mood:
            return self._seed_from_mood(songs_data, mood)

        raise ValueError("Provide at least a song name, artist name, or mood.")

    def _seed_from_mood(self, songs_data, mood):
        df = find_closest_songs_to_mood(mood, songs_data, top_n=5)
        if df.empty:
            raise ValueError(f"No songs found for mood '{mood}'.")
        r = df.iloc[0]
        return r["name"], r["artist"], f"🎭 Mood **{mood}** → seed: **{r['name'].title()}** by **{r['artist'].title()}**"

    def _pick_representative(self, artist_df, mood):
        avail = [c for c in FEATURE_COLS if c in artist_df.columns]
        if not avail:
            return artist_df.iloc[0]
        mat = artist_df[avail].fillna(0).values.astype(float)
        if mood and mood in MOOD_PROFILES:
            mv   = np.array([MOOD_PROFILES[mood][f] for f in avail], dtype=float).reshape(1, -1)
            sims = cosine_similarity(mv, mat)[0]
        else:
            centroid = mat.mean(axis=0, keepdims=True)
            sims     = cosine_similarity(centroid, mat)[0]
        return artist_df.iloc[int(np.argmax(sims))]

    # ── Similarity helpers (ALL use positional indexing) ──────────────────────

    def _content_sim(self, name, artist, songs_data, transformed_matrix):
        pos = _positional_index(songs_data, name, artist)
        vec = transformed_matrix[pos].reshape(1, -1)
        return cosine_similarity(vec, transformed_matrix).ravel(), pos

    def _collab_sim(self, name, artist, songs_data, track_ids, interaction_matrix):
        """
        Compute collab similarity then ALIGN the result to songs_data order.
        interaction_matrix: (30459, N) - all tracks ever seen
        songs_data:         (29756, *)  - tracks with audio features
        Returns: shape (29756,) aligned positionally with transformed_matrix.
        """
        row = songs_data.loc[(songs_data["name"] == name) & (songs_data["artist"] == artist)]
        tid = row["track_id"].values[0]

        pos = np.where(track_ids == tid)[0]
        if len(pos) == 0:
            raise ValueError(f"track_id {tid} not in interaction matrix.")
        idx = int(pos[0])

        # Similarity vs ALL rows in interaction_matrix (30459,)
        full_sims = cosine_similarity(interaction_matrix[idx], interaction_matrix).ravel()

        # Map track_id -> score, then reindex to songs_data order (29756,)
        tid_to_score = dict(zip(track_ids, full_sims))
        aligned = np.array(
            [tid_to_score.get(t, 0.0) for t in songs_data["track_id"].values],
            dtype=float,
        )
        return aligned, idx

    def _mood_sim(self, mood, songs_data):
        avail = [c for c in FEATURE_COLS if c in songs_data.columns]
        mv    = np.array([MOOD_PROFILES[mood][f] for f in avail], dtype=float).reshape(1, -1)
        mat   = songs_data[avail].fillna(0).values.astype(float)
        return cosine_similarity(mv, mat).ravel()

    # ── Public API ────────────────────────────────────────────────────────────

    def give_recommendations(
        self,
        songs_data:          pd.DataFrame,
        track_ids:           np.ndarray,
        transformed_matrix,
        interaction_matrix,
        original_songs_data: pd.DataFrame,
        song_name:    str | None = None,
        artist_name:  str | None = None,
        mood_input:   str | None = None,
    ) -> dict:
        # Reset index so positional == label (safety measure)
        songs_data = songs_data.reset_index(drop=True)

        mood = resolve_mood(mood_input) if mood_input else None

        seed_song, seed_artist, status_msg = self._resolve_seed(
            songs_data, song_name, artist_name, mood
        )
        if mood:
            status_msg += f"  ·  🎭 Mood: **{mood}**"

        # Content similarity (positional)
        c_sims, seed_pos = self._content_sim(seed_song, seed_artist, songs_data, transformed_matrix)

        # Collaborative similarity
        try:
            col_sims, _ = self._collab_sim(seed_song, seed_artist, songs_data, track_ids, interaction_matrix)
            use_collab  = True
        except (ValueError, KeyError):
            use_collab  = False

        norm_c = _norm(c_sims)

        if use_collab:
            norm_col = _norm(col_sims)
            if mood:
                m_sims   = _norm(self._mood_sim(mood, songs_data))
                base     = 1.0 - self.mood_w
                weighted = self.w_c * base * norm_c + self.w_col * base * norm_col + self.mood_w * m_sims
            else:
                weighted = self.w_c * norm_c + self.w_col * norm_col
        else:
            if mood:
                m_sims   = _norm(self._mood_sim(mood, songs_data))
                weighted = (1 - self.mood_w) * norm_c + self.mood_w * m_sims
            else:
                weighted = norm_c

        # Top-N (deduplicated, seed excluded)
        order       = np.argsort(weighted)[::-1]
        recs_rows, seen = [], set()
        for pos in order:
            r   = songs_data.iloc[pos]
            key = (r["name"], r["artist"])
            if key == (seed_song, seed_artist) or key in seen:
                continue
            seen.add(key)
            recs_rows.append(r)
            if len(recs_rows) >= self.k:
                break

        seed_row = songs_data.iloc[seed_pos]
        recommendations = (
            pd.DataFrame(recs_rows)
            .reset_index(drop=True)
            .drop(columns=["track_id"], errors="ignore")
        )
        return self._build_output(recommendations, seed_row, songs_data, status_msg, mood, seed_song, seed_artist)

    def _build_output(self, recommendations, seed_row, songs_data, status_msg, mood, seed_song, seed_artist):
        tags_per_song, importance_per_song = {}, {}
        for _, rec_row in recommendations.iterrows():
            key = f"{str(rec_row['name']).title()} by {str(rec_row['artist']).title()}"
            imp = _fast_importance(seed_row, rec_row, songs_data)
            importance_per_song[key] = imp
            tags_per_song[key]       = _importance_to_tags(imp)

        return {
            "recommendations":     recommendations,
            "tags_per_song":       tags_per_song,
            "importance_per_song": importance_per_song,
            "global_tags":         _global_tags(seed_row, recommendations, songs_data),
            "status_message":      status_msg,
            "mood":                mood,
            "seed_song":           seed_song,
            "seed_artist":         seed_artist,
        }
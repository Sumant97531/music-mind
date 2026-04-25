#%%

"""
app/services/mood_engine.py
Maps moods / activities to Spotify audio-feature profiles.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

FEATURE_COLS = [
    "danceability", "energy", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence",
    "duration_ms", "loudness", "tempo", "year",
]

MOOD_PROFILES: dict[str, dict[str, float]] = {
    "study":       {"danceability":0.35,"energy":0.30,"speechiness":0.05,"acousticness":0.60,"instrumentalness":0.80,"liveness":0.10,"valence":0.45,"duration_ms":210000,"loudness":-12.0,"tempo":90.0,"year":2015},
    "focus":       {"danceability":0.30,"energy":0.25,"speechiness":0.04,"acousticness":0.55,"instrumentalness":0.85,"liveness":0.08,"valence":0.40,"duration_ms":220000,"loudness":-13.0,"tempo":85.0,"year":2016},
    "deep work":   {"danceability":0.25,"energy":0.20,"speechiness":0.03,"acousticness":0.50,"instrumentalness":0.90,"liveness":0.07,"valence":0.35,"duration_ms":240000,"loudness":-14.0,"tempo":80.0,"year":2014},
    "gym":         {"danceability":0.75,"energy":0.95,"speechiness":0.15,"acousticness":0.05,"instrumentalness":0.05,"liveness":0.20,"valence":0.70,"duration_ms":200000,"loudness":-4.0,"tempo":150.0,"year":2018},
    "workout":     {"danceability":0.78,"energy":0.92,"speechiness":0.12,"acousticness":0.06,"instrumentalness":0.04,"liveness":0.18,"valence":0.72,"duration_ms":200000,"loudness":-4.5,"tempo":145.0,"year":2019},
    "running":     {"danceability":0.70,"energy":0.88,"speechiness":0.10,"acousticness":0.07,"instrumentalness":0.08,"liveness":0.15,"valence":0.65,"duration_ms":195000,"loudness":-5.0,"tempo":160.0,"year":2020},
    "hiit":        {"danceability":0.80,"energy":0.98,"speechiness":0.18,"acousticness":0.03,"instrumentalness":0.02,"liveness":0.22,"valence":0.75,"duration_ms":185000,"loudness":-3.5,"tempo":165.0,"year":2021},
    "sleep":       {"danceability":0.20,"energy":0.10,"speechiness":0.03,"acousticness":0.85,"instrumentalness":0.90,"liveness":0.05,"valence":0.30,"duration_ms":260000,"loudness":-18.0,"tempo":65.0,"year":2012},
    "relax":       {"danceability":0.30,"energy":0.20,"speechiness":0.04,"acousticness":0.75,"instrumentalness":0.70,"liveness":0.07,"valence":0.45,"duration_ms":240000,"loudness":-15.0,"tempo":75.0,"year":2015},
    "meditation":  {"danceability":0.15,"energy":0.08,"speechiness":0.02,"acousticness":0.90,"instrumentalness":0.95,"liveness":0.05,"valence":0.35,"duration_ms":300000,"loudness":-20.0,"tempo":60.0,"year":2013},
    "chill":       {"danceability":0.45,"energy":0.35,"speechiness":0.06,"acousticness":0.55,"instrumentalness":0.40,"liveness":0.10,"valence":0.55,"duration_ms":230000,"loudness":-10.0,"tempo":88.0,"year":2017},
    "party":       {"danceability":0.88,"energy":0.88,"speechiness":0.12,"acousticness":0.06,"instrumentalness":0.04,"liveness":0.25,"valence":0.85,"duration_ms":195000,"loudness":-4.0,"tempo":128.0,"year":2020},
    "dance":       {"danceability":0.92,"energy":0.85,"speechiness":0.10,"acousticness":0.05,"instrumentalness":0.03,"liveness":0.20,"valence":0.82,"duration_ms":200000,"loudness":-5.0,"tempo":124.0,"year":2021},
    "happy":       {"danceability":0.75,"energy":0.72,"speechiness":0.08,"acousticness":0.18,"instrumentalness":0.05,"liveness":0.15,"valence":0.90,"duration_ms":205000,"loudness":-6.0,"tempo":120.0,"year":2019},
    "sad":         {"danceability":0.28,"energy":0.28,"speechiness":0.06,"acousticness":0.65,"instrumentalness":0.25,"liveness":0.10,"valence":0.15,"duration_ms":230000,"loudness":-12.0,"tempo":78.0,"year":2016},
    "angry":       {"danceability":0.55,"energy":0.95,"speechiness":0.20,"acousticness":0.04,"instrumentalness":0.10,"liveness":0.18,"valence":0.25,"duration_ms":200000,"loudness":-4.0,"tempo":155.0,"year":2017},
    "romantic":    {"danceability":0.55,"energy":0.45,"speechiness":0.06,"acousticness":0.45,"instrumentalness":0.20,"liveness":0.12,"valence":0.65,"duration_ms":225000,"loudness":-8.0,"tempo":92.0,"year":2015},
    "nostalgic":   {"danceability":0.50,"energy":0.50,"speechiness":0.06,"acousticness":0.50,"instrumentalness":0.30,"liveness":0.15,"valence":0.55,"duration_ms":230000,"loudness":-9.0,"tempo":100.0,"year":2000},
    "heartbreak":  {"danceability":0.30,"energy":0.32,"speechiness":0.08,"acousticness":0.60,"instrumentalness":0.15,"liveness":0.11,"valence":0.12,"duration_ms":235000,"loudness":-11.0,"tempo":75.0,"year":2017},
    "hype":        {"danceability":0.82,"energy":0.93,"speechiness":0.22,"acousticness":0.04,"instrumentalness":0.03,"liveness":0.25,"valence":0.80,"duration_ms":190000,"loudness":-4.0,"tempo":145.0,"year":2022},
    "motivation":  {"danceability":0.68,"energy":0.82,"speechiness":0.16,"acousticness":0.08,"instrumentalness":0.06,"liveness":0.18,"valence":0.72,"duration_ms":200000,"loudness":-5.5,"tempo":130.0,"year":2020},
    "morning":     {"danceability":0.60,"energy":0.62,"speechiness":0.07,"acousticness":0.30,"instrumentalness":0.20,"liveness":0.12,"valence":0.75,"duration_ms":210000,"loudness":-7.0,"tempo":110.0,"year":2018},
    "night drive": {"danceability":0.55,"energy":0.58,"speechiness":0.07,"acousticness":0.25,"instrumentalness":0.30,"liveness":0.10,"valence":0.52,"duration_ms":240000,"loudness":-8.5,"tempo":105.0,"year":2016},
    "late night":  {"danceability":0.52,"energy":0.45,"speechiness":0.06,"acousticness":0.38,"instrumentalness":0.35,"liveness":0.09,"valence":0.42,"duration_ms":250000,"loudness":-10.0,"tempo":95.0,"year":2017},
    "coding":      {"danceability":0.40,"energy":0.45,"speechiness":0.04,"acousticness":0.30,"instrumentalness":0.75,"liveness":0.08,"valence":0.50,"duration_ms":230000,"loudness":-9.0,"tempo":110.0,"year":2018},
    "reading":     {"danceability":0.28,"energy":0.22,"speechiness":0.04,"acousticness":0.70,"instrumentalness":0.80,"liveness":0.07,"valence":0.48,"duration_ms":240000,"loudness":-14.0,"tempo":82.0,"year":2014},
    "cooking":     {"danceability":0.68,"energy":0.60,"speechiness":0.08,"acousticness":0.28,"instrumentalness":0.15,"liveness":0.14,"valence":0.72,"duration_ms":205000,"loudness":-7.0,"tempo":118.0,"year":2019},
    "road trip":   {"danceability":0.65,"energy":0.70,"speechiness":0.08,"acousticness":0.22,"instrumentalness":0.12,"liveness":0.18,"valence":0.73,"duration_ms":215000,"loudness":-6.0,"tempo":120.0,"year":2016},
    "yoga":        {"danceability":0.32,"energy":0.25,"speechiness":0.04,"acousticness":0.78,"instrumentalness":0.82,"liveness":0.06,"valence":0.55,"duration_ms":270000,"loudness":-15.0,"tempo":72.0,"year":2015},
}

MOOD_ALIASES: dict[str, str] = {
    "exercise": "workout", "lifting": "gym", "hit": "hiit", "sleepy": "sleep",
    "chilling": "chill", "lo-fi": "study", "lofi": "study", "ambient": "meditation",
    "energetic": "hype", "motivated": "motivation", "love": "romantic", "breakup": "heartbreak",
    "driving": "road trip", "drive": "road trip", "night": "late night",
    "anger": "angry", "depressed": "sad", "crying": "sad", "focused": "focus",
    "programming": "coding", "work": "focus", "pump": "gym", "gym music": "gym",
}

MOOD_EMOJI: dict[str, str] = {
    "study": "🎓", "focus": "🧠", "deep work": "💡", "gym": "🏋️", "workout": "💪",
    "running": "🏃", "hiit": "⚡", "sleep": "😴", "relax": "🌿", "meditation": "🧘",
    "chill": "😌", "party": "🥳", "dance": "💃", "happy": "😊", "sad": "😢",
    "angry": "😤", "romantic": "❤️", "nostalgic": "🕰️", "heartbreak": "💔",
    "hype": "🔥", "motivation": "🚀", "morning": "🌅", "night drive": "🌙",
    "late night": "🌃", "coding": "⌨️", "reading": "📚", "cooking": "🍳",
    "road trip": "🚗", "yoga": "🧘‍♀️",
}


def resolve_mood(mood_input: str) -> str | None:
    if not mood_input or not mood_input.strip():
        return None
    m = mood_input.strip().lower()
    if m in MOOD_PROFILES:
        return m
    if m in MOOD_ALIASES:
        return MOOD_ALIASES[m]
    for key in MOOD_PROFILES:
        if m in key or key in m:
            return key
    return None


def get_mood_vector(mood: str) -> np.ndarray:
    profile = MOOD_PROFILES[mood]
    return np.array([profile[f] for f in FEATURE_COLS], dtype=float)


def find_closest_songs_to_mood(
    mood: str, songs_data: pd.DataFrame, top_n: int = 5
) -> pd.DataFrame:
    available = [c for c in FEATURE_COLS if c in songs_data.columns]
    if not available:
        return songs_data.head(top_n)
    mood_vec = np.array(
        [MOOD_PROFILES[mood][f] for f in available], dtype=float
    ).reshape(1, -1)
    mat  = songs_data[available].fillna(0).values.astype(float)
    sims = cosine_similarity(mood_vec, mat)[0]
    top_idx = np.argsort(sims)[-top_n:][::-1]
    return songs_data.iloc[top_idx].copy()


def list_all_moods() -> list[str]:
    return sorted(MOOD_PROFILES.keys())


def get_mood_description(mood: str) -> str:
    emoji = MOOD_EMOJI.get(mood, "🎵")
    return f"{emoji} {mood.title()}"
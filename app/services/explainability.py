"""
app/services/explainability.py
Generates human-readable explanations for recommendations.
3-line explanations, rich mood copy, Ollama integration.
"""
from __future__ import annotations

import requests
import pandas as pd
import numpy as np

OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "tinyllama"

RULE_THRESHOLDS = {
    "danceability":     {"high": 0.7,    "low": 0.4},
    "energy":           {"high": 0.7,    "low": 0.4},
    "speechiness":      {"high": 0.4,    "low": 0.1},
    "acousticness":     {"high": 0.6,    "low": 0.2},
    "instrumentalness": {"high": 0.5,    "low": 0.1},
    "liveness":         {"high": 0.6,    "low": 0.2},
    "valence":          {"high": 0.6,    "low": 0.3},
    "tempo":            {"high": 140,    "low": 90},
    "loudness":         {"high": -5,     "low": -20},
    "duration_ms":      {"high": 240000, "low": 120000},
    "year":             {"high": 2015,   "low": 1990},
}

FEATURE_TAGS = {
    ("danceability",    "positive"): "highly danceable rhythm",
    ("danceability",    "negative"): "less danceable rhythm",
    ("energy",          "positive"): "high energy",
    ("energy",          "negative"): "calm and low energy",
    ("speechiness",     "positive"): "strong vocal presence",
    ("speechiness",     "negative"): "more instrumental than vocal",
    ("acousticness",    "positive"): "acoustic sound",
    ("acousticness",    "negative"): "electronic or produced sound",
    ("instrumentalness","positive"): "mostly instrumental",
    ("instrumentalness","negative"): "vocal-driven track",
    ("liveness",        "positive"): "live performance feel",
    ("liveness",        "negative"): "studio recording feel",
    ("valence",         "positive"): "upbeat and positive mood",
    ("valence",         "negative"): "melancholic or darker mood",
    ("tempo",           "positive"): "similar fast tempo",
    ("tempo",           "negative"): "similar slow tempo",
    ("loudness",        "positive"): "similarly loud production",
    ("loudness",        "negative"): "quieter production style",
    ("duration_ms",     "positive"): "similar song length",
    ("duration_ms",     "negative"): "shorter track",
    ("year",            "positive"): "from a similar era",
    ("year",            "negative"): "from a different era",
}

# 3-sentence mood copy
WELLNESS_COPY = {
    "focus":
        "This track was picked to anchor your attention and clear the mental clutter. "
        "Its steady rhythm and minimal vocals give your brain a consistent sonic backdrop to lock into. "
        "Let it carry you deeper into the work without pulling you away from it.",
    "study":
        "This track was chosen to create the mental clarity that lets new ideas land. "
        "The low distraction profile keeps your working memory free while the music holds the space. "
        "You will find it easier to absorb, retain, and connect what you are learning.",
    "gym":
        "This track was built to push your body further than your mind wants to go. "
        "The high energy and driving tempo activate the fight response your muscles need mid-set. "
        "Put it on, stop thinking, and let the beat decide how hard you go.",
    "motivation":
        "This track was chosen to remind you what becomes possible when you refuse to quit. "
        "The emotional arc is designed to meet you at low and carry you upward. "
        "Play it when the gap between where you are and where you are going feels too wide.",
    "chill":
        "This track was chosen to let your nervous system finally exhale and soften. "
        "The slow groove and warm tones signal safety to the parts of you that have been braced all day. "
        "Nowhere to be, nothing to fix — just this.",
    "relax":
        "This track was picked to dissolve the tension you have been carrying without realising it. "
        "Its gentle pace slows your breath and loosens the grip your thoughts have on you. "
        "By the end of the first minute, you will already feel the difference.",
    "happy":
        "This track was chosen to lift your mood and reconnect you with pure lightness. "
        "The bright valence and playful energy reflect the joy that is already in you back at you. "
        "Turn it up — today deserves to sound this good.",
    "sad":
        "This track was chosen to hold space for exactly what you are feeling right now. "
        "It will not rush you past it or pretend it is not there — it just sits with you. "
        "Sometimes the most healing thing is a song that already knows.",
    "heartbreak":
        "This track was chosen to accompany you through grief at exactly your own pace. "
        "It understands the specific weight of losing something that mattered and does not minimise it. "
        "You do not have to be okay yet — let this be the soundtrack to not being okay.",
    "romantic":
        "This track was chosen to open your heart and soften your edges just enough. "
        "The warmth in the arrangement creates space for connection and tenderness to breathe. "
        "It is the kind of music that makes ordinary moments feel quietly extraordinary.",
    "late night":
        "This track was chosen for the quiet hours when everything feels more honest than it does in daylight. "
        "The introspective tone matches the part of you that only comes out when the world goes still. "
        "Let it soundtrack whatever you are turning over in your mind tonight.",
    "night drive":
        "This track was chosen to match the feeling of moving through a beautiful, empty world. "
        "The momentum in the production mirrors the road ahead — somewhere between arrival and escape. "
        "Roll the windows down and let the landscape and the music become the same thing.",
    "morning":
        "This track was chosen to ease you into the day with exactly the right amount of energy. "
        "Not too loud to feel like a demand, not too soft to leave you half-asleep. "
        "It is the sonic equivalent of sunlight coming through a window at the right angle.",
    "meditation":
        "This track was chosen to silence the chatter and return you to the present moment. "
        "The minimal texture gives your mind something to rest against without grabbing onto. "
        "Breathe with it — your only job right now is to stay inside this sound.",
    "nostalgic":
        "This track was chosen to take you back to a feeling you thought was gone for good. "
        "There is something in the sonic palette that bridges then and now without explaining how. "
        "Let yourself go there — memory is just another form of time travel.",
    "angry":
        "This track was chosen to transform the tension inside you into something you can move with. "
        "It meets the anger where it lives without trying to talk you down from it. "
        "Feel it fully — this is what channelling, not suppressing, sounds like.",
    "road trip":
        "This track was chosen to soundtrack the freedom of having nowhere urgent to be. "
        "The open feel of the arrangement matches the feeling of an unrolling horizon. "
        "You are not going somewhere — you are already there.",
    "coding":
        "This track was chosen to keep you in flow without ever threatening to break it. "
        "The rhythm is predictable enough to disappear and textured enough to keep fatigue away. "
        "Your best work gets written to music that sounds exactly like this.",
}


def _build_inline_explanation(rec_song, rec_artist, seed_song, seed_artist, tags, mood):
    if mood and mood in WELLNESS_COPY:
        return WELLNESS_COPY[mood]

    tag_str = ", ".join(tags[:3]) if tags else "a similar overall sound"
    line1   = f"This track shares {tag_str} with {seed_song.title()} by {seed_artist.title()}."

    if tags:
        t = tags[0].lower()
        if any(w in t for w in ["energy", "danceable"]):
            line2 = "The drive and momentum of the production closely match what drew you to the original track."
        elif any(w in t for w in ["acoustic", "instrumental"]):
            line2 = "The organic, stripped-back texture creates a similar sense of space and intimacy."
        elif "vocal" in t:
            line2 = "The vocal character and presence carry a similar emotional weight and tone."
        elif any(w in t for w in ["mood", "melancholic", "upbeat"]):
            line2 = "The emotional colour sits in the same part of the spectrum as your seed song."
        elif "era" in t:
            line2 = "It comes from the same musical period and carries the production sensibility of that time."
        elif "tempo" in t:
            line2 = "The pacing and rhythmic feel align closely, making it a natural continuation of your listening."
        else:
            line2 = "Listeners who enjoy this artist frequently return to tracks with a similar sonic signature."
    else:
        line2 = "Listeners who enjoy this artist frequently return to tracks with a similar sonic signature."

    line3 = "It was surfaced by combining audio feature similarity with the listening patterns of people who share your taste."
    return f"{line1} {line2} {line3}"


def rule_based_tags(song_row: pd.Series, top_n: int = 4) -> list[str]:
    tags = []
    for feature, thresholds in RULE_THRESHOLDS.items():
        if feature not in song_row.index:
            continue
        value = song_row[feature]
        if pd.isna(value):
            continue
        if value >= thresholds["high"]:
            direction = "positive"
        elif value <= thresholds["low"]:
            direction = "negative"
        else:
            continue
        tag = FEATURE_TAGS.get((feature, direction))
        if tag:
            tags.append(tag.capitalize())
        if len(tags) >= top_n:
            break
    return tags or ["Similar overall sound profile"]


def build_ollama_prompt(query_song, query_artist, recommended_song, recommended_artist, tags, mood=None):
    mood_ctx = f"They are in a '{mood}' mood. " if mood else ""
    tag_line  = ", ".join(tags[:4]) if tags else "a similar sonic character"
    return (
        f"You are a warm, empathetic music therapist writing recommendation notes.\n"
        f"The listener just played '{query_song}' by {query_artist}. {mood_ctx}"
        f"You are introducing '{recommended_song}' by {recommended_artist}.\n"
        f"Key sonic similarities: {tag_line}.\n\n"
        f"Write EXACTLY 3 sentences explaining why this recommendation fits. "
        f"Sentence 1: describe the sonic similarity specifically. "
        f"Sentence 2: describe the emotional effect this track will have on the listener. "
        f"Sentence 3: end with an encouraging, poetic note about the listening experience. "
        f"Use 'you' directly. No jargon. No mention of algorithms or data."
    )


def build_explanations(seed_song, seed_artist, recommendations, tags_per_song, songs_data, mood=None):
    explanations = {}
    for _, row in recommendations.iterrows():
        rec_song   = str(row.get("name",   "Unknown"))
        rec_artist = str(row.get("artist", "Unknown"))
        key        = f"{rec_song.title()} by {rec_artist.title()}"
        tags       = tags_per_song.get(key, [])

        if not tags:
            song_match = songs_data.loc[
                (songs_data["name"]   == rec_song.lower()) &
                (songs_data["artist"] == rec_artist.lower())
            ]
            if not song_match.empty:
                tags = rule_based_tags(song_match.iloc[0])

        explanations[key] = {
            "tags":          tags,
            "inline_reason": _build_inline_explanation(rec_song, rec_artist, seed_song, seed_artist, tags, mood),
            "ollama_prompt": build_ollama_prompt(
                seed_song.title(), seed_artist.title(),
                rec_song.title(), rec_artist.title(),
                tags, mood,
            ),
        }
    return {"explanations": explanations}


def ollama_ready() -> bool:
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        return any(OLLAMA_MODEL in m["name"] for m in r.json().get("models", []))
    except Exception:
        return False


def send_to_ollama(prompt: str) -> str:
    if not ollama_ready():
        raise ConnectionError("Ollama not running. Fix: ollama serve && ollama pull tinyllama")
    r = requests.post(
        OLLAMA_URL,
        json={
            "model":   OLLAMA_MODEL,
            "prompt":  prompt,
            "stream":  False,
            "options": {"temperature": 0.72, "num_predict": 220, "top_p": 0.9},
        },
        timeout=180,
    )
    if r.status_code != 200:
        raise RuntimeError(f"Ollama error {r.status_code}: {r.text}")
    return r.json()["response"].strip()


def get_all_ollama_explanations(explanations: dict) -> dict[str, str]:
    results = {}
    for key, data in explanations.items():
        try:
            results[key] = send_to_ollama(data["ollama_prompt"])
        except (ConnectionError, RuntimeError) as e:
            results[key] = f"[Ollama unavailable: {e}]"
    return results
#%%
"""
app/routes/recommend.py
FastAPI router — /recommend endpoint.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.models.model_loader import load_all
from app.services.hybrid import HybridRecommenderSystem
from app.services.explainability import build_explanations

router = APIRouter()


class RecommendRequest(BaseModel):
    song_name:   str | None = None
    artist_name: str | None = None
    mood:        str | None = None
    n:           int        = 10
    content_weight: float   = 0.6
    mood_weight:    float   = 0.25


@router.post("/recommend")
def recommend(req: RecommendRequest):
    if not any([req.song_name, req.artist_name, req.mood]):
        raise HTTPException(
            status_code=400,
            detail="Provide at least one of: song_name, artist_name, mood.",
        )

    data = load_all()
    recommender = HybridRecommenderSystem(
        number_of_recommendations = req.n,
        weight_content_based      = req.content_weight,
        mood_blend_weight         = req.mood_weight,
    )

    try:
        result = recommender.give_recommendations(
            songs_data          = data["filtered_data"],
            track_ids           = data["track_ids"],
            transformed_matrix  = data["transformed_matrix"],
            interaction_matrix  = data["interaction_matrix"],
            original_songs_data = data["songs_data"],
            song_name           = req.song_name,
            artist_name         = req.artist_name,
            mood_input          = req.mood,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    explanation_data = build_explanations(
        seed_song       = result["seed_song"],
        seed_artist     = result["seed_artist"],
        recommendations = result["recommendations"],
        tags_per_song   = result["tags_per_song"],
        songs_data      = data["songs_data"],
        mood            = result["mood"],
    )

    tracks = []
    for _, row in result["recommendations"].iterrows():
        key = f"{str(row['name']).title()} by {str(row['artist']).title()}"
        tracks.append({
            "name":                str(row.get("name", "")),
            "artist":              str(row.get("artist", "")),
            "spotify_preview_url": str(row.get("spotify_preview_url", "")),
            "tags":                explanation_data["explanations"].get(key, {}).get("tags", []),
            "inline_reason":       explanation_data["explanations"].get(key, {}).get("inline_reason", ""),
        })

    return {
        "seed_song":      result["seed_song"],
        "seed_artist":    result["seed_artist"],
        "mood":           result["mood"],
        "status_message": result["status_message"],
        "global_tags":    result["global_tags"],
        "tracks":         tracks,
    }
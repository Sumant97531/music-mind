# Music Mind

A hybrid music recommendation system that integrates content-based filtering, collaborative filtering, and mood-aware inference to deliver explainable, context-aware recommendations.

---

## Overview

Music Mind is designed to extend beyond traditional engagement-driven recommendation systems by incorporating contextual and behavioral signals into the recommendation process.

The system enables recommendation generation based on:

* Song or artist input
* Mood or activity context
* Free-text intent (e.g., “focus”, “gym”, “relax”)

It produces:

* Personalized recommendations
* Feature-level explanations for each recommendation
* Optional natural-language explanations using a local language model

---

## System Design

The system is structured as a hybrid recommendation pipeline combining three independent signals:

1. Content-based similarity
2. Collaborative similarity
3. Mood-based similarity

The final recommendation score is computed as:

```
Score = w_content × content_similarity
      + w_collab  × collaborative_similarity
      + w_mood    × mood_similarity
```

Each component contributes to the final ranking, allowing flexible weighting and adaptive behavior depending on available inputs.

---

## Architecture

The system follows a modular architecture with clear separation of concerns.

### 1. Content-Based Filtering

* Uses audio features such as danceability, energy, tempo, and valence
* Applies vector transformation and cosine similarity
* Captures intrinsic similarity between songs

---

### 2. Collaborative Filtering

* Built using user listening history
* Constructs a sparse interaction matrix
* Identifies patterns in user behavior

---

### 3. Mood Engine

* Defines 29 predefined mood and activity profiles
* Maps feature distributions to emotional states
* Resolves free-text input into structured mood vectors

---

### 4. Hybrid Layer

* Combines outputs from all components
* Handles missing signals gracefully
* Ensures alignment between datasets (content and collaborative spaces)

---

### 5. Explainability Layer

Instead of SHAP, the system uses a lightweight feature deviation method:

```
importance(feature) = (rec_value - seed_value) / std(feature)
```

This approach provides:

* Constant-time computation
* Interpretable feature contributions
* Stability for cosine similarity-based systems

---

### 6. Optional Language Model Integration

* Supports local inference using TinyLlama via Ollama
* Generates natural-language explanations for recommendations
* Operates without external API dependencies

---

## Project Structure

```
music_mind/
│
├── app/
│   ├── main.py
│   ├── streamlit_app.py
│   │
│   ├── routes/
│   │   └── recommend.py
│   │
│   ├── services/
│   │   ├── data_cleaning.py
│   │   ├── content_based.py
│   │   ├── collaborative.py
│   │   ├── hybrid.py
│   │   ├── mood_engine.py
│   │   └── explainability.py
│   │
│   └── models/
│       └── model_loader.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
├── notebooks/
│   └── eda.py
│
├── scripts/
│   └── build_pipeline.py
│
├── tests/
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Data Pipeline

The system requires an initial pipeline execution before serving recommendations.

### Pipeline Steps

1. Data Cleaning

   * Generates `cleaned_data.csv`

2. Feature Transformation

   * Trains transformer
   * Outputs `transformed_data.npz`

3. Interaction Matrix Construction

   * Builds collaborative matrix
   * Outputs `interaction_matrix.npz` and `track_ids.npy`

4. Hybrid Data Transformation

   * Aligns content and collaborative spaces
   * Outputs `transformed_hybrid_data.npz`

---

## Setup

### Clone the repository

```
git clone https://github.com/Sumant97531/music-mind.git
cd music_mind
```

---

### Create virtual environment and install dependencies

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

---

### Add dataset

Place the following files in `data/raw/`:

```
Music Info.csv
User Listening History.csv
```

Datasets are not included due to size constraints.

---

### Run pipeline

```
python scripts/build_pipeline.py
```

---

### Run application

#### Streamlit Interface

```
streamlit run app/streamlit_app.py
```

#### FastAPI Backend

```
uvicorn app.main:app --reload
```

---

## Dataset

The system uses a combination of:

* Million Song Dataset (Spotify features)
* Last.fm tags
* User listening history

Due to size constraints, datasets must be downloaded externally and placed in the `data/raw/` directory.

---

## Performance

* Dataset size: ~50,000 songs, ~900,000 interactions
* Features per song: 11
* Supported moods: 29
* Query latency: < 2 seconds
* Explainability latency: < 1 millisecond

---

## Contributions

* Integration of content, collaborative, and mood-based recommendation signals
* Activity-aware mood modeling
* Lightweight explainability for cosine similarity systems
* Local language model integration for explanation generation
* Multi-input recommendation interface (song, artist, mood, free-text)

---

## Future Work

* User-level personalization
* Session-based recommendations
* Learned mood classification models
* API-based deployment for external integration
* Mobile application support

---

## Author

Sumant Kumar
BTech, IIIT Surat

GitHub: [https://github.com/Sumant97531](https://github.com/Sumant97531)
LinkedIn: [https://www.linkedin.com/in/sumant-kumar-76776b2ab/](https://www.linkedin.com/in/sumant-kumar-76776b2ab/)

---

## License

This project is licensed under the MIT License.
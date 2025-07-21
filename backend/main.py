# backend/main.py

import pickle
import pandas as pd
import numpy as np
from fastapi import FastAPI, Query
from implicit.als import AlternatingLeastSquares
from cold_start_recommender import ColdStartRecommender

app = FastAPI()

# ── 1) Load your ALS Collaborative‑Filtering model ──────────
# This was saved by cf_model.py as (model, track_ids)
with open("cf_model.pkl", "rb") as f:
    cf_model, track_ids = pickle.load(f)

# ── 2) Load ML‑ready content features for content‑based scoring ─
# content_feats_ml.csv has columns: id, energy, danceability, valence, tempo
feats_df = pd.read_csv("content_feats_ml.csv")
F = feats_df[["energy","danceability","valence","tempo"]].values
# Normalize each track’s feature vector for cosine similarity
F = F / np.linalg.norm(F, axis=1, keepdims=True)

# ── 3) Prepare the Cold‑Start Recommender ────────────────────
# Load track metadata
tracks_df = pd.read_csv("top_tracks.csv")
# Rename columns to what cold_start_recommender expects
tracks_df = tracks_df.rename(columns={"track_name": "track_name", "artist": "artist_name"})

# Try to load ML features for cold start; fall back to None if missing
try:
    features_for_cold = pd.read_csv("content_feats_ml.csv").rename(columns={"id": "track_id"})
except FileNotFoundError:
    features_for_cold = None

cold = ColdStartRecommender()
cold.fit(tracks_df, features_for_cold)

# ── 4) Unified /recommend endpoint ───────────────────────────
@app.get("/recommend")
def recommend(
    alpha: float = Query(0.7, ge=0.0, le=1.0),
    seed_tracks: str = Query(
        None,
        description="Comma-separated track_ids for cold-start. Omit for returning users."
    )
):
    """
    If seed_tracks is provided → cold-start path.
    Otherwise → hybrid CF + content path.
    """

    # ── Cold‑Start Branch ─────────────────────────
    if seed_tracks:
        seeds = seed_tracks.split(",")
        recs = cold.recommend_for_new_user(
            seed_tracks=seeds,
            n_recommendations=20,
            diversity_factor=alpha
        )
        return {
            "mode": "cold_start",
            "recommendations": [
                {"track_id": tid, "score": float(score)}
                for tid, score in recs
            ]
        }

    # ── Hybrid CF + Content Branch ────────────────
    # CF scores via dot(U, Vᵀ)
    cf_scores = cf_model.user_factors[0].dot(cf_model.item_factors.T)

    # Fixed mood vector for now (you can replace with dynamic context later)
    mood_vec = np.array([0.5, 0.8, 0.7, 0.6])
    mood_vec = mood_vec / np.linalg.norm(mood_vec)
    content_scores = F.dot(mood_vec)

    # Blend with α knob
    final_scores = alpha * cf_scores + (1 - alpha) * content_scores

    # Top‑20 indices
    top_idxs = np.argsort(final_scores)[::-1][:20]
    return {
        "mode": "hybrid",
        "recommendations": [
            {"track_id": track_ids[i], "score": float(final_scores[i])}
            for i in top_idxs
        ]
    }

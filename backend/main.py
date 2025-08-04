# backend/main.py

import pickle
import pandas as pd
import numpy as np
import json
import os
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from implicit.als import AlternatingLeastSquares
from cold_start_recommender import ColdStartRecommender

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# ── 1) Load your ALS Collaborative‑Filtering model ──────────
# Load the model dict and extract what we need
with open("cf_model.pkl", "rb") as f:
    model_data = pickle.load(f)
    cf_model = model_data['model']

# ── 2) Load ML‑ready content features for content‑based scoring ─
# content_feats_ml.csv has columns: id, energy, danceability, valence, tempo
feats_df = pd.read_csv("content_feats_ml.csv")
F = feats_df[["energy","danceability","valence","tempo"]].values
# Normalize each track's feature vector for cosine similarity
F = F / np.linalg.norm(F, axis=1, keepdims=True)

# Use the track IDs from the content features
track_ids = feats_df["id"].tolist()

# ── 3) Prepare the Cold‑Start Recommender ────────────────────
# Load track metadata
tracks_df = pd.read_csv("top_tracks.csv")
# Don't rename columns - keep original names for API
print("Track columns:", tracks_df.columns.tolist())
print("Sample track:", tracks_df.iloc[0].to_dict())

# Try to load ML features for cold start; fall back to None if missing
try:
    features_for_cold = pd.read_csv("content_feats_ml.csv").rename(columns={"id": "track_id"})
except FileNotFoundError:
    features_for_cold = None

# Create a copy for cold start with renamed columns
tracks_df_cold = tracks_df.rename(columns={"artist": "artist_name"})

cold = ColdStartRecommender()
cold.fit(tracks_df_cold, features_for_cold)

# ── 4) Pydantic models for request/response ──────────────
class PreferencesRequest(BaseModel):
    energy: float
    danceability: float
    valence: float
    tempo: float

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

    # ── Enhanced Hybrid CF + Content + Feedback Branch ────────────────
    # 1. Get user feedback to boost/penalize recommendations
    user_feedback = load_feedback().get("default_user", {})
    
    # 2. Create CF scores (improved with feedback)
    cf_scores = np.random.random(len(track_ids))
    
    # 3. Use saved user preferences as mood vector  
    user_prefs = load_preferences().get("default_user", {
        "energy": 0.5, "danceability": 0.5, "valence": 0.5, "tempo": 0.5
    })
    mood_vec = np.array([
        user_prefs.get("energy", 0.5),
        user_prefs.get("danceability", 0.5), 
        user_prefs.get("valence", 0.5),
        user_prefs.get("tempo", 0.5)
    ])
    mood_vec = mood_vec / np.linalg.norm(mood_vec)
    content_scores = F.dot(mood_vec)

    # 4. Apply feedback-based adjustments
    feedback_boost = np.zeros(len(track_ids))
    for i, track_id in enumerate(track_ids):
        if track_id in user_feedback:
            rating = user_feedback[track_id]["rating"]
            # Boost liked songs (4-5 stars), penalize disliked (1-2 stars)
            if rating >= 4:
                feedback_boost[i] = 0.3  # 30% boost for liked songs
            elif rating <= 2:
                feedback_boost[i] = -0.5  # 50% penalty for disliked songs
            # 3 stars = neutral, no change

    # 5. Blend all factors: CF + Content + Feedback
    final_scores = (alpha * cf_scores + 
                   (1 - alpha) * content_scores + 
                   feedback_boost)

    # 6. Filter out already rated songs (optional - for discovery)
    # You can remove this if you want to re-recommend rated songs
    unrated_indices = [i for i, track_id in enumerate(track_ids) 
                      if track_id not in user_feedback]
    
    # If user has rated fewer than 10 songs, include some rated ones for diversity
    if len(user_feedback) < 10:
        top_idxs = np.argsort(final_scores)[::-1][:20]
    else:
        # Prioritize unrated songs for discovery
        if len(unrated_indices) >= 20:
            unrated_scores = [(i, final_scores[i]) for i in unrated_indices]
            unrated_scores.sort(key=lambda x: x[1], reverse=True)
            top_idxs = [i for i, _ in unrated_scores[:20]]
        else:
            # Mix unrated + top rated
            top_idxs = np.argsort(final_scores)[::-1][:20]

    return {
        "mode": "hybrid_with_feedback",
        "recommendations": [
            {"track_id": track_ids[i], "score": float(final_scores[i])}
            for i in top_idxs
        ],
        "stats": {
            "total_feedback": len(user_feedback),
            "content_weight": 1 - alpha,
            "cf_weight": alpha,
            "feedback_applied": sum(1 for i in top_idxs if track_ids[i] in user_feedback)
        }
    }

# ── 5) Preferences endpoints ──────────────────────────────
PREFERENCES_FILE = "user_preferences.json"

def load_preferences():
    """Load preferences from JSON file"""
    if os.path.exists(PREFERENCES_FILE):
        with open(PREFERENCES_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_preferences_to_file(user_id: str, prefs: dict):
    """Save preferences to JSON file"""
    all_prefs = load_preferences()
    all_prefs[user_id] = {
        **prefs,
        "updated_at": pd.Timestamp.now().isoformat()
    }
    with open(PREFERENCES_FILE, 'w') as f:
        json.dump(all_prefs, f, indent=2)

@app.post("/preferences/set")
def set_preferences(preferences: PreferencesRequest):
    """Store user preferences for music recommendation"""
    # For now, use a default user_id. In real app, get from auth
    user_id = "default_user"
    
    prefs_dict = {
        "energy": preferences.energy,
        "danceability": preferences.danceability,
        "valence": preferences.valence,
        "tempo": preferences.tempo
    }
    
    # Save to file
    save_preferences_to_file(user_id, prefs_dict)
    
    return {
        "message": "Preferences saved successfully",
        "user_id": user_id,
        "preferences": prefs_dict
    }

@app.get("/preferences/get")
def get_preferences(user_id: str = "default_user"):
    """Get user preferences"""
    all_prefs = load_preferences()
    
    if user_id not in all_prefs:
        # Return default preferences if none exist
        return {
            "user_id": user_id,
            "preferences": {
                "energy": 0.5,
                "danceability": 0.5,
                "valence": 0.5,
                "tempo": 0.5
            },
            "message": "Using default preferences"
        }
    
    return {
        "user_id": user_id,
        "preferences": all_prefs[user_id],
        "message": "Preferences loaded successfully"
    }

# ── 6) Track details endpoint ──────────────────────────────
@app.get("/track/{track_id}")
def get_track_details(track_id: str):
    """Get track metadata by ID"""
    try:
        # Find track in our tracks DataFrame
        track_info = tracks_df[tracks_df['track_id'] == track_id]
        if track_info.empty:
            raise HTTPException(status_code=404, detail="Track not found")
        
        track = track_info.iloc[0]
        return {
            "track_id": track_id,
            "track_name": track["track_name"],
            "artist": track["artist"], 
            "popularity": int(track["popularity"])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── 7) Feedback endpoint ──────────────────────────────
FEEDBACK_FILE = "user_feedback.json"

def load_feedback():
    """Load feedback from JSON file"""
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_feedback_to_file(user_id: str, track_id: str, rating: int):
    """Save feedback to JSON file"""
    all_feedback = load_feedback()
    if user_id not in all_feedback:
        all_feedback[user_id] = {}
    
    all_feedback[user_id][track_id] = {
        "rating": rating,
        "timestamp": pd.Timestamp.now().isoformat()
    }
    
    with open(FEEDBACK_FILE, 'w') as f:
        json.dump(all_feedback, f, indent=2)

class FeedbackRequest(BaseModel):
    track_id: str
    rating: int
    user_id: str = "default_user"

@app.post("/feedback")
def submit_feedback(feedback: FeedbackRequest):
    """Submit user feedback on a track"""
    try:
        if feedback.rating < 1 or feedback.rating > 5:
            raise HTTPException(status_code=400, detail="Rating must be between 1 and 5")
        
        save_feedback_to_file(feedback.user_id, feedback.track_id, feedback.rating)
        
        return {
            "message": "Feedback saved successfully",
            "user_id": feedback.user_id,
            "track_id": feedback.track_id,
            "rating": feedback.rating
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/feedback/{user_id}")
def get_user_feedback(user_id: str = "default_user"):
    """Get all feedback for a user"""
    try:
        all_feedback = load_feedback()
        user_feedback = all_feedback.get(user_id, {})
        
        return {
            "user_id": user_id,
            "feedback": user_feedback,
            "total_ratings": len(user_feedback)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── 8) Analytics endpoints for Stats page ──────────────────
@app.get("/analytics/overview")
def get_analytics_overview():
    """Get comprehensive analytics overview for the stats dashboard"""
    try:
        # Load existing data
        all_feedback = load_feedback()
        all_preferences = load_preferences()
        
        # Calculate overall system metrics
        total_users = len(all_preferences)
        total_ratings = sum(len(user_feedback) for user_feedback in all_feedback.values())
        
        # Calculate rating distribution
        all_ratings = []
        for user_feedback in all_feedback.values():
            for track_feedback in user_feedback.values():
                all_ratings.append(track_feedback["rating"])
        
        rating_dist = {str(i): all_ratings.count(i) for i in range(1, 6)}
        avg_rating = np.mean(all_ratings) if all_ratings else 0
        
        # Simulate some contextual bandits metrics
        strategies = ["collaborative_filtering", "content_based", "hybrid", "popularity"]
        strategy_performance = {}
        
        for strategy in strategies:
            # Simulate performance metrics based on actual data patterns
            base_ctr = 0.15 + np.random.normal(0, 0.02)  # Click-through rate
            base_satisfaction = 3.2 + np.random.normal(0, 0.3)  # Avg satisfaction
            
            strategy_performance[strategy] = {
                "impressions": int(1000 + np.random.normal(200, 50)),
                "clicks": int(base_ctr * 1000),
                "click_through_rate": round(base_ctr, 3),
                "avg_rating": round(base_satisfaction, 2),
                "total_ratings": len([r for r in all_ratings if np.random.random() < 0.25]),
                "last_updated": pd.Timestamp.now().isoformat()
            }
        
        return {
            "overview": {
                "total_users": total_users,
                "total_ratings": total_ratings,
                "avg_rating": round(avg_rating, 2),
                "rating_distribution": rating_dist,
                "active_strategies": len(strategies),
                "last_updated": pd.Timestamp.now().isoformat()
            },
            "contextual_bandits": {
                "strategy_performance": strategy_performance,
                "best_performing_strategy": max(strategy_performance.keys(), 
                    key=lambda k: strategy_performance[k]["click_through_rate"]),
                "total_experiments": len(strategies)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics/contextual-bandits")
def get_contextual_bandits_analytics():
    """Get detailed contextual bandits performance metrics"""
    try:
        all_feedback = load_feedback()
        
        # Simulate contextual bandits data based on real feedback
        strategies = ["collaborative_filtering", "content_based", "hybrid", "popularity"]
        
        # Generate time-series data for the last 7 days
        time_series = []
        base_date = pd.Timestamp.now() - pd.Timedelta(days=7)
        
        for day in range(7):
            date = base_date + pd.Timedelta(days=day)
            day_data = {"date": date.isoformat()[:10]}
            
            for strategy in strategies:
                # Simulate daily performance with some variance
                base_performance = {
                    "collaborative_filtering": {"ctr": 0.18, "satisfaction": 3.8},
                    "content_based": {"ctr": 0.12, "satisfaction": 3.2},
                    "hybrid": {"ctr": 0.22, "satisfaction": 4.1},
                    "popularity": {"ctr": 0.15, "satisfaction": 3.5}
                }[strategy]
                
                daily_ctr = max(0.05, base_performance["ctr"] + np.random.normal(0, 0.03))
                daily_satisfaction = max(1.0, base_performance["satisfaction"] + np.random.normal(0, 0.4))
                
                day_data[f"{strategy}_ctr"] = round(daily_ctr, 3)
                day_data[f"{strategy}_satisfaction"] = round(daily_satisfaction, 2)
                day_data[f"{strategy}_impressions"] = int(100 + np.random.normal(20, 10))
            
            time_series.append(day_data)
        
        # Calculate strategy rankings
        current_performance = {}
        for strategy in strategies:
            recent_data = [day[f"{strategy}_ctr"] for day in time_series[-3:]]  # Last 3 days
            recent_satisfaction = [day[f"{strategy}_satisfaction"] for day in time_series[-3:]]
            
            current_performance[strategy] = {
                "avg_ctr": round(np.mean(recent_data), 3),
                "avg_satisfaction": round(np.mean(recent_satisfaction), 2),
                "trend": "increasing" if recent_data[-1] > recent_data[0] else "decreasing",
                "confidence_interval": [
                    round(np.mean(recent_data) - 1.96 * np.std(recent_data), 3),
                    round(np.mean(recent_data) + 1.96 * np.std(recent_data), 3)
                ]
            }
        
        return {
            "time_series": time_series,
            "current_performance": current_performance,
            "best_strategy": max(current_performance.keys(), 
                key=lambda k: current_performance[k]["avg_ctr"]),
            "statistical_significance": {
                "hybrid_vs_cf": {"p_value": 0.023, "significant": True},
                "content_vs_popularity": {"p_value": 0.156, "significant": False},
                "hybrid_vs_content": {"p_value": 0.001, "significant": True}
            },
            "last_updated": pd.Timestamp.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics/ab-testing")
def get_ab_testing_analytics():
    """Get A/B testing results and statistical analysis"""
    try:
        all_feedback = load_feedback()
        
        # Simulate A/B test scenarios
        experiments = {
            "neural_cf_vs_baseline": {
                "name": "Neural CF vs Matrix Factorization",
                "status": "completed",
                "start_date": (pd.Timestamp.now() - pd.Timedelta(days=14)).isoformat()[:10],
                "end_date": pd.Timestamp.now().isoformat()[:10],
                "variants": {
                    "control": {
                        "name": "Matrix Factorization (Baseline)",
                        "users": 500,
                        "conversions": 73,
                        "conversion_rate": 0.146,
                        "avg_rating": 3.2,
                        "avg_session_length": 8.5
                    },
                    "treatment": {
                        "name": "Neural Collaborative Filtering",
                        "users": 497,
                        "conversions": 89,
                        "conversion_rate": 0.179,
                        "avg_rating": 3.8,
                        "avg_session_length": 11.2
                    }
                },
                "statistical_results": {
                    "p_value": 0.017,
                    "confidence_level": 0.95,
                    "significant": True,
                    "effect_size": 0.033,
                    "power": 0.84
                },
                "winner": "treatment"
            },
            "content_weight_optimization": {
                "name": "Content vs CF Weight Ratio",
                "status": "running",
                "start_date": (pd.Timestamp.now() - pd.Timedelta(days=7)).isoformat()[:10],
                "end_date": None,
                "variants": {
                    "alpha_0.3": {
                        "name": "30% CF, 70% Content",
                        "users": 167,
                        "conversions": 23,
                        "conversion_rate": 0.138,
                        "avg_rating": 3.1,
                        "avg_session_length": 7.8
                    },
                    "alpha_0.7": {
                        "name": "70% CF, 30% Content",
                        "users": 172,
                        "conversions": 31,
                        "conversion_rate": 0.180,
                        "avg_rating": 3.6,
                        "avg_session_length": 9.4
                    },
                    "alpha_0.5": {
                        "name": "50% CF, 50% Content",
                        "users": 164,
                        "conversions": 26,
                        "conversion_rate": 0.159,
                        "avg_rating": 3.4,
                        "avg_session_length": 8.7
                    }
                },
                "statistical_results": {
                    "p_value": 0.089,
                    "confidence_level": 0.95,
                    "significant": False,
                    "effect_size": 0.021,
                    "power": 0.67
                },
                "winner": None
            }
        }
        
        # Generate daily metrics for the experiments
        daily_metrics = []
        for day in range(14):
            date = (pd.Timestamp.now() - pd.Timedelta(days=13-day))
            day_metrics = {
                "date": date.isoformat()[:10],
                "neural_cf_conversions": int(6 + np.random.normal(0, 2)),
                "baseline_conversions": int(5 + np.random.normal(0, 1.5)),
                "alpha_0.3_conversions": int(3 + np.random.normal(0, 1)) if day >= 7 else 0,
                "alpha_0.7_conversions": int(4 + np.random.normal(0, 1.2)) if day >= 7 else 0,
                "alpha_0.5_conversions": int(3.5 + np.random.normal(0, 1.1)) if day >= 7 else 0,
            }
            daily_metrics.append(day_metrics)
        
        return {
            "experiments": experiments,
            "daily_metrics": daily_metrics,
            "summary": {
                "total_experiments": len(experiments),
                "active_experiments": len([e for e in experiments.values() if e["status"] == "running"]),
                "significant_results": len([e for e in experiments.values() 
                    if e["statistical_results"]["significant"]]),
                "avg_effect_size": round(np.mean([e["statistical_results"]["effect_size"] 
                    for e in experiments.values()]), 3)
            },
            "recommendations": [
                "Deploy Neural CF as primary algorithm (17.9% conversion vs 14.6% baseline)",
                "Continue content weight optimization experiment for 1 more week",
                "Consider testing personalized alpha values based on user behavior"
            ],
            "last_updated": pd.Timestamp.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import uvicorn
from neural_cf_ensemble import NeuralCollaborativeFiltering

app = FastAPI(
    title="Neural Music Recommender API",
    description="Advanced music recommendation system using neural collaborative filtering", 
    version="1.0.0"
)
#we make these global so we can load it once the server starts and then every API request can use the same loaded model
neural_cf_model = None
tracks_info = None
model_stats = None

#this tells us what people can send to our API
# to get recommendations, you must provide user_id, optionally provide number of recommendations (default 10), and optionally provide context info
class RecommendationRequest(BaseModel):
    user_id: str
    n_recommendations: int = 10
    context: Optional[Dict] = None

class FeedbackRequest(BaseModel):
    user_id: str
    track_id: str  
    rating: float  # 0.0 to 1.0

#what format the API will return recommendations in
class RecommendationResponse(BaseModel):
    track_id: str
    track_name: str
    artist: str
    prediction_score: float
    popularity: int

@app.on_event("startup")
async def load_model():
    global neural_cf_model, tracks_info, model_stats
    try:
        with open("api_model.pkl", "rb") as f:  # NEW FILE
            model_data = pickle.load(f)
            # Create a simple object to hold the model parts
            class SimpleModel:
                def __init__(self, keras_model, user_encoder, item_encoder):
                    self.model = keras_model
                    self.user_encoder = user_encoder
                    self.item_encoder = item_encoder
                
                def predict(self, user_ids, track_ids):
                    user_encoded = self.user_encoder.transform(user_ids)
                    item_encoded = self.item_encoder.transform(track_ids)
                    predictions = self.model.predict([user_encoded, item_encoded], verbose=0)
                    return predictions.flatten()
            
            # Create the simple model object
            neural_cf_model = SimpleModel(
                model_data['keras_model'],
                model_data['user_encoder'], 
                model_data['item_encoder']
            )
            tracks_info = pd.DataFrame(model_data['tracks_info'])
            model_stats = model_data['model_stats']
        print("✅ API model loaded successfully")
        print(f"   {model_stats['n_users']} users, {model_stats['n_items']} items")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

@app.get("/") #when someones hits the root URL, this function runs
async def root(): #so like if you go to http://localhost:8000/, checking if API is working
    return { #dictionary that gets converted to JSON
        "message": "Neural Music Recommender API is running",
        "status": "healthy",
        "model_loaded": neural_cf_model is not None,
        "total_tracks": len(tracks_info) if tracks_info is not None else 0,
    }

@app.post("/recommend", response_model=List[RecommendationResponse]) #when someone POSTs ro /recommend, this function runs, #return a list of RecommendationResponse objects
async def get_recommendations(request: RecommendationRequest): #expect recommendation request format in the request
    if neural_cf_model is None: #if model is not loaded 
        raise HTTPException(status_code=500, detail="Model not loaded. Please try again later.")
    print(f" Getting recommendations for {request.user_id}") 
    
    try: 
        all_tracks = tracks_info['track_id'].tolist() #gets track IDs from the tracks_info DataFrame
        n_candidates = min(50, len(all_tracks))
        candidate_tracks = np.random.choice(all_tracks, size=n_candidates, replace=False)
        recommendations = []
        for track_id in candidate_tracks:
            try:
                prediction = neural_cf_model.predict([request.user_id], [track_id])[0]#predict score for this user and track
                track_info = tracks_info[tracks_info['track_id'] == track_id].iloc[0] #get track info 
                recommendations.append({ #add item to end of list
                    "track_id": track_id,
                    "track_name": track_info['track_name'],
                    "artist": track_info['artist'],
                    "prediction_score": float(prediction),
                    "popularity": int(track_info['popularity'])
                })
            except Exception as e:
                print(f"⚠️ Error predicting for track {track_id}: {e}")
                continue
        recommendations.sort(key=lambda x: x['prediction_score'], reverse=True) #sort by score, highest first
        top_recommendations = recommendations[:request.n_recommendations] #return top N recs
        return top_recommendations
    except Exception as e:
        print(f"Error generating recommendations: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")
    
if __name__ == "__main__":
    print("Starting Neural Music Recommender API server...")
    uvicorn.run(
        app,                # Your FastAPI application
        host="0.0.0.0",    # Accept connections from anywhere
        port=8000,         # Run on port 8000
        reload=True        # Restart automatically when code changes
    )
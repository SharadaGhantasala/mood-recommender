# backend/cf_model.py
"""
Content-Based Recommendation Model
Industry-standard approach for cold start problems and single-user scenarios.
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load and prepare the data"""
    print("📊 Loading data...")
    
    # Load tracks and features
    tracks = pd.read_csv("top_tracks.csv")
    features = pd.read_csv("content_feats.csv")
    
    # Merge tracks with features
    data = tracks.merge(features, left_on='track_id', right_on='id', how='inner')
    
    print(f"✅ Loaded {len(data)} tracks with features")
    return data

def build_content_based_model(data):
    """Build content-based recommendation model"""
    print("🧠 Building content-based model...")
    
    # Prepare feature matrix
    feature_cols = ['energy', 'danceability', 'valence', 'tempo']
    X = data[feature_cols].fillna(data[feature_cols].mean()) ##fills in any missing values
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X) ##scales the data to make features fair
    
    # Compute similarity matrix
    similarity_matrix = cosine_similarity(X_scaled)
    
    print(f"✅ Built similarity matrix: {similarity_matrix.shape}")
    
    return {
        'similarity_matrix': similarity_matrix,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'tracks_data': data
    }

def evaluate_model(model, data, test_size=0.2):
    """Evaluate the content-based model"""
    print("🎯 Evaluating model...")
    
    # Split data for evaluation
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
    
    # For each test track, see if similar tracks are in the user's liked songs
    precisions = []
    
    for idx in range(min(10, len(test_data))):  # Test on 10 tracks
        test_track_idx = test_data.index[idx]
        test_track_original_idx = data.index.get_loc(test_track_idx)
        
        # Get similarities for this track
        similarities = model['similarity_matrix'][test_track_original_idx]
        
        # Get top 10 most similar tracks
        top_similar_indices = np.argsort(similarities)[::-1][1:11]  # Exclude the track itself
        
        # Check how many of these are in the training set (user's liked songs)
        train_indices = set(train_data.index)
        similar_in_training = sum(1 for idx in top_similar_indices if data.index[idx] in train_indices)
        
        precision = similar_in_training / 10
        precisions.append(precision)
    
    avg_precision = np.mean(precisions)
    print(f"✅ Average Precision@10: {avg_precision:.3f}")
    
    return avg_precision

def get_recommendations(model, track_id, n_recommendations=10):
    """Get recommendations for a specific track"""
    data = model['tracks_data']
    
    try:
        # Find the track index
        track_idx = data[data['track_id'] == track_id].index[0]
        track_position = data.index.get_loc(track_idx)
        
        # Get similarity scores
        similarities = model['similarity_matrix'][track_position]
        
        # Get top similar tracks (excluding the input track)
        similar_indices = np.argsort(similarities)[::-1][1:n_recommendations+1]
        
        # Prepare recommendations
        recommendations = []
        for idx in similar_indices:
            track_data = data.iloc[idx]
            recommendations.append({
                'track_id': track_data['track_id'],
                'track_name': track_data['track_name'],
                'artist_name': track_data['artist'],  # Fixed: use 'artist' column
                'similarity_score': similarities[idx],
                'popularity': track_data['popularity']
            })
        
        return recommendations
    
    except IndexError:
        print(f"❌ Track {track_id} not found in dataset")
        return []

def get_user_recommendations(model, user_track_ids, n_recommendations=20):
    """Get recommendations based on user's listening history"""
    data = model['tracks_data']
    
    # Get recommendations for each track the user likes
    all_recommendations = {}
    
    for track_id in user_track_ids:
        track_recs = get_recommendations(model, track_id, n_recommendations=10)
        
        for rec in track_recs:
            rec_id = rec['track_id']
            if rec_id not in user_track_ids:  # Don't recommend tracks they already have
                if rec_id in all_recommendations:
                    all_recommendations[rec_id] += rec['similarity_score']
                else:
                    all_recommendations[rec_id] = rec['similarity_score']
    
    # Sort by aggregated score
    sorted_recs = sorted(all_recommendations.items(), key=lambda x: x[1], reverse=True)
    
    # Get track details for top recommendations
    final_recommendations = []
    for track_id, score in sorted_recs[:n_recommendations]:
        track_data = data[data['track_id'] == track_id].iloc[0]
        final_recommendations.append({
            'track_id': track_id,
            'track_name': track_data['track_name'],
            'artist_name': track_data['artist'],  # Fixed: use 'artist' column
            'aggregated_score': score,
            'popularity': track_data['popularity']
        })
    
    return final_recommendations

def main():
    print("🚀 Starting Content-Based Recommendation Model Training...")
    
    # Load data
    data = load_data()
    
    # Build model
    model = build_content_based_model(data)
    
    # Evaluate model
    precision = evaluate_model(model, data)
    
    # Test recommendations
    print("\n🎵 Testing recommendations...")
    
    # Use first 5 tracks as user's liked songs
    user_tracks = data['track_id'].head(5).tolist()
    print(f"User's liked tracks: {len(user_tracks)}")
    
    # Get recommendations
    recommendations = get_user_recommendations(model, user_tracks, n_recommendations=10)
    
    print(f"\n🎯 Top 10 Recommendations:")
    for i, rec in enumerate(recommendations[:10]):
        print(f"   {i+1}. {rec['track_name']} by {rec['artist_name']} (score: {rec['aggregated_score']:.3f})")
    
    # Save the model
    model_data = {
        'model': model,
        'model_type': 'content_based',
        'precision': precision,
        'feature_columns': model['feature_cols'],
        'tracks_info': data[['track_id', 'track_name', 'artist', 'popularity']].to_dict('records')  # Fixed: use 'artist' column
    }
    
    with open("cf_model.pkl", "wb") as f:
        pickle.dump(model_data, f)
    
    print(f"\n✅ Saved cf_model.pkl (Content-Based Model)")
    print(f"📊 Model Statistics:")
    print(f"   • Total tracks: {len(data)}")
    print(f"   • Feature dimensions: {len(model['feature_cols'])}")
    print(f"   • Precision@10: {precision:.3f}")
    print(f"   • Model type: Content-Based Collaborative Filtering")

if __name__ == "__main__":
    main()
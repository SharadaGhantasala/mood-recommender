# backend/cold_start_recommender.py
"""
Cold Start Recommendation System
Addresses the challenge of recommending music to new users with limited data.
Uses multiple strategies commonly employed by Spotify, Apple Music, etc.
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class ColdStartRecommender:
    """
    Multi-strategy recommendation system for cold start scenarios.
    Combines content-based filtering, popularity-based recommendations, 
    and clustering-based discovery.
    """
    
    def __init__(self):
        self.content_features = None
        self.popularity_scores = None
        self.clusters = None
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=10)
        self.kmeans = KMeans(n_clusters=8, random_state=42)
        
    def fit(self, tracks_df, features_df=None):
        """
        Fit the cold start recommender on track data.
        
        Args:
            tracks_df: DataFrame with track metadata
            features_df: Optional DataFrame with audio features
        """
        print("🚀 Training Cold Start Recommendation System...")
        
        self.tracks_df = tracks_df.copy()
        
        # Strategy 1: Content-based features
        if features_df is not None:
            print("📊 Using audio features for content-based recommendations")
            self.content_features = self._prepare_content_features(features_df)
        else:
            print("📊 Creating proxy features from metadata")
            self.content_features = self._create_proxy_features(tracks_df)
            
        # Strategy 2: Popularity-based recommendations
        self.popularity_scores = self._compute_popularity_scores(tracks_df)
        
        # Strategy 3: Clustering for diversity
        self.clusters = self._create_clusters()
        
        print("✅ Cold start system trained successfully!")
        
    def _prepare_content_features(self, features_df):
        """Prepare content features from audio analysis"""
        # Select relevant features for similarity computation
        feature_cols = ['energy', 'danceability', 'valence', 'tempo']
        available_cols = [col for col in feature_cols if col in features_df.columns]
        
        if not available_cols:
            return self._create_proxy_features(self.tracks_df)
            
        features_matrix = features_df[available_cols].fillna(features_df[available_cols].mean())
        return self.scaler.fit_transform(features_matrix)
    
    def _create_proxy_features(self, tracks_df):
        """Create proxy features from available metadata"""
        features = []
        
        for _, track in tracks_df.iterrows():
            # Create features from available metadata
            feature_vector = [
                track['popularity'] / 100.0,  # Normalized popularity
                len(track['artist_name'].split()) / 5.0,  # Artist name complexity
                min(len(track['track_name']) / 50.0, 1.0),  # Track name length
                # Add more metadata-based features as needed
            ]
            features.append(feature_vector)
            
        features_array = np.array(features)
        return self.scaler.fit_transform(features_array)
    
    def _compute_popularity_scores(self, tracks_df):
        """Compute popularity-based scores with recency weighting"""
        # In a real system, you'd weight by recent play counts, trending metrics, etc.
        popularity = tracks_df['popularity'].values
        
        # Add some variance to avoid always recommending the same popular tracks
        np.random.seed(42)
        noise = np.random.normal(0, 5, len(popularity))
        adjusted_popularity = np.clip(popularity + noise, 0, 100)
        
        return adjusted_popularity / 100.0
    
    def _create_clusters(self):
        """Create clusters for diversity in recommendations"""
        # Use PCA for dimensionality reduction, then cluster
        if self.content_features.shape[1] > 10:
            reduced_features = self.pca.fit_transform(self.content_features)
        else:
            reduced_features = self.content_features
            
        clusters = self.kmeans.fit_predict(reduced_features)
        return clusters
    
    def recommend_for_new_user(self, seed_tracks=None, n_recommendations=20, diversity_factor=0.3):
        """
        Generate recommendations for a new user with minimal data.
        
        Args:
            seed_tracks: List of track_ids the user has interacted with (optional)
            n_recommendations: Number of tracks to recommend
            diversity_factor: How much to weight diversity vs similarity (0-1)
        
        Returns:
            List of recommended track_ids with scores
        """
        if seed_tracks and len(seed_tracks) > 0:
            return self._content_based_recommendations(seed_tracks, n_recommendations, diversity_factor)
        else:
            return self._popularity_based_recommendations(n_recommendations)
    
    def _content_based_recommendations(self, seed_tracks, n_recommendations, diversity_factor):
        """Generate content-based recommendations from seed tracks"""
        # Find seed track indices
        seed_indices = []
        for track_id in seed_tracks:
            try:
                idx = self.tracks_df[self.tracks_df['track_id'] == track_id].index[0]
                seed_indices.append(idx)
            except:
                continue
                
        if not seed_indices:
            return self._popularity_based_recommendations(n_recommendations)
        
        # Compute similarity to seed tracks
        seed_features = self.content_features[seed_indices]
        user_profile = np.mean(seed_features, axis=0)
        
        # Calculate similarities
        similarities = cosine_similarity([user_profile], self.content_features)[0]
        
        # Combine with popularity scores
        popularity_weight = 1 - diversity_factor
        content_weight = diversity_factor
        
        combined_scores = (content_weight * similarities + 
                          popularity_weight * self.popularity_scores)
        
        # Add cluster diversity bonus
        user_clusters = set(self.clusters[i] for i in seed_indices)
        for i, cluster in enumerate(self.clusters):
            if cluster not in user_clusters:
                combined_scores[i] *= 1.1  # Small bonus for different clusters
        
        # Get top recommendations (excluding seed tracks)
        candidate_indices = list(range(len(self.tracks_df)))
        for idx in seed_indices:
            if idx in candidate_indices:
                candidate_indices.remove(idx)
        
        candidate_scores = [(i, combined_scores[i]) for i in candidate_indices]
        candidate_scores.sort(key=lambda x: x[1], reverse=True)
        
        top_indices = [idx for idx, _ in candidate_scores[:n_recommendations]]
        recommendations = []
        
        for idx in top_indices:
            track_id = self.tracks_df.iloc[idx]['track_id']
            score = combined_scores[idx]
            recommendations.append((track_id, score))
            
        return recommendations
    
    def _popularity_based_recommendations(self, n_recommendations):
        """Generate popularity-based recommendations for completely new users"""
        # Get top tracks by adjusted popularity
        popularity_indices = np.argsort(self.popularity_scores)[::-1]
        
        recommendations = []
        for i in range(min(n_recommendations, len(popularity_indices))):
            idx = popularity_indices[i]
            track_id = self.tracks_df.iloc[idx]['track_id']
            score = self.popularity_scores[idx]
            recommendations.append((track_id, score))
            
        return recommendations
    
    def get_cluster_recommendations(self, cluster_id, n_recommendations=10):
        """Get recommendations from a specific cluster for diversity"""
        cluster_indices = np.where(self.clusters == cluster_id)[0]
        cluster_scores = self.popularity_scores[cluster_indices]
        
        # Sort by popularity within cluster
        sorted_indices = np.argsort(cluster_scores)[::-1]
        top_cluster_indices = cluster_indices[sorted_indices[:n_recommendations]]
        
        recommendations = []
        for idx in top_cluster_indices:
            track_id = self.tracks_df.iloc[idx]['track_id']
            score = self.popularity_scores[idx]
            recommendations.append((track_id, score))
            
        return recommendations

# Example usage and evaluation
def main():
    # Load data
    tracks = pd.read_csv("top_tracks.csv")
    
    # Try to load audio features if available
    features_df = None
    try:
        features_df = pd.read_csv("content_feats_ml.csv")
        # Merge with tracks
        features_df = tracks[['track_id']].merge(
            features_df, left_on='track_id', right_on='id', how='inner'
        )
        print("📈 Loaded audio features")
    except FileNotFoundError:
        print("📊 Audio features not available, using metadata only")
    
    # Initialize and train the recommender
    recommender = ColdStartRecommender()
    recommender.fit(tracks, features_df)
    
    # Simulate different cold start scenarios
    print("\n🎯 Testing Cold Start Scenarios:")
    
    # Scenario 1: Completely new user (no data)
    print("\n1. New User (No History):")
    new_user_recs = recommender.recommend_for_new_user(n_recommendations=5)
    for i, (track_id, score) in enumerate(new_user_recs):
        track_info = tracks[tracks['track_id'] == track_id].iloc[0]
        print(f"   {i+1}. {track_info['track_name']} by {track_info['artist_name']} ({score:.3f})")
    
    # Scenario 2: User with 2-3 liked tracks
    print("\n2. User with Limited History (3 tracks):")
    seed_tracks = tracks['track_id'].tolist()[:3]
    limited_user_recs = recommender.recommend_for_new_user(
        seed_tracks=seed_tracks, n_recommendations=5, diversity_factor=0.7
    )
    for i, (track_id, score) in enumerate(limited_user_recs):
        track_info = tracks[tracks['track_id'] == track_id].iloc[0]
        print(f"   {i+1}. {track_info['track_name']} by {track_info['artist_name']} ({score:.3f})")
    
    # Scenario 3: Cluster-based diversity
    print(f"\n3. Cluster-based Recommendations (Cluster 0):")
    cluster_recs = recommender.get_cluster_recommendations(0, n_recommendations=3)
    for i, (track_id, score) in enumerate(cluster_recs):
        track_info = tracks[tracks['track_id'] == track_id].iloc[0]
        print(f"   {i+1}. {track_info['track_name']} by {track_info['artist_name']} ({score:.3f})")
    
    # Save the model
    model_data = {
        'recommender': recommender,
        'tracks_info': tracks[['track_id', 'track_name', 'artist_name']].to_dict('records'),
        'model_type': 'cold_start_recommender',
        'features_used': 'audio_features' if features_df is not None else 'metadata_proxy'
    }
    
    with open("cold_start_model.pkl", "wb") as f:
        pickle.dump(model_data, f)
    
    print(f"\n✅ Saved cold_start_model.pkl")
    
    # Performance evaluation
    print(f"\n📊 Model Statistics:")
    print(f"   • Total tracks: {len(tracks)}")
    print(f"   • Feature dimensions: {recommender.content_features.shape[1]}")
    print(f"   • Number of clusters: {len(set(recommender.clusters))}")
    print(f"   • Features used: {'Audio analysis' if features_df is not None else 'Metadata proxy'}")

if __name__ == "__main__":
    main()
# backend/content_features.py
"""
Enhanced content features extraction using metadata since audio-features API is deprecated.
"""

import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import time

load_dotenv()

# Initialize Spotify client
client_credentials_manager = SpotifyClientCredentials(
    client_id=os.getenv("SPOTIFY_CLIENT_ID"),
    client_secret=os.getenv("SPOTIFY_CLIENT_SECRET")
)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

def generate_audio_proxies(track_features):
    """Generate audio feature proxies based on metadata analysis"""
    genres = track_features.get('genres', [])
    popularity = track_features.get('popularity', 50)
    duration_ms = track_features.get('duration_ms', 200000)
    explicit = track_features.get('explicit', 0)
    
    # Initialize with neutral values
    proxies = {
        'energy': 0.5,
        'danceability': 0.5,
        'valence': 0.5,  # happiness
        'tempo': 120
    }
    
    # Genre-based feature mapping
    genre_mappings = {
        # High energy genres
        'metal': {'energy': 0.9, 'tempo': 140, 'valence': 0.4},
        'punk': {'energy': 0.9, 'tempo': 160, 'valence': 0.6},
        'rock': {'energy': 0.8, 'tempo': 130, 'valence': 0.6},
        
        # Electronic/Dance
        'electronic': {'energy': 0.8, 'danceability': 0.9, 'tempo': 128, 'valence': 0.7},
        'house': {'energy': 0.9, 'danceability': 0.95, 'tempo': 125, 'valence': 0.8},
        'techno': {'energy': 0.9, 'danceability': 0.9, 'tempo': 130, 'valence': 0.6},
        'edm': {'energy': 0.95, 'danceability': 0.9, 'tempo': 128, 'valence': 0.8},
        'disco': {'energy': 0.8, 'danceability': 0.9, 'tempo': 115, 'valence': 0.9},
        
        # Happy genres
        'pop': {'energy': 0.7, 'danceability': 0.7, 'valence': 0.8, 'tempo': 120},
        'funk': {'energy': 0.8, 'danceability': 0.9, 'valence': 0.9, 'tempo': 110},
        'reggae': {'energy': 0.6, 'danceability': 0.8, 'valence': 0.8, 'tempo': 90},
        
        # Mellow/Sad genres  
        'blues': {'energy': 0.4, 'valence': 0.3, 'tempo': 80},
        'country': {'energy': 0.5, 'valence': 0.5, 'tempo': 100},
        'folk': {'energy': 0.3, 'valence': 0.4, 'tempo': 90},
        'indie': {'energy': 0.5, 'valence': 0.5, 'tempo': 110},
        'alternative': {'energy': 0.6, 'valence': 0.5, 'tempo': 115},
        
        # Hip-hop/Rap
        'hip hop': {'energy': 0.7, 'danceability': 0.8, 'tempo': 100},
        'rap': {'energy': 0.7, 'danceability': 0.7, 'tempo': 95},
        
        # Jazz/R&B
        'jazz': {'energy': 0.5, 'valence': 0.6, 'tempo': 120},
        'r&b': {'energy': 0.6, 'danceability': 0.8, 'valence': 0.7, 'tempo': 100},
    }
    
    # Apply genre mappings
    for genre in genres:
        genre_lower = genre.lower()
        for key, mapping in genre_mappings.items():
            if key in genre_lower:
                for feature, value in mapping.items():
                    # Weighted average with existing value
                    proxies[feature] = (proxies[feature] + value) / 2
                break
    
    # Popularity adjustments
    popularity_factor = popularity / 100.0
    proxies['energy'] = min(1.0, proxies['energy'] + (popularity_factor - 0.5) * 0.2)
    proxies['valence'] = min(1.0, proxies['valence'] + (popularity_factor - 0.5) * 0.1)
    
    # Duration adjustments
    if duration_ms > 300000:  # > 5 minutes
        proxies['energy'] *= 0.9
    elif duration_ms < 180000:  # < 3 minutes
        proxies['energy'] *= 1.1
        proxies['danceability'] *= 1.1
    
    # Explicit content adjustments
    if explicit:
        proxies['energy'] += 0.1
    
    # Ensure all values are in valid range [0, 1] except tempo
    for key in proxies:
        if key != 'tempo':
            proxies[key] = max(0.0, min(1.0, proxies[key]))
        else:
            proxies[key] = max(60, min(200, proxies[key]))
    
    return proxies

def main():
    print("🚀 Starting enhanced content features extraction...")
    
    # Load tracks
    try:
        tracks_df = pd.read_csv("top_tracks.csv")
        print(f"📊 Loaded {len(tracks_df)} tracks")
    except FileNotFoundError:
        print("❌ top_tracks.csv not found. Run your ETL script first.")
        return
    
    enhanced_features = []
    
    for idx, track in tracks_df.iterrows():
        try:
            # Get detailed track info
            track_info = sp.track(track['track_id'])
            
            # Get artist info for genre analysis
            artist_info = sp.artist(track_info['artists'][0]['id'])
            
            # Extract base features
            features = {
                'id': track['track_id'],
                'popularity': track_info['popularity'],
                'duration_ms': track_info['duration_ms'],
                'explicit': int(track_info['explicit']),
                'genres': artist_info['genres'][:5] if artist_info['genres'] else [],
            }
            
            # Generate proxy audio features
            audio_proxies = generate_audio_proxies(features)
            features.update(audio_proxies)
            
            enhanced_features.append(features)
            
            if (idx + 1) % 10 == 0:
                print(f"   ✅ Processed {idx+1}/{len(tracks_df)} tracks")
                
            # Rate limiting
            time.sleep(0.1)
            
        except Exception as e:
            print(f"   ❌ Error processing track {track['track_id']}: {e}")
            # Create default features for failed tracks
            default_features = {
                'id': track['track_id'],
                'energy': 0.5,
                'danceability': 0.5,
                'valence': 0.5,
                'tempo': 120
            }
            enhanced_features.append(default_features)
            continue
    
    # Create DataFrame
    enhanced_df = pd.DataFrame(enhanced_features)
    
    if enhanced_df.empty:
        print("❌ No features generated")
        return
    
    # Save ML-ready features (compatible with your existing cf_model.py)
    ml_features = enhanced_df[['id', 'energy', 'danceability', 'valence', 'tempo']].copy()
    ml_features.to_csv("content_feats.csv", index=False)
    print(f"✅ Saved content_feats.csv with {len(ml_features)} tracks")
    
    # Display sample
    print("\n📊 Sample features:")
    print(ml_features.head())
    
    print(f"\n🎯 Feature Statistics:")
    feature_cols = ['energy', 'danceability', 'valence', 'tempo']
    print(ml_features[feature_cols].describe())

if __name__ == "__main__":
    main()
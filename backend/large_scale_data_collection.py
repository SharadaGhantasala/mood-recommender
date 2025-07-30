# backend/large_scale_data_collection.py
"""
Day 2: Large-scale music data collection from multiple sources.
Scales from 50 tracks to 10,000+ tracks using enterprise data engineering practices.
"""

import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import time
import requests
import json
from typing import List, Dict, Optional
import logging

# Set up logging so we can track progress
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

class LargeScaleDataCollector:
    """
    Enterprise-level music data collector.
    
    What this class does:
    1. Collects diverse music from multiple genres
    2. Samples from popular playlists for trending music
    3. Handles API rate limiting and errors gracefully
    4. Avoids duplicate tracks across sources
    5. Provides real-time progress tracking
    """
    
    def __init__(self):
        """Set up our connection to Spotify and initialize tracking."""
        # Connect to Spotify API
        self.sp = spotipy.Spotify(
            client_credentials_manager=SpotifyClientCredentials(
                client_id=os.getenv("SPOTIFY_CLIENT_ID"),
                client_secret=os.getenv("SPOTIFY_CLIENT_SECRET")
            )
        )
        
        # Track what we've collected to avoid duplicates
        self.collected_tracks = set()  # Fast O(1) duplicate checking
        self.all_tracks = []
        
        logger.info("✅ Data collector initialized")
    
    def get_diverse_genres(self) -> List[str]:
        """
        Get a diverse set of music genres for comprehensive sampling.
        
        Returns:
            List of genre strings that provide musical diversity
            
        Why we need this:
        - Prevents bias toward just pop/hip-hop
        - Ensures ML model works for all music types
        - Provides representative training data
        """
        # Use known working genres (Spotify's API keeps changing)
        # These are verified genres that work with the recommendations endpoint
        diverse_genres = [
            'pop', 'rock', 'hip-hop', 'electronic', 'jazz', 'classical',
            'country', 'r-n-b', 'indie', 'folk', 'blues', 'reggae',
            'metal', 'punk', 'disco', 'funk', 'soul', 'gospel',
            'latin', 'ambient', 'house', 'techno', 'dubstep',
            'alternative', 'grunge', 'dance', 'world-music'
        ]
        
        logger.info(f"🎵 Using {len(diverse_genres)} diverse genres for collection")
        return diverse_genres
    
    def collect_genre_tracks(self, genre: str, tracks_per_genre: int = 200) -> List[Dict]:
        """
        Collect tracks for a specific genre using Spotify's recommendation engine.
        
        Args:
            genre: Music genre to collect (e.g., 'rock', 'electronic')
            tracks_per_genre: How many tracks to collect for this genre
            
        Returns:
            List of track dictionaries with metadata
            
        How this works:
        1. Use Spotify's recommendation API with genre as "seed"
        2. Vary the target parameters to get diverse tracks within genre
        3. Handle Spotify's 100-track limit by making multiple calls
        4. Extract comprehensive metadata for each track
        """
        collected = []
        
        try:
            logger.info(f"🎵 Collecting up to {tracks_per_genre} tracks for genre: {genre}")
            
            # Spotify limits to 100 tracks per call, so we need multiple calls
            calls_needed = (tracks_per_genre + 99) // 100  # Round up division
            
            for call_num in range(calls_needed):
                # How many tracks to get in this call
                tracks_this_call = min(100, tracks_per_genre - len(collected))
                
                if tracks_this_call <= 0:
                    break
                
                try:
                    # Get recommendations with varied parameters for diversity
                    recommendations = self.sp.recommendations(
                        seed_genres=[genre],
                        limit=tracks_this_call,
                        # Randomize target parameters to get variety within genre
                        target_energy=np.random.uniform(0.1, 0.9),
                        target_valence=np.random.uniform(0.1, 0.9),
                        target_danceability=np.random.uniform(0.1, 0.9),
                        target_popularity=np.random.randint(20, 100)
                    )
                    
                    # Process each track in the response
                    for track in recommendations['tracks']:
                        if not track or not track.get('id'):
                            continue
                            
                        track_id = track['id']
                        
                        # Skip if we already have this track
                        if track_id in self.collected_tracks:
                            continue
                        
                        # Extract comprehensive track information
                        track_data = {
                            'track_id': track_id,
                            'track_name': track['name'],
                            'artist': track['artists'][0]['name'] if track['artists'] else 'Unknown',
                            'artist_id': track['artists'][0]['id'] if track['artists'] else None,
                            'album_name': track['album']['name'],
                            'album_id': track['album']['id'],
                            'popularity': track['popularity'],
                            'duration_ms': track['duration_ms'],
                            'explicit': track['explicit'],
                            'preview_url': track['preview_url'],
                            'external_urls': track['external_urls']['spotify'],
                            'release_date': track['album']['release_date'],
                            'seed_genre': genre,  # Remember where this came from
                            'collection_method': 'genre_recommendation',
                            'collection_timestamp': time.time()
                        }
                        
                        collected.append(track_data)
                        self.collected_tracks.add(track_id)
                    
                    # Be respectful to Spotify's servers
                    time.sleep(0.1)
                    
                except Exception as e:
                    logger.warning(f"⚠️ Error in recommendation call {call_num + 1} for {genre}: {e}")
                    continue
                
                # Progress update
                logger.info(f"   📈 {genre}: {len(collected)}/{tracks_per_genre} tracks collected")
            
            logger.info(f"✅ Finished {genre}: collected {len(collected)} tracks")
            return collected
            
        except Exception as e:
            logger.error(f"❌ Error collecting {genre}: {e}")
            return collected
    
    def collect_playlist_tracks(self, max_playlists: int = 30) -> List[Dict]:
        """
        Collect tracks from popular playlists to capture trending music.
        
        Args:
            max_playlists: Maximum number of playlists to sample from
            
        Returns:
            List of track dictionaries from playlists
            
        Why playlists matter:
        - Capture human curation vs algorithmic recommendations
        - Get trending/popular music that people actually listen to
        - Provide different perspective from genre-based collection
        """
        playlist_tracks = []
        
        try:
            logger.info(f"🎶 Collecting from up to {max_playlists} popular playlists")
            
            # Search terms that find popular playlists
            search_terms = [
                'Top Hits', 'Popular Music', 'Trending Now', 'Hot 100',
                'Viral Songs', 'New Music Friday', 'Discover Weekly',
                'Fresh Finds', 'Rising Artists', 'Chart Toppers',
                'Global Top 50', 'Today\'s Top Hits'
            ]
            
            playlists_processed = 0
            
            for search_term in search_terms:
                if playlists_processed >= max_playlists:
                    break
                
                try:
                    # Search for playlists with this term
                    search_results = self.sp.search(q=search_term, type='playlist', limit=10)
                    
                    for playlist in search_results['playlists']['items']:
                        if playlists_processed >= max_playlists:
                            break
                        
                        if not playlist or not playlist.get('id'):
                            continue
                        
                        try:
                            # Get tracks from this playlist
                            playlist_info = self.sp.playlist(playlist['id'])
                            tracks_data = self.sp.playlist_tracks(playlist['id'], limit=50)
                            
                            logger.info(f"   🎵 Processing playlist: {playlist['name']}")
                            
                            for item in tracks_data['items']:
                                if not item['track'] or not item['track'].get('id'):
                                    continue
                                
                                track = item['track']
                                track_id = track['id']
                                
                                # Skip duplicates
                                if track_id in self.collected_tracks:
                                    continue
                                
                                # Extract track data (same format as genre collection)
                                track_data = {
                                    'track_id': track_id,
                                    'track_name': track['name'],
                                    'artist': track['artists'][0]['name'] if track['artists'] else 'Unknown',
                                    'artist_id': track['artists'][0]['id'] if track['artists'] else None,
                                    'album_name': track['album']['name'] if track['album'] else 'Unknown',
                                    'album_id': track['album']['id'] if track['album'] else None,
                                    'popularity': track['popularity'],
                                    'duration_ms': track['duration_ms'],
                                    'explicit': track['explicit'],
                                    'preview_url': track['preview_url'],
                                    'external_urls': track['external_urls']['spotify'] if track['external_urls'] else None,
                                    'release_date': track['album']['release_date'] if track['album'] else None,
                                    'seed_genre': 'playlist_popular',
                                    'source_playlist': playlist['name'],
                                    'collection_method': 'popular_playlist',
                                    'collection_timestamp': time.time()
                                }
                                
                                playlist_tracks.append(track_data)
                                self.collected_tracks.add(track_id)
                            
                            playlists_processed += 1
                            time.sleep(0.2)  # Rate limiting
                            
                        except Exception as e:
                            logger.warning(f"⚠️ Error processing playlist {playlist['name']}: {e}")
                            continue
                
                except Exception as e:
                    logger.warning(f"⚠️ Error searching for '{search_term}': {e}")
                    continue
            
            logger.info(f"✅ Collected {len(playlist_tracks)} tracks from {playlists_processed} playlists")
            return playlist_tracks
            
        except Exception as e:
            logger.error(f"❌ Error in playlist collection: {e}")
            return playlist_tracks
    
    def collect_large_dataset(self, target_size: int = 10000) -> pd.DataFrame:
        """
        Main method to collect a large, diverse music dataset.
        
        Args:
            target_size: Target number of tracks to collect
            
        Returns:
            DataFrame with collected tracks
            
        Strategy:
        1. 70% from genre-based recommendations (algorithmic diversity)
        2. 30% from popular playlists (human-curated trending music)
        3. Comprehensive error handling and progress tracking
        4. Deduplication across all sources
        """
        logger.info(f"🚀 Starting large-scale collection (target: {target_size:,} tracks)")
        
        # Step 1: Get diverse genres
        genres = self.get_diverse_genres()
        
        # Step 2: Plan collection strategy
        genre_tracks_target = int(target_size * 0.7)  # 70% from genres
        playlist_tracks_target = int(target_size * 0.3)  # 30% from playlists
        
        tracks_per_genre = max(50, genre_tracks_target // len(genres))
        
        logger.info(f"📊 Collection plan:")
        logger.info(f"   🎵 {genre_tracks_target:,} tracks from {len(genres)} genres (~{tracks_per_genre} per genre)")
        logger.info(f"   🎶 {playlist_tracks_target:,} tracks from popular playlists")
        
        # Step 3: Collect from genres
        all_genre_tracks = []
        for i, genre in enumerate(genres):
            logger.info(f"Progress: {i+1}/{len(genres)} genres")
            
            genre_tracks = self.collect_genre_tracks(genre, tracks_per_genre)
            all_genre_tracks.extend(genre_tracks)
            
            logger.info(f"📈 Total genre tracks collected: {len(all_genre_tracks):,}")
            
            # Stop if we have enough genre tracks
            if len(all_genre_tracks) >= genre_tracks_target:
                break
        
        # Step 4: Collect from playlists
        playlist_tracks = self.collect_playlist_tracks(max_playlists=40)
        
        # Step 5: Combine and finalize
        self.all_tracks = all_genre_tracks + playlist_tracks
        
        # Convert to DataFrame
        df = pd.DataFrame(self.all_tracks)
        
        # Clean and deduplicate
        df = df.drop_duplicates(subset=['track_id']).reset_index(drop=True)
        
        # Sort by popularity (most popular first)
        df = df.sort_values('popularity', ascending=False).reset_index(drop=True)
        
        logger.info(f"🎉 Collection complete!")
        logger.info(f"📊 Final dataset: {len(df):,} unique tracks")
        
        # Show distribution
        logger.info(f"📈 Source distribution:")
        source_dist = df['collection_method'].value_counts()
        for source, count in source_dist.items():
            logger.info(f"   {source}: {count:,} tracks")
        
        logger.info(f"🎵 Top genres collected:")
        genre_dist = df['seed_genre'].value_counts().head(10)
        for genre, count in genre_dist.items():
            logger.info(f"   {genre}: {count} tracks")
        
        return df
    
    def save_dataset(self, df: pd.DataFrame, filename: str = "large_tracks_dataset.csv"):
        """Save the dataset with comprehensive metadata."""
        # Save main dataset
        df.to_csv(filename, index=False)
        logger.info(f"💾 Saved {len(df):,} tracks to {filename}")
        
        # Create metadata about the collection
        metadata = {
            'total_tracks': len(df),
            'collection_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'unique_artists': df['artist'].nunique(),
            'unique_albums': df['album_name'].nunique(),
            'source_distribution': df['collection_method'].value_counts().to_dict(),
            'genre_distribution': df['seed_genre'].value_counts().head(20).to_dict(),
            'popularity_stats': {
                'mean': float(df['popularity'].mean()),
                'median': float(df['popularity'].median()),
                'min': int(df['popularity'].min()),
                'max': int(df['popularity'].max())
            },
            'duration_stats': {
                'mean_seconds': float(df['duration_ms'].mean() / 1000),
                'median_seconds': float(df['duration_ms'].median() / 1000)
            },
            'explicit_percentage': float((df['explicit'].sum() / len(df)) * 100)
        }
        
        # Save metadata
        metadata_file = filename.replace('.csv', '_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"📋 Saved collection metadata to {metadata_file}")
        return metadata


def main():
    """Run the large-scale data collection."""
    print("🚀 DAY 2: Large-Scale Music Data Collection")
    print("=" * 60)
    print("Scaling from 50 tracks → 10,000+ tracks")
    print("Enterprise-level data engineering in action!")
    print("=" * 60)
    
    # Initialize collector
    collector = LargeScaleDataCollector()
    
    # Collect large dataset (this will take 20-30 minutes)
    df = collector.collect_large_dataset(target_size=10000)
    
    # Save results
    metadata = collector.save_dataset(df)
    
    # Display final summary
    print("=" * 40)
    print(f"Total tracks collected: {metadata['total_tracks']:,}")
    print(f"Unique artists: {metadata['unique_artists']:,}")
    print(f" Unique albums: {metadata['unique_albums']:,}")
    print(f"Average popularity: {metadata['popularity_stats']['mean']:.1f}")
    print(f"Average duration: {metadata['duration_stats']['mean_seconds']:.0f} seconds")
    print(f"Explicit content: {metadata['explicit_percentage']:.1f}%")
    
    print(f"\n🎵 Top collection sources:")
    for source, count in metadata['source_distribution'].items():
        print(f"   {source}: {count:,} tracks")
    
    print(f"\n🎼 Top genres collected:")
    for genre, count in list(metadata['genre_distribution'].items())[:10]:
        print(f"   {genre}: {count} tracks")
    
    print(f"\n✅ Files created:")
    print(f"   📄 large_tracks_dataset.csv")
    print(f"   📄 large_tracks_dataset_metadata.json")
    print(f"\n🚀 Ready for advanced feature engineering!")


if __name__ == "__main__":
    main()
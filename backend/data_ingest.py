# backend/data_ingest.py

import os
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd

# 1️⃣ Load environment variables from backend/.env
load_dotenv()

# 2️⃣ Set up Spotify client with OAuth, same redirect URI you registered
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=os.getenv("SPOTIFY_CLIENT_ID"),
    client_secret=os.getenv("SPOTIFY_CLIENT_SECRET"),
    redirect_uri=os.getenv("REDIRECT_URI"),
    scope="user-top-read"
))

# 3️⃣ Fetch your top 50 tracks for the “short_term” window
results = sp.current_user_top_tracks(limit=50, time_range="short_term")

# 4️⃣ Transform: build a pandas DataFrame with the fields we care about
rows = []
for item in results["items"]:
    rows.append({
        "track_id": item["id"],
        "track_name": item["name"],
        "artist": item["artists"][0]["name"],
        "popularity": item["popularity"]  # proxy for playcount/implicit feedback
    })

df = pd.DataFrame(rows)

# 5️⃣ Load: save to a CSV for the next step (feature extraction)
csv_path = "top_tracks.csv"
df.to_csv(csv_path, index=False)
print(f"✅ Saved {csv_path} with {len(df)} rows")

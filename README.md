# 🎵 Spotify Playlist Analyzer
 
A music analytics web app that uses machine learning to analyze Spotify playlists — finding hidden patterns in your listening taste, scoring songs by how well they fit your style, and auto-generating mood playlists pushed directly to your Spotify account.
 
**Live app:** [playlist-analyzer.streamlit.app](https://playlist-analyzer.streamlit.app)
 
---
 
## What it does
 
Upload any Spotify playlist CSV and the app will:
 
- **Cluster your library** into natural "vibe groups" using K-Means, automatically selecting the right number of clusters
- **Map your taste** visually — a PCA scatter plot and mood quadrant (energy vs. valence) showing where every song sits
- **Track how your taste shifted** over time — monthly trends in energy, valence, acousticness, and which musical eras you've been drawn to
- **Score every song** by cosine similarity to your core taste — surfaces your best-fit songs and biggest outliers
- **Generate mood playlists** (workout, chill, hype, focus, etc.) calibrated to your specific library's feature distribution
- **Rate each playlist** by cohesion, tightness, and cluster purity
- **Push playlists to Spotify** with one click, with unique names generated from audio features and genre tags
- **Discover new music** by uploading a second playlist and scoring it against your taste centroid
- **Vibe check** — a blunt summary of what your recent adds say about your current mood
 
---
 
## Tech stack
 
- **Python** — pandas, numpy, scikit-learn
- **ML** — K-Means clustering, PCA, cosine similarity
- **Viz** — Plotly (interactive charts)
- **App** — Streamlit
- **API** — Spotify Web API via Spotipy
 
---
 
## How to use it
 
### Option 1 — Use the live app
Go to [playlist-analyzer.streamlit.app](https://playlist-analyzer.streamlit.app), export a playlist from [exportify.net](https://exportify.net), and upload it.
 
### Option 2 — Run locally
```bash
git clone https://github.com/npockriss/spotify-analyzer
cd spotify-analyzer
pip install -r requirements.txt
streamlit run spotify_app.py
```
 
---
 
## Exporting your playlist
 
This app uses CSVs exported from [exportify.net](https://exportify.net), which includes audio features (energy, valence, danceability, etc.) for every track. Native Spotify exports also work.
 
1. Go to [exportify.net](https://exportify.net) and log in with Spotify
2. Find your playlist and click **Export**
3. Upload the downloaded CSV to the app
 
---
 
## Pushing playlists to Spotify (optional)
 
To push auto-generated playlists to your account you need a free Spotify Developer app:
 
1. Go to [developer.spotify.com/dashboard](https://developer.spotify.com/dashboard) and log in
2. Click **Create app**, name it anything
3. Add `https://playlist-analyzer.streamlit.app/` as a Redirect URI
4. Copy your Client ID and Client Secret into the app sidebar
 
---
 
## How the ML works
 
**Clustering** — K-Means groups songs by minimizing distance to cluster centroids across 10 audio features (energy, danceability, valence, acousticness, instrumentalness, speechiness, liveness, BPM, loudness, popularity). The optimal number of clusters is selected automatically via the second derivative of the elbow curve.
 
**Dimensionality reduction** — PCA projects the 10-feature space down to 2D for visualization. The mood quadrant plots energy vs. valence directly for a more intuitive view.
 
**Recommender** — Cosine similarity measures the angle between each song's feature vector and the centroid of the top 30% most popular songs in the library. Songs closest in "direction" to that centroid score highest, regardless of absolute feature magnitudes.
 
**Playlist quality metrics:**
- *Cohesion* — average pairwise cosine similarity within the playlist (0–1)
- *Tightness* — how much less variance the playlist has vs. the full library (0–100)
- *Cluster purity* — % of songs from the single dominant K-Means cluster (0–100)
 
---
 
## Author
 
Noah Pockriss — sophomore at Vassar College, studying Math/Statistics with a Data Science minor

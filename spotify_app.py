"""
spotify_app.py — Streamlit Spotify Playlist Analyzer
-----------------------------------------------------
SETUP:
  1. pip install streamlit pandas numpy scikit-learn plotly spotipy requests
  2. streamlit run spotify_app.py
  3. Deploy free at share.streamlit.io — share the URL with anyone

SPOTIFY SETUP (each user needs their own):
  1. Go to developer.spotify.com/dashboard → Create app
  2. Set Redirect URI to your app URL + "/"  (e.g. https://yourapp.streamlit.app/)
     For local dev use: http://127.0.0.1:8501/
  3. Enter Client ID + Secret in the app sidebar
"""

import hashlib, time, warnings, base64, urllib.parse
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import spotipy

REDIRECT_URI = 'https://spotify-app-onnyqzrqbzkqxnrpkdnxlg.streamlit.app/'

warnings.filterwarnings('ignore')

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title='Playlist Analyzer',
    page_icon='🎵',
    layout='wide',
    initial_sidebar_state='expanded',
)

# ── Minimal styling ───────────────────────────────────────
st.markdown("""
<style>
    .stTabs [data-baseweb="tab"] { font-size: 0.9rem; }
    .metric-card {
        background: #1a1a2e; border-radius: 10px;
        padding: 1rem 1.25rem; margin-bottom: 0.5rem;
    }
    .big-number { font-size: 2rem; font-weight: 700; color: #1db954; }
    .label { font-size: 0.75rem; color: #aaa; text-transform: uppercase; letter-spacing: 0.1em; }
</style>
""", unsafe_allow_html=True)

FEATURES = ['Energy', 'Dance', 'Valence', 'Acoustic',
            'Instrumental', 'Speech', 'Live', 'BPM', 'Loud (Db)', 'Popularity']

COLORS = ['#f97316','#8b5cf6','#06b6d4','#ec4899','#22c55e',
          '#f59e0b','#ef4444','#14b8a6','#a855f7','#0ea5e9','#84cc16','#f43f5e']


# ════════════════════════════════════════════════════════════
#  DATA LOADING
# ════════════════════════════════════════════════════════════

@st.cache_data
def load_csv(file_bytes, filename):
    df = pd.read_csv(pd.io.common.BytesIO(file_bytes))

    df = df.rename(columns={
        'Track Name':       'Song',
        'Artist Name(s)':   'Artist',
        'Album Name':       'Album',
        'Release Date':     'Album Date',
        'Popularity':       'Popularity',
        'Danceability':     'Dance',
        'Acousticness':     'Acoustic',
        'Instrumentalness': 'Instrumental',
        'Speechiness':      'Speech',
        'Liveness':         'Live',
        'Tempo':            'BPM',
        'Loudness':         'Loud (Db)',
    })

    if 'Track URI' in df.columns:
        df['Spotify Track Id'] = df['Track URI'].str.split(':').str[-1]

    # Only convert if actually on 0-1 scale
    for col in ['Energy', 'Dance', 'Valence', 'Acoustic', 'Instrumental', 'Speech', 'Live']:
        if col in df.columns and df[col].max() <= 1.5:
            df[col] = (df[col] * 100).round(1)

    if 'Duration (ms)' in df.columns:
        df['dur_sec'] = df['Duration (ms)'] // 1000

    df = df.dropna(subset=['Song', 'Artist']).reset_index(drop=True)
    df['Added At']   = pd.to_datetime(df['Added At'],   errors='coerce')
    df['Album Date'] = pd.to_datetime(df['Album Date'], errors='coerce')

    playlist_name = filename.replace('.csv', '')
    return df, playlist_name


# ════════════════════════════════════════════════════════════
#  ML PIPELINE (all cached)
# ════════════════════════════════════════════════════════════

@st.cache_data
def build_features(df):
    df = df.dropna(subset=FEATURES).reset_index(drop=True)
    X_raw    = df[FEATURES].values
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    return df, X_raw, X_scaled, scaler


@st.cache_data
def run_auto_select_k(X_scaled, k_min=5, k_max=12):
    k_range  = list(range(k_min, min(k_max + 1, len(X_scaled))))
    inertias = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
    diffs2 = np.diff(np.diff(inertias))
    best_k = k_range[int(np.argmax(diffs2)) + 1]
    return best_k, k_range, inertias


def name_cluster(profile, all_profiles):
    avg = all_profiles.mean()
    std = all_profiles.std().replace(0, 1)
    z   = (profile - avg) / std

    e, v, d = profile['Energy'],  profile['Valence'], profile['Dance']
    ac, ins = profile['Acoustic'], profile['Instrumental']
    sp, bpm = profile['Speech'],   profile['BPM']

    # Use 80th percentile as instrumental threshold — avoids mislabeling
    ins_threshold = max(60, float(all_profiles['Instrumental'].quantile(0.80)))
    sp_threshold  = max(25, float(all_profiles['Speech'].quantile(0.80)))

    if ins > ins_threshold:                return 'Instrumental'
    if sp  > sp_threshold:                 return 'Rap & Spoken Word'
    if ac  > 60 and e < 40:               return 'Quiet Acoustic'
    if e   > 78 and bpm > 125:            return 'High Energy'
    if v   > 65 and d > 65:               return 'Feel-Good Dance'
    if v   < 35 and e < 50:               return 'Dark & Melancholic'

    traits = {
        'Energetic':  z['Energy'],
        'Mellow':    -z['Energy'],
        'Happy':      z['Valence'],
        'Brooding':  -z['Valence'],
        'Danceable':  z['Dance'],
        'Acoustic':   z['Acoustic'],
        'Driving':    z['BPM'],
        'Slow Burn': -z['BPM'],
        'Raw':        z['Speech'],
    }
    top2 = sorted(traits, key=traits.get, reverse=True)[:2]
    return f'{top2[0]} & {top2[1]}'


@st.cache_data
def run_clustering(df, X_scaled, k):
    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    df = df.copy()
    df['cluster'] = km.fit_predict(X_scaled)
    profiles = df.groupby('cluster')[FEATURES].mean()

    cluster_names, seen = {}, {}
    for c in range(k):
        name = name_cluster(profiles.loc[c], profiles)
        if name in seen:
            seen[name] += 1
            name = f'{name} {seen[name]}'
        else:
            seen[name] = 1
        cluster_names[c] = name

    df['cluster_name'] = df['cluster'].map(cluster_names)
    return df, cluster_names, profiles


@st.cache_data
def run_pca(X_scaled):
    pca    = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    return coords, pca.explained_variance_ratio_


@st.cache_data
def run_recommender(df, X_raw):
    top_songs = df[df['Popularity'] >= df['Popularity'].quantile(0.70)]
    ideal_raw = top_songs[FEATURES].mean().values.reshape(1, -1)
    sims      = cosine_similarity(X_raw, ideal_raw).flatten()
    df = df.copy()
    df['rec_score'] = (sims * 100).round(1)
    return df


# ════════════════════════════════════════════════════════════
#  PLAYLIST NAMING
# ════════════════════════════════════════════════════════════

PURPOSE_MOOD_VOCAB = {
    ('Workout', 'high', 'high', 'low'): [
        'turn this shit up', 'windows down', 'fuck it we ball',
        'friday at 5', 'bops', 'no bad days',
        'we going out', 'say less', 'on one',
        'all the way up', 'good energy only',
    ],
    ('Workout', 'high', 'mid', 'low'): [
        'no days off', 'locked in', 'push through',
        'something to prove', 'second wind', 'eyes on it',
        'grind mode', 'built different', 'work rate',
    ],
    ('Workout', 'high', 'low', 'low'): [
        'no mercy', 'pressure', 'storm front',
        'grit', 'relentless', 'high output',
        'dark energy', 'intensity',
    ],
    ('Hype', 'high', 'high', 'low'): [
        'certified', 'all gas', 'saturday night',
        'run the city', 'on one', 'opening credits',
        'heat', 'absolute heaters',
    ],
    ('Hype', 'high', 'mid', 'low'): [
        'full send', 'no chill', 'volume warning',
        'red line', 'going stupid', 'maximum',
        'unstoppable', 'peak hours', 'the gas',
    ],
    ('Hype', 'high', 'low', 'low'): [
        'city at 2am', 'midnight run', 'edge of something',
        'demons on the dashboard', 'survival mode',
        'the long way home', 'dark and fast',
    ],
    ('Deep Focus', 'low', 'low', 'high'): [
        'head down', 'background brain', 'clock out the world',
        'tunnel vision', 'deep work', 'no distractions',
        'locked in', 'concentrated',
    ],
    ('Deep Focus', 'low', 'mid', 'high'): [
        'slow work afternoon', 'quiet and productive',
        'working from the couch', 'rainy day work music',
        'soft background', 'low key focus', 'just working',
    ],
    ('Deep Focus', 'mid', 'mid', 'high'): [
        'keeping it moving', 'productive afternoon',
        'on autopilot', 'decent tuesday',
        'steady background', 'getting things done',
    ],
    ('Deep Focus', 'mid', 'low', 'high'): [
        'brooding work hours', 'heavy background',
        'moody productivity', 'dark focus',
    ],
    ('Chill', 'low', 'mid', 'high'): [
        'sunday with no plans', 'nowhere to be',
        'slow morning no alarm', 'nothing urgent',
        'no rush', 'reading by a window',
        'do not disturb', 'soft hours', 'no thoughts head empty',
    ],
    ('Chill', 'low', 'high', 'high'): [
        'slow morning', 'porch sitting',
        'end of a good day', 'morning with no alarm',
        'wind down', 'lights getting low',
        'settling in', 'finally home', 'decompressing',
    ],
    ('Chill', 'mid', 'high', 'high'): [
        'sunday morning coffee', 'unhurried',
        'light through the curtains', 'nowhere to be',
        'warm acoustic', 'easy afternoon', 'soft folk hours',
    ],
    ('Chill', 'low', 'low', 'high'): [
        'quiet and a little sad', 'overcast but okay',
        'mellow and blue', 'low key melancholy',
        'soft and kind of sad', 'background sad',
    ],
    ('Easy Company', 'mid', 'high', 'mid'): [
        'dinner table', 'cooking something good',
        'having people over', 'no agenda',
        'neighbourhood walk', 'good week energy',
        'easy like sunday', 'background happiness',
    ],
    ('Easy Company', 'mid', 'high', 'low'): [
        'dinner party', 'sunday errands',
        'pleasant afternoon', 'background happy',
        'ambient good vibes', 'the warm middle',
    ],
    ('Easy Company', 'mid', 'high', 'high'): [
        'acoustic dinner party', 'porch music',
        'warm acoustic hosting', 'acoustic and pleasant',
    ],
    ('Easy Company', 'low', 'high', 'high'): [
        'quiet good vibes', 'soft acoustic evening',
        'easy and acoustic', 'gentle background',
    ],
    ('Low Light', 'low', 'low', 'high'): [
        '2am', 'the cry playlist', 'empty apartment',
        'alone with a guitar', 'missing someone',
        'not okay rn', 'in my feelings',
        'just let it out', 'grief playlist',
        'crying in the shower music', 'bad day playlist',
    ],
    ('Low Light', 'low', 'low', 'low'): [
        '3am', 'cant sleep', 'too in my head',
        'heavy late night', 'insomnia playlist',
        'riding it out', 'overthinking again',
        'brain wont stop', 'not great rn',
    ],
    ('Low Light', 'mid', 'low', 'high'): [
        'still getting over it', 'slow damage',
        'sad guitar hours', 'acoustic heartbreak',
        'sitting with it', 'aftermath',
        'in my bag (sad)', 'healing arc', 'processing',
    ],
    ('Low Light', 'mid', 'low', 'low'): [
        'late night thoughts', 'overthinking playlist',
        'existential tuesday', 'something is off',
        'grey area', 'dissociating',
        'main character sadness', 'not vibing rn',
    ],
}

MOOD_VOCAB_FALLBACK = {
    ('high', 'high', 'low'):  ['feel-good bangers', 'dance floor energy', 'loud and happy'],
    ('high', 'mid', 'low'):   ['high energy mix', 'driving fast', 'no chill energy'],
    ('high', 'low', 'low'):   ['dark energy', 'intense and brooding', 'storm front'],
    ('mid',  'high', 'low'):  ['background feel-good', 'easy pop hours', 'pleasant background'],
    ('mid',  'high', 'high'): ['warm acoustic', 'folk and coffee', 'sunday morning acoustic'],
    ('mid',  'high', 'mid'):  ['dinner party music', 'easy background', 'good vibes cooking'],
    ('mid',  'mid',  'low'):  ['the daily commute', 'background driving', 'keeping it moving'],
    ('mid',  'mid',  'high'): ['acoustic background', 'rainy day acoustic', 'soft focus'],
    ('mid',  'low',  'low'):  ['late night thoughts', 'moody background', 'introspective'],
    ('mid',  'low',  'high'): ['acoustic sad hours', 'quiet heartbreak', 'moody acoustic'],
    ('low',  'high', 'low'):  ['quiet feel-good', 'soft happy background', 'barely there bliss'],
    ('low',  'high', 'high'): ['acoustic wind-down', 'gentle acoustic evening', 'wind down'],
    ('low',  'mid',  'high'): ['acoustic chill hours', 'slow acoustic afternoon', 'bedroom acoustic'],
    ('low',  'low',  'low'):  ['late night dark', '3am mix', 'heavy midnight'],
    ('low',  'low',  'high'): ['late night acoustic sad', 'sad guitar playlist', 'quiet sad acoustic'],
}


def classify_feature(val, low_thresh=45, high_thresh=60):
    if val < low_thresh:  return 'low'
    if val > high_thresh: return 'high'
    return 'mid'


def top_genres(df, mask, n=10):
    genre_col = next((c for c in df.columns if 'genre' in c.lower()), None)
    if not genre_col:
        return []
    series = df[mask][genre_col].dropna()
    if series.empty:
        return []
    all_genres = []
    for g in series:
        all_genres.extend([x.strip().lower() for x in str(g).split(',') if x.strip()])
    return pd.Series(all_genres).value_counts().head(n).index.tolist() if all_genres else []


def playlist_name_and_desc(base_name, df, mask, source_name, used_names=None):
    if used_names is None:
        used_names = set()

    subset  = df[mask]
    n_songs = len(subset)
    genres  = top_genres(df, mask)

    avg_e   = subset['Energy'].mean()
    avg_v   = subset['Valence'].mean()
    avg_ac  = subset['Acoustic'].mean()
    avg_bpm = subset['BPM'].mean()

    e_level  = classify_feature(avg_e)
    v_level  = classify_feature(avg_v)
    ac_level = classify_feature(avg_ac, low_thresh=30, high_thresh=45)

    specific_key = (base_name, e_level, v_level, ac_level)
    feature_key  = (e_level, v_level, ac_level)
    seed = int(hashlib.md5(f'{source_name}{base_name}'.encode()).hexdigest(), 16)

    if specific_key in PURPOSE_MOOD_VOCAB:
        opts = PURPOSE_MOOD_VOCAB[specific_key]
    elif feature_key in MOOD_VOCAB_FALLBACK:
        opts = MOOD_VOCAB_FALLBACK[feature_key]
    else:
        opts = [base_name.lower()]

    full_name = None
    for attempt in range(max(len(opts), 20)):
        candidate = f'{opts[(seed + attempt) % len(opts)]} (from {source_name})'
        if candidate not in used_names:
            full_name = candidate
            used_names.add(candidate)
            break

    if not full_name:
        full_name = f'{base_name} (from {source_name})'
        used_names.add(full_name)

    feature_parts = [f'avg {avg_bpm:.0f} BPM']
    if avg_ac > 50:  feature_parts.append('acoustic-leaning')
    if avg_v  > 65:  feature_parts.append('high valence')
    elif avg_v < 35: feature_parts.append('low valence')
    if avg_e  > 65:  feature_parts.append('high energy')
    elif avg_e < 40: feature_parts.append('low energy')

    genre_str = ', '.join(genres[:3]) if genres else 'mixed genres'
    desc = f'{genre_str} · {" · ".join(feature_parts)} · {n_songs} songs · Generated from {source_name}'

    return full_name, desc


def build_playlist_definitions(df):
    e75     = df['Energy'].quantile(0.75)
    e50     = df['Energy'].median()
    e25     = df['Energy'].quantile(0.25)
    v35     = df['Valence'].quantile(0.35)
    v65     = df['Valence'].quantile(0.65)
    d65     = df['Dance'].quantile(0.65)
    ac50    = df['Acoustic'].quantile(0.50)
    bpm_med = df['BPM'].median()
    return [
        {'base_name': 'Workout',
         'mask': (df['Energy'] > e75) & (df['Dance'] > d65) & (df['Instrumental'] < 30)},
        {'base_name': 'Deep Focus',
         'mask': (df['Instrumental'] > 35) |
                 ((df['Acoustic'] > ac50) & (df['Energy'] < e50) & (df['Speech'] < 10))},
        {'base_name': 'Chill',
         'mask': (df['Energy'] < e50) & df['Valence'].between(v35, v65) & (df['Acoustic'] > ac50)},
        {'base_name': 'Hype',
         'mask': (df['Energy'] > e75) & (df['BPM'] > bpm_med)},
        {'base_name': 'Low Light',
         'mask': (df['Valence'] < v35) & (df['Energy'] < e50)},
        {'base_name': 'Easy Company',
         'mask': (df['Valence'] > v65) & df['Energy'].between(e25, e75) & (df['Speech'] < 15)},
    ]


def evaluate_playlists(df, X_raw, X_scaled, playlist_defs):
    overall_std = pd.DataFrame(X_scaled, columns=FEATURES).std()
    results = []
    for pl in playlist_defs:
        idx = np.where(pl['mask'])[0]
        if len(idx) < 2:
            results.append({'Playlist': pl['full_name'], 'Songs': len(idx),
                            'Cohesion': None, 'Tightness': None, 'Cluster Purity': None})
            continue
        subset_raw    = X_raw[idx]
        subset_scaled = X_scaled[idx]
        sim_matrix    = cosine_similarity(subset_raw)
        n             = len(idx)
        upper         = sim_matrix[np.triu_indices(n, k=1)]
        cohesion      = round(float(upper.mean()), 3)
        subset_std    = pd.DataFrame(subset_scaled, columns=FEATURES).std()
        relative_var  = (subset_std / overall_std).mean()
        tightness     = round((1 - relative_var) * 100, 1)
        purity = None
        if 'cluster' in df.columns:
            counts = df.iloc[idx]['cluster'].value_counts()
            purity = round(counts.iloc[0] / counts.sum() * 100, 1)
        results.append({'Playlist': pl['full_name'], 'Songs': n,
                        'Cohesion': cohesion, 'Tightness': tightness,
                        'Cluster Purity': purity})
    return pd.DataFrame(results).sort_values('Cohesion', ascending=False).reset_index(drop=True)


# ════════════════════════════════════════════════════════════
#  SPOTIFY OAUTH (browser-based, no local server needed)
# ════════════════════════════════════════════════════════════

def get_auth_url(client_id, redirect_uri):
    params = {
        'client_id':     client_id,
        'response_type': 'code',
        'redirect_uri':  redirect_uri,
        'scope':         'playlist-modify-private playlist-modify-public',
    }
    return 'https://accounts.spotify.com/authorize?' + urllib.parse.urlencode(params)


def exchange_code(code, client_id, client_secret, redirect_uri):
    auth    = base64.b64encode(f'{client_id}:{client_secret}'.encode()).decode()
    resp    = requests.post('https://accounts.spotify.com/api/token',
                headers={'Authorization': f'Basic {auth}'},
                data={'grant_type': 'authorization_code',
                      'code': code, 'redirect_uri': redirect_uri})
    return resp.json()


def push_playlist_web(access_token, full_name, description, track_ids):
    sp  = spotipy.Spotify(auth=access_token)
    pl  = sp._post('me/playlists', payload={
        'name': full_name, 'public': False, 'description': description
    })
    pid = pl['id']
    url = pl['external_urls']['spotify']
    uris = [f'spotify:track:{tid}' for tid in track_ids if pd.notna(tid) and tid]
    for i in range(0, len(uris), 100):
        sp.playlist_add_items(pid, uris[i:i+100])
        time.sleep(0.3)
    return url


# ════════════════════════════════════════════════════════════
#  CHARTS (Plotly)
# ════════════════════════════════════════════════════════════

def plot_elbow(k_range, inertias, best_k):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(k_range), y=inertias, mode='lines+markers',
                             line=dict(color='#8b5cf6', width=2), marker=dict(size=7)))
    fig.add_vline(x=best_k, line_dash='dash', line_color='#ef4444',
                  annotation_text=f'k={best_k}', annotation_position='top right')
    fig.update_layout(title='Elbow Method', xaxis_title='k', yaxis_title='Inertia',
                      height=300, margin=dict(t=40, b=30))
    return fig


def plot_pca_scatter(df, coords, ev, k, cluster_names):
    df2 = df.copy()
    df2['px'] = coords[:, 0]
    df2['py'] = coords[:, 1]
    df2['color'] = df2['cluster'].map(lambda c: COLORS[c % len(COLORS)])

    fig = go.Figure()
    for c in range(k):
        mask = df2['cluster'] == c
        sub  = df2[mask]
        fig.add_trace(go.Scatter(
            x=sub['px'], y=sub['py'], mode='markers',
            name=f'{cluster_names[c]} ({mask.sum()})',
            marker=dict(color=COLORS[c % len(COLORS)], size=5, opacity=0.6),
            text=sub['Song'] + ' — ' + sub['Artist'],
            hovertemplate='%{text}<extra></extra>',
        ))
    fig.update_layout(
        title=f'Cluster Map (PC1={ev[0]:.1%}, PC2={ev[1]:.1%})',
        xaxis_title=f'PC1 ({ev[0]:.1%})', yaxis_title=f'PC2 ({ev[1]:.1%})',
        height=500, legend=dict(font=dict(size=10)),
    )
    return fig


def plot_cluster_bars(profiles, k, cluster_names):
    display = ['Energy', 'Dance', 'Valence', 'Acoustic', 'Instrumental', 'Speech']
    cols    = min(k, 3)
    rows    = (k + cols - 1) // cols

    from plotly.subplots import make_subplots
    fig = make_subplots(rows=rows, cols=cols,
                        subplot_titles=[cluster_names[c] for c in range(k)])
    for c in range(k):
        r, col = divmod(c, cols)
        vals   = profiles.loc[c, display].values
        fig.add_trace(
            go.Bar(x=display, y=vals, marker_color=COLORS[c % len(COLORS)],
                   showlegend=False),
            row=r+1, col=col+1
        )
        fig.update_yaxes(range=[0, 100], row=r+1, col=col+1)
    fig.update_layout(height=250 * rows, title_text='Cluster Feature Profiles')
    return fig


def plot_time_trends(df):
    if df['Added At'].isna().all():
        return None
    df2 = df.copy()
    df2['added_month'] = df2['Added At'].dt.to_period('M')
    monthly = (df2.groupby('added_month')
                  .agg(count=('Song','count'),
                       avg_energy=('Energy','mean'),
                       avg_valence=('Valence','mean'),
                       avg_dance=('Dance','mean'),
                       avg_acoustic=('Acoustic','mean'))
                  .round(2).reset_index())
    monthly = monthly[monthly['count'] <= monthly['count'].median() * 3]
    if len(monthly) < 3:
        return None
    labels = monthly['added_month'].astype(str)
    fig = go.Figure()
    for col, color, name in [
        ('avg_energy',   '#f97316', 'Energy'),
        ('avg_valence',  '#06b6d4', 'Valence'),
        ('avg_acoustic', '#8b5cf6', 'Acoustic'),
        ('avg_dance',    '#22c55e', 'Dance'),
    ]:
        fig.add_trace(go.Scatter(x=labels, y=monthly[col], mode='lines+markers',
                                 name=name, line=dict(color=color, width=2),
                                 marker=dict(size=5)))
    fig.update_layout(title='Taste Evolution Over Time', yaxis=dict(range=[0, 100]),
                      height=400, xaxis_tickangle=-45)
    return fig


def plot_rec_scores(df, n=20):
    top = df.nlargest(n, 'rec_score')[['Song','Artist','rec_score','Energy','Valence','Dance']].copy()
    top['label'] = top['Song'] + ' — ' + top['Artist']
    fig = go.Figure(go.Bar(
        x=top['rec_score'], y=top['label'], orientation='h',
        marker_color='#1db954',
        text=top['rec_score'].astype(str),
        textposition='outside',
    ))
    fig.update_layout(title=f'Top {n} Best-Fit Songs', height=600,
                      xaxis_title='Score', yaxis=dict(autorange='reversed'))
    return fig


# ════════════════════════════════════════════════════════════
#  APP
# ════════════════════════════════════════════════════════════


query = st.query_params
if 'code' in query and 'spotify_token' not in st.session_state:
    cid  = st.session_state.get('sp_client_id', '')
    csec = st.session_state.get('sp_client_secret', '')
    if cid and csec:
        token_data = exchange_code(query['code'], cid, csec, REDIRECT_URI)
        if 'access_token' in token_data:
            st.session_state['spotify_token'] = token_data['access_token']
            st.query_params.clear()
            st.rerun()


# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.title('🎵 Playlist Analyzer')
    st.caption('Export any playlist from exportify.net, upload it here.')

    uploaded = st.file_uploader('Upload playlist CSV', type='csv')

    st.divider()
    st.subheader('Spotify Connection')
    st.caption('Only needed to push playlists to Spotify.')

    client_id     = st.text_input('Client ID',     type='password')
    client_secret = st.text_input('Client Secret', type='password')

    REDIRECT_URI = 'https://spotify-app-onnyqzrqbzkqxnrpkdnxlg.streamlit.app/'

    if client_id:
        st.session_state['sp_client_id']     = client_id
        st.session_state['sp_client_secret'] = client_secret

    if 'spotify_token' not in st.session_state:
        if client_id and client_secret:

            auth_url = get_auth_url(client_id, REDIRECT_URI)
            st.markdown(f'**Step 1:** [Click here to log in to Spotify]({auth_url})')
            st.caption(
                'After clicking, Spotify will redirect you to a page that '
                'looks broken or blank. That is normal. '
                'Copy the full URL from your browser address bar and '
                'paste it in the box below.'
            )

            pasted = st.text_input('Step 2: Paste the redirect URL here')
            if pasted and '?code=' in pasted:
                try:
                    code = urllib.parse.parse_qs(
                        urllib.parse.urlparse(pasted).query
                    )['code'][0]
                    token_data = exchange_code(
                        code, client_id, client_secret, REDIRECT_URI
                    )
                    if 'access_token' in token_data:
                        st.session_state['spotify_token'] = token_data['access_token']
                        st.success('Connected!')
                        st.rerun()
                    else:
                        st.error('Auth failed — double check your Client ID and Secret.')
                except Exception as e:
                    st.error(f'Something went wrong: {e}')
    else:
        st.success('✓ Spotify connected')
        if st.button('Disconnect'):
            del st.session_state['spotify_token']
            st.rerun()
    

# ── Main area ─────────────────────────────────────────────
if not uploaded:
    st.title('Spotify Playlist Analyzer')
    st.markdown("""
    **Upload a playlist CSV to get started.**

    #### How to export your playlist:
    1. Go to **[exportify.net](https://exportify.net)** and log in with Spotify
    2. Find your playlist and click **Export**
    3. Upload the downloaded CSV here

    #### What you'll get:
    - **Cluster map** — see the natural vibes in your library
    - **Taste timeline** — how your music taste shifted over time
    - **Best-fit songs** — which songs match your core taste most
    - **Auto-playlists** — 6 mood playlists pushed straight to Spotify
    """)
    st.stop()

# ── Run analysis ──────────────────────────────────────────
file_bytes    = uploaded.read()
df_raw, pname = load_csv(file_bytes, uploaded.name)
df, X_raw, X_scaled, scaler = build_features(df_raw)

with st.spinner('Finding clusters...'):
    best_k, k_range, inertias = run_auto_select_k(X_scaled)
    df, cluster_names, profiles = run_clustering(df, X_scaled, best_k)

coords, ev = run_pca(X_scaled)
df         = run_recommender(df, X_raw)

playlist_defs = build_playlist_definitions(df)
used_names    = set()
for pl in playlist_defs:
    pl['full_name'], pl['desc'] = playlist_name_and_desc(
        pl['base_name'], df, pl['mask'], pname, used_names
    )
scores = evaluate_playlists(df, X_raw, X_scaled, playlist_defs)

# ── Header stats ──────────────────────────────────────────
st.title(f'🎵 {pname}')
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric('Songs',        f'{len(df):,}')
c2.metric('Clusters',     best_k)
c3.metric('Avg Energy',   f'{df["Energy"].mean():.0f}')
c4.metric('Avg Valence',  f'{df["Valence"].mean():.0f}')
c5.metric('Avg BPM',      f'{df["BPM"].mean():.0f}')

# ── Tabs ──────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ['🔮 Clusters', '📈 Taste Over Time', '⭐ Recommender', '🎛️ Playlists', '📊 Raw Data']
)

with tab1:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.plotly_chart(plot_pca_scatter(df, coords, ev, best_k, cluster_names),
                        use_container_width=True)
    with col2:
        st.plotly_chart(plot_elbow(k_range, inertias, best_k),
                        use_container_width=True)
        st.caption(f'Auto-selected **k={best_k}** via second derivative of elbow curve.')

    st.plotly_chart(plot_cluster_bars(profiles, best_k, cluster_names),
                    use_container_width=True)

    st.subheader('Cluster breakdown')
    for c in range(best_k):
        n    = (df['cluster'] == c).sum()
        p    = profiles.loc[c]
        with st.expander(f'{cluster_names[c]} — {n} songs'):
            cc = st.columns(6)
            for i, feat in enumerate(['Energy','Dance','Valence','Acoustic','Instrumental','Speech']):
                cc[i].metric(feat, f'{p[feat]:.0f}')
            top5 = df[df['cluster'] == c].nlargest(5, 'Popularity')[['Song','Artist','Popularity']]
            st.dataframe(top5, hide_index=True, use_container_width=True)

with tab2:
    fig = plot_time_trends(df)
    if fig:
        st.plotly_chart(fig, use_container_width=True)

        df['added_month'] = df['Added At'].dt.to_period('M')
        monthly = (df.groupby('added_month')
                     .agg(count=('Song','count'),
                          avg_energy=('Energy','mean'),
                          avg_valence=('Valence','mean'),
                          avg_acoustic=('Acoustic','mean'))
                     .round(1).reset_index())
        monthly = monthly[monthly['count'] <= monthly['count'].median() * 3]
        monthly['added_month'] = monthly['added_month'].astype(str)

        first3 = monthly.head(3)
        last3  = monthly.tail(3)
        de = last3['avg_energy'].mean()  - first3['avg_energy'].mean()
        dv = last3['avg_valence'].mean() - first3['avg_valence'].mean()
        da = last3['avg_acoustic'].mean()- first3['avg_acoustic'].mean()

        st.markdown(f"""
        **Your taste shift:**
        Energy {'↑' if de>0 else '↓'} {abs(de):.1f} pts ·
        Valence {'↑' if dv>0 else '↓'} {abs(dv):.1f} pts ·
        Acoustic {'↑' if da>0 else '↓'} {abs(da):.1f} pts
        """)
    else:
        st.info('Not enough monthly data to show a trend (need 3+ months).')

with tab3:
    st.plotly_chart(plot_rec_scores(df), use_container_width=True)

    st.subheader('Biggest outliers')
    st.caption('Songs least like your core taste')
    out = df.nsmallest(10, 'rec_score')[['Song','Artist','rec_score','Energy','Valence','Acoustic','cluster_name']]
    st.dataframe(out, hide_index=True, use_container_width=True)

with tab4:
    st.subheader('Generated playlists')

    # Show scores table
    st.dataframe(scores, hide_index=True, use_container_width=True)
    st.caption('Cohesion: 0–1 (higher = songs sound more alike) · Tightness: 0–100 · Cluster Purity: 0–100')

    st.divider()
    st.subheader('Push to Spotify')

    if 'spotify_token' not in st.session_state:
        st.info('Connect Spotify in the sidebar to push playlists.')
    else:
        st.caption('Select which playlists to push:')

        push_col1, push_col2 = st.columns([3, 1])
        with push_col1:
            selected_names = st.multiselect(
                'Playlists to push',
                options=[pl['full_name'] for pl in playlist_defs],
                default=[pl['full_name'] for pl in playlist_defs],
                label_visibility='collapsed',
            )
        with push_col2:
            push_btn = st.button('🚀 Push Selected', type='primary', use_container_width=True)

        if push_btn and selected_names:
            selected_defs = [pl for pl in playlist_defs if pl['full_name'] in selected_names]
            progress = st.progress(0)
            results  = []
            for i, pl in enumerate(selected_defs):
                track_ids = df[pl['mask']]['Spotify Track Id'].tolist()
                if not track_ids:
                    st.warning(f'No songs matched for {pl["full_name"]} — skipping.')
                    continue
                with st.spinner(f'Creating {pl["full_name"]}...'):
                    try:
                        url = push_playlist_web(
                            st.session_state['spotify_token'],
                            pl['full_name'], pl['desc'], track_ids
                        )
                        results.append({'name': pl['full_name'], 'songs': len(track_ids), 'url': url})
                        progress.progress((i + 1) / len(selected_defs))
                    except Exception as e:
                        st.error(f'Error pushing {pl["full_name"]}: {e}')

            if results:
                st.success(f'✅ {len(results)} playlists created!')
                for r in results:
                    st.markdown(f'**[{r["name"]}]({r["url"]})** — {r["songs"]} songs')

        # Per-playlist preview
        st.divider()
        for pl in playlist_defs:
            n = int(pl['mask'].sum())
            with st.expander(f'{pl["full_name"]} — {n} songs'):
                row = scores[scores['Playlist'] == pl['full_name']]
                if not row.empty:
                    mc = st.columns(3)
                    mc[0].metric('Cohesion',       f'{row["Cohesion"].values[0]:.3f}')
                    mc[1].metric('Tightness',       f'{row["Tightness"].values[0]:.1f}')
                    mc[2].metric('Cluster Purity',  f'{row["Cluster Purity"].values[0]:.1f}%'
                                 if row["Cluster Purity"].values[0] else 'n/a')
                sample = df[pl['mask']].nlargest(10, 'Popularity')[['Song','Artist','Energy','Valence','Dance']]
                st.dataframe(sample, hide_index=True, use_container_width=True)

with tab5:
    st.subheader('Full dataset')
    display_cols = ['Song','Artist','cluster_name','rec_score',
                    'Energy','Valence','Dance','Acoustic','BPM','Popularity']
    available = [c for c in display_cols if c in df.columns]
    st.dataframe(df[available].sort_values('rec_score', ascending=False),
                 hide_index=True, use_container_width=True)
    csv_out = df[available].to_csv(index=False)
    st.download_button('Download with cluster labels', csv_out,
                       file_name=f'{pname}_analyzed.csv', mime='text/csv')

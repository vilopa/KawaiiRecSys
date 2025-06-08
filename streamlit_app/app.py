import streamlit as st
import pandas as pd
import sys
import os
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
from collections import Counter
import json
from datetime import datetime

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import from our new modular structure
from src.hybrid import hybrid_recommend, profiled_hybrid_recommend
from utils.helpers import (
    get_anime_image,
    genre_to_color,
    load_anime_data,
    get_random_quote
)

# Streamlit page setup
st.set_page_config(
    page_title="KawaiiRecSys",
    page_icon="🎌",
    layout="wide"
)

# Persistent Storage Functions
def get_user_data_path():
    """Get the path for user data storage"""
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "user_data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    return os.path.join(data_dir, "kawaii_user_data.json")

def save_user_data():
    """Save user data to persistent storage"""
    try:
        data = {
            'watchlist': list(st.session_state.watchlist),
            'favorites': list(st.session_state.favorites),
            'user_ratings': st.session_state.user_ratings,
            'viewing_status': st.session_state.viewing_status,
            'last_saved': datetime.now().isoformat()
        }
        
        with open(get_user_data_path(), 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        st.error(f"Failed to save data: {e}")
        return False

def load_user_data():
    """Load user data from persistent storage"""
    try:
        data_path = get_user_data_path()
        if os.path.exists(data_path):
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Restore data to session state
            st.session_state.watchlist = set(data.get('watchlist', []))
            st.session_state.favorites = set(data.get('favorites', []))
            st.session_state.user_ratings = data.get('user_ratings', {})
            st.session_state.viewing_status = data.get('viewing_status', {})
            
            return True, data.get('last_saved', 'Unknown')
    except Exception as e:
        st.error(f"Failed to load data: {e}")
    
    return False, None

def create_backup():
    """Create a timestamped backup of user data"""
    try:
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "user_data", "backups")
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(data_dir, f"kawaii_backup_{timestamp}.json")
        
        data = {
            'watchlist': list(st.session_state.watchlist),
            'favorites': list(st.session_state.favorites),
            'user_ratings': st.session_state.user_ratings,
            'viewing_status': st.session_state.viewing_status,
            'backup_created': datetime.now().isoformat()
        }
        
        with open(backup_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return backup_path
    except Exception as e:
        st.error(f"Failed to create backup: {e}")
        return None

# Initialize ALL session state variables FIRST (before any widgets)
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None
if 'explanations' not in st.session_state:
    st.session_state.explanations = None
if 'recommending' not in st.session_state:
    st.session_state.recommending = False
if 'profiling' not in st.session_state:
    st.session_state.profiling = False
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = set()
if 'favorites' not in st.session_state:
    st.session_state.favorites = set()
if 'user_ratings' not in st.session_state:
    st.session_state.user_ratings = {}
if 'viewing_status' not in st.session_state:
    st.session_state.viewing_status = {}  # 'watching', 'completed', 'plan_to_watch'
if 'search_query' not in st.session_state:
    st.session_state.search_query = ""
if 'active_filters' not in st.session_state:
    st.session_state.active_filters = {}
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

# Auto-load user data on first run
if not st.session_state.data_loaded:
    loaded, last_saved = load_user_data()
    if loaded and last_saved:
        st.success(f"📚 Welcome back! Your data from {last_saved[:19]} has been restored.")
    st.session_state.data_loaded = True

# Enhanced Netflix-style CSS
st.markdown("""
    <style>
        html, body {
            background-color: #0e0e10;
            color: white;
        }
        .input-card, .output-card {
            display: none;
        }
        .block-container {
            padding: 2px !important;
            background-color: #0e0e10;
            margin: 0 !important;
        }
        h1, h2, h3 {
            font-family: 'Helvetica Neue', sans-serif;
            font-weight: 600;
            margin-top: 0rem;
            margin-bottom: 0rem;
        }
        .stButton button {
            background: linear-gradient(145deg, #4ecdc4, #26a69a);
            color: white;
            border-radius: 5px;
            border: none;
            font-weight: bold;
            padding: 6px 3px;
            transition: all 0.3s ease;
            font-size: 0.95em;
            width: 100%;
            margin-top: 11px;
            height: 36px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .stButton button:hover {
            transform: scale(1.02);
            box-shadow: 0 0 5px rgba(78, 205, 196, 0.5);
        }
        .stCheckbox {
            margin: 11px;
        }
        .stSlider > div[data-baseweb="slider"] > div {
            background: linear-gradient(to right, #4ecdc4, #26a69a);
        }
        .stNumberInput>div>div>input, .stTextInput>div>div>input {
            background-color: #1c1c1e;
            color: white;
            border-radius: 5px;
            border: 1px solid #4ecdc4;
            padding: 3px;
            font-size: 0.9em;
        }
        .quote-footer {
            margin-top: 5px;
            color: #aaaaaa;
            font-style: italic;
            text-align: center;
            padding: 5px;
            border-top: 1px solid rgba(255, 75, 175, 0.2);
        }
        .stMultiSelect > div {
            background-color: #1c1c1e;
            color: white;
            border-radius: 5px;
            border: 1px solid #4ecdc4;
            padding: 5px;
        }
        .stMultiSelect > div > div > div {
            color: white;
        }
        .recommendation-title {
            color: #4ecdc4;
            margin: 0;
            padding: 0;
            font-size: 1.2em;
            text-align: center;
        }
        .anime-card {
            background: linear-gradient(135deg, rgba(28, 28, 30, 0.95), rgba(44, 44, 46, 0.95));
            backdrop-filter: blur(10px);
            border-radius: 10px;
            padding: 8px;
            box-shadow: 0 4px 16px rgba(0,0,0,0.3), 0 0 0 1px rgba(255, 255, 255, 0.1);
            transition: all 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94);
            overflow: hidden;
            text-align: center;
            color: white;
            margin: 3px;
            border: 1px solid rgba(78, 205, 196, 0.3);
            position: relative;
        }
        .anime-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(135deg, rgba(78, 205, 196, 0.1), rgba(38, 166, 154, 0.1));
            opacity: 0;
            transition: opacity 0.3s ease;
            border-radius: 15px;
            z-index: 1;
        }
        .anime-card:hover::before {
            opacity: 1;
        }
        .anime-card:hover {
            transform: translateY(-8px) scale(1.02);
            box-shadow: 0 16px 48px rgba(78, 205, 196, 0.2), 0 0 0 1px rgba(78, 205, 196, 0.4);
            border-color: #4ecdc4;
        }
        .anime-card * {
            position: relative;
            z-index: 2;
        }
        .anime-image {
            border-radius: 8px;
            width: 100%;
            height: 160px;
            object-fit: cover;
            margin-bottom: 6px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
            transition: all 0.3s ease;
        }
        .anime-card:hover .anime-image {
            transform: scale(1.05);
            box-shadow: 0 8px 24px rgba(0,0,0,0.4);
        }
        .anime-title {
            margin: 4px 0;
            font-size: 1em;
            font-weight: 700;
            color: #ffffff;
            text-shadow: 0 1px 2px rgba(0,0,0,0.5);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            letter-spacing: 0.3px;
        }
        .anime-score {
            font-size: 0.9em;
            font-weight: 600;
            background: linear-gradient(135deg, #4ecdc4, #80cbc4);
            -webkit-background-clip: text;
            background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: none;
            margin: 3px 0;
        }
        .genre-tag {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 15px;
            font-size: 0.7em;
            font-weight: 500;
            margin: 2px;
            background: linear-gradient(135deg, rgba(78, 205, 196, 0.2), rgba(38, 166, 154, 0.2));
            color: #ffffff;
            border: 1px solid rgba(78, 205, 196, 0.3);
            backdrop-filter: blur(5px);
            text-shadow: 0 1px 2px rgba(0,0,0,0.3);
            transition: all 0.2s ease;
        }
        .genre-tag:hover {
            background: linear-gradient(135deg, rgba(78, 205, 196, 0.4), rgba(38, 166, 154, 0.4));
            transform: translateY(-1px);
        }
        .input-card {
            background: linear-gradient(135deg, rgba(28, 28, 30, 0.95), rgba(44, 44, 46, 0.95));
            backdrop-filter: blur(15px);
            border-radius: 10px;
            padding: 6px;
            margin-top: 0px;
            margin-bottom: 3px;
            margin-right: 5px;
            border: 1px solid rgba(66, 133, 244, 0.3);
            box-shadow: 0 4px 16px rgba(0,0,0,0.3), 0 0 0 1px rgba(255, 255, 255, 0.05);
            height: 100%;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
        }
        .input-card:hover {
            border-color: rgba(66, 133, 244, 0.5);
            box-shadow: 0 16px 48px rgba(66, 133, 244, 0.1), 0 0 0 1px rgba(255, 255, 255, 0.1);
        }
        .output-card {
            background: linear-gradient(135deg, rgba(28, 28, 30, 0.95), rgba(44, 44, 46, 0.95));
            backdrop-filter: blur(15px);
            border-radius: 10px;
            padding: 6px;
            margin: 0;
            margin-left: 5px;
            border: 1px solid rgba(66, 133, 244, 0.3);
            box-shadow: 0 4px 16px rgba(0,0,0,0.3), 0 0 0 1px rgba(255, 255, 255, 0.05);
            height: 100%;
            min-height: 300px;
            transition: all 0.3s ease;
        }
        .output-card:hover {
            border-color: rgba(66, 133, 244, 0.5);
            box-shadow: 0 16px 48px rgba(66, 133, 244, 0.1), 0 0 0 1px rgba(255, 255, 255, 0.1);
        }
        .loading-text {
            text-align: center;
            background: linear-gradient(135deg, #4ecdc4, #80cbc4, #26a69a);
            -webkit-background-clip: text;
            background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 1.3em;
            font-weight: 600;
            margin: 15px 0;
            animation: pulse 1.5s infinite, gradient-flow 3s ease-in-out infinite;
        }
        @keyframes gradient-flow {
            0%, 100% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
        }
        .feedback-buttons {
            display: flex;
            justify-content: space-around;
            margin-top: 6px;
            gap: 6px;
        }
        .like-button, .dislike-button {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            text-align: center;
            background: linear-gradient(135deg, rgba(255, 75, 175, 0.15), rgba(138, 43, 226, 0.15));
            backdrop-filter: blur(5px);
            border: 1px solid rgba(255, 75, 175, 0.3);
            transition: all 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94);
            min-width: 50px;
            color: white;
            text-shadow: 0 1px 2px rgba(0,0,0,0.3);
        }
        .like-button:hover {
            background: linear-gradient(135deg, rgba(0, 255, 100, 0.3), rgba(0, 200, 80, 0.3));
            border-color: rgba(0, 255, 100, 0.5);
            transform: translateY(-2px) scale(1.05);
            box-shadow: 0 4px 16px rgba(0, 255, 100, 0.2);
        }
        .dislike-button:hover {
            background: linear-gradient(135deg, rgba(255, 60, 60, 0.3), rgba(220, 20, 60, 0.3));
            border-color: rgba(255, 60, 60, 0.5);
            transform: translateY(-2px) scale(1.05);
            box-shadow: 0 4px 16px rgba(255, 60, 60, 0.2);
        }
        @keyframes pulse {
            0% { opacity: 0.6; }
            50% { opacity: 1; }
            100% { opacity: 0.6; }
        }
        .vertical-chart {
            margin-top: 5px;
        }
        
        /* Compact the native Streamlit components */
        div.row-widget.stRadio > div {
            flex-direction: row;
            align-items: center;
        }
        div.row-widget.stRadio > div[role="radiogroup"] > label {
            margin: 0 5px 0 0;
            padding: 5px;
            min-height: 0;
        }
        .streamlit-expanderHeader {
            padding-top: 5px !important;
            padding-bottom: 5px !important;
        }
        
        /* Remove extra padding from containers */
        .element-container {
            margin-top: 5px;
            margin-bottom: 5px;
        }
        
        /* Compact Streamlit select box */
        .stSelectbox div[data-baseweb="select"] {
            padding-top: 5px;
            padding-bottom: 5px;
        }
        
        /* Reduce multiselect padding */
        .stMultiSelect div[data-baseweb="select"] span {
            padding-top: 5px;
            padding-bottom: 5px;
        }
        
        /* Reduce caption spacing */
        .caption {
            margin-top: 0;
            padding-top: 0;
            font-size: 0.75rem;
            padding: 5px;
        }
        
        /* Chart compact style */
        .vertical-chart .stMarkdown p {
            margin-bottom: 5px;
        }
        
        /* Reduce input spacing */
        .stTextInput, .stNumberInput, .stSelectbox, .stMultiSelect {
            margin-bottom: 5px;
            padding: 5px;
        }
        
        /* Reduce header spacing */
        .section-title {
            margin-bottom: 5px;
            padding: 5px;
        }
        
        .compact-h3 {
            margin-top: 5px;
            margin-bottom: 5px;
            font-size: 1.1rem;
            padding: 5px;
        }
        
        /* Reduce slider padding */
        .stSlider {
            padding-top: 5px;
            padding-bottom: 5px;
        }
        
        /* Reduce info block padding */
        .stAlert {
            padding: 5px;
        }
        div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"] {
            gap: 5px;
        }
        
        /* Additional padding adjustments */
        .stMarkdown {
            padding: 2px;
            margin: 0;
        }
        
        .stForm {
            padding: 2px;
            margin: 0;
        }
        
        .st-emotion-cache-1gulkj5 {
            padding: 2px;
            margin: 0;
        }
        
        .st-emotion-cache-16txtl3 {
            padding: 2px;
            margin: 0;
        }
        
        .st-emotion-cache-4z1n4l {
            padding: 2px;
            margin: 0;
        }
        
        div[data-testid="stVerticalBlock"] {
            gap: 2px;
            padding: 2px;
            margin: 0;
        }
        
        /* Remove all default Streamlit spacing */
        .element-container {
            margin: 0 !important;
            padding: 0 !important;
        }
        
        .stVerticalBlock > div {
            margin: 0 !important;
            padding: 0 !important;
        }
    </style>
    """, unsafe_allow_html=True)

# Streamlit cache for data loading
@st.cache_data
def cached_load_anime_data():
    return load_anime_data()

# Streamlit cache for recommendations
@st.cache_data
def get_recommendations(user_id, selected_anime, alpha, beta, gamma, ratings_df, anime_df, enable_profiling=False):
    if enable_profiling:
        return profiled_hybrid_recommend(
            user_id=user_id,
            selected_anime=selected_anime,
            ratings_df=ratings_df,
            anime_df=anime_df,
            top_n=5,
            alpha=alpha,
            beta=beta,
            gamma=gamma
        )
    else:
        return hybrid_recommend(
            user_id=user_id,
            selected_anime=selected_anime,
            ratings_df=ratings_df,
            anime_df=anime_df,
            top_n=5,
            alpha=alpha,
            beta=beta,
            gamma=gamma
        )

# Load anime data - Pre-load at startup to reduce delay
anime_df = cached_load_anime_data()
anime_list = anime_df['name'].tolist()

# Pre-load ratings data at startup (cached)
@st.cache_data
def load_ratings_data():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, "data/ratings.csv")
    return pd.read_csv(data_path)

# Load ratings data at startup
ratings_df = load_ratings_data()

# Function to handle feedback clicks without using nested columns
def handle_feedback(anime_name, feedback_type):
    if 'feedback' not in st.session_state:
        st.session_state.feedback = {}
    
    st.session_state.feedback[anime_name] = feedback_type
    st.toast(f"You {feedback_type}d {anime_name}!")

# Netflix-style recommendation display
def show_netflix_style_recommendations(df, explanations=None):
    if df.empty:
        st.warning("No recommendations found. Try different settings or anime selections.")
        return
        
    df["image_url"] = df["name"].apply(get_anime_image)
    
    # Create columns for each recommendation
    cols = st.columns(len(df))
    
    for idx, (col, row) in enumerate(zip(cols, df.itertuples())):
        with col:
            anime_name = row.name
            # Create a dynamic accent color based on score
            score_color = "rgba(255, 215, 0, 0.8)" if row.final_score >= 8.5 else "rgba(255, 75, 175, 0.8)" if row.final_score >= 7.5 else "rgba(138, 43, 226, 0.8)"
            
            # Display status badges
            status_badges = ""
            if anime_name in st.session_state.watchlist:
                status_badges += "📚 "
            if anime_name in st.session_state.favorites:
                status_badges += "❤️ "
            if anime_name in st.session_state.viewing_status:
                status_map = {'watching': '👀', 'completed': '✅', 'plan_to_watch': '📅'}
                status_badges += status_map.get(st.session_state.viewing_status[anime_name], '') + " "
            
            st.markdown(f"""
            <div class="anime-card">
                <div class="card-glow" style="background: {score_color}; position: absolute; top: -2px; left: -2px; right: -2px; bottom: -2px; border-radius: 17px; opacity: 0.3; z-index: 0;"></div>
                <img src="{row.image_url}" class="anime-image" alt="{anime_name}" />
                <h5 class="anime-title">{anime_name}</h5>
                <p class="anime-score">⭐ {row.final_score:.2f}</p>
                <div style="font-size: 0.8em; margin: 2px 0; height: 12px;">{status_badges}</div>
                <div class="genre-tags">
                    {''.join([f'<span class="genre-tag">{genre.strip()}</span>' for genre in row.genre.split(',')[:2]])}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Interactive action buttons
            action_col1, action_col2, action_col3 = st.columns(3)
            
            with action_col1:
                # Watchlist toggle
                watchlist_text = "➖" if anime_name in st.session_state.watchlist else "➕"
                if st.button(f"{watchlist_text}", key=f"watchlist_{idx}", help="Add/Remove from Watchlist"):
                    if anime_name in st.session_state.watchlist:
                        st.session_state.watchlist.remove(anime_name)
                        st.toast(f"Removed {anime_name} from watchlist")
                    else:
                        st.session_state.watchlist.add(anime_name)
                        st.toast(f"Added {anime_name} to watchlist")
                    save_user_data()  # Auto-save
                    st.rerun()
            
            with action_col2:
                # Favorite toggle
                fav_text = "💔" if anime_name in st.session_state.favorites else "❤️"
                if st.button(f"{fav_text}", key=f"fav_{idx}", help="Add/Remove from Favorites"):
                    if anime_name in st.session_state.favorites:
                        st.session_state.favorites.remove(anime_name)
                        st.toast(f"Removed {anime_name} from favorites")
                    else:
                        st.session_state.favorites.add(anime_name)
                        st.toast(f"Added {anime_name} to favorites")
                    save_user_data()  # Auto-save
                    st.rerun()
            
            with action_col3:
                # Status selector
                current_status = st.session_state.viewing_status.get(anime_name, "none")
                status_options = ["none", "plan_to_watch", "watching", "completed"]
                status_labels = ["📝", "📅", "👀", "✅"]
                
                status_idx = status_options.index(current_status)
                if st.button(f"{status_labels[status_idx]}", key=f"status_{idx}", help="Change viewing status"):
                    # Cycle through statuses
                    next_idx = (status_idx + 1) % len(status_options)
                    new_status = status_options[next_idx]
                    
                    if new_status == "none":
                        if anime_name in st.session_state.viewing_status:
                            del st.session_state.viewing_status[anime_name]
                    else:
                        st.session_state.viewing_status[anime_name] = new_status
                    
                    status_names = {"none": "No status", "plan_to_watch": "Plan to Watch", 
                                  "watching": "Currently Watching", "completed": "Completed"}
                    st.toast(f"{anime_name}: {status_names[new_status]}")
                    save_user_data()  # Auto-save
                    st.rerun()
            
            # Enhanced explanation with better styling
            if explanations and idx < len(explanations):
                st.markdown(f"""
                    <div class='caption' style='
                        background: linear-gradient(135deg, rgba(255, 75, 175, 0.1), rgba(138, 43, 226, 0.1));
                        border-radius: 8px;
                        padding: 8px;
                        margin-top: 8px;
                        border: 1px solid rgba(255, 75, 175, 0.2);
                        backdrop-filter: blur(5px);
                    '>
                        💡 {explanations[idx]}
                    </div>
                """, unsafe_allow_html=True)
    
    # Attractive Genre Visualization
    st.markdown("### 🎭 Genre DNA Analysis")
    
    genres = []
    for g in df['genre']:
        genres.extend([x.strip() for x in g.split(',')])
    genre_counts = Counter(genres)
    
    # Genre emoji mapping
    genre_emojis = {
        'Action': '⚔️', 'Adventure': '🗺️', 'Comedy': '😂', 'Drama': '🎭',
        'Fantasy': '🧙‍♂️', 'Horror': '👻', 'Mystery': '🔍', 'Romance': '💕',
        'Sci-Fi': '🚀', 'Thriller': '😰', 'Animation': '🎨', 'Crime': '🕵️',
        'Supernatural': '👻', 'Slice of Life': '🌸', 'Mecha': '🤖',
        'Psychological': '🧠', 'School': '🏫', 'Military': '🎖️'
    }
    
    # Prepare data for chart
    chart_data = []
    colors = ['#ff4baf', '#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57', '#ff9ff3', '#54a0ff']
    
    for idx, (genre, count) in enumerate(genre_counts.most_common()):
        emoji = genre_emojis.get(genre, '🎌')
        percentage = (count / len(df)) * 100
        chart_data.append({
            'Genre': f"{emoji} {genre}",
            'Percentage': percentage,
            'Count': count,
            'Color': colors[idx % len(colors)]
        })
    
    # Create attractive horizontal bar chart
    import plotly.express as px
    import plotly.graph_objects as go
    
    df_chart = pd.DataFrame(chart_data)
    
    fig = go.Figure()
    
    # Add bars with gradient colors
    fig.add_trace(go.Bar(
        y=df_chart['Genre'],
        x=df_chart['Percentage'],
        orientation='h',
        marker=dict(
            color=df_chart['Color'],
            line=dict(color='rgba(255, 255, 255, 0.2)', width=1)
        ),
        text=[f"{p:.0f}%" for p in df_chart['Percentage']],  # Shorter text
        textposition='inside',
        textfont=dict(color='white', size=10),  # Smaller text
        hovertemplate='<b>%{y}</b><br>Percentage: %{x:.1f}%<br>Count: %{customdata} anime<extra></extra>',
        customdata=df_chart['Count']
    ))
    
    # Customize the layout - Compact version
    fig.update_layout(
        title=dict(
            text="🧬 Genre DNA",
            font=dict(size=14, color='white'),
            x=0.5
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=min(250, len(chart_data) * 25 + 80),  # Much more compact
        margin=dict(l=10, r=10, t=35, b=10),  # Reduced margins
        xaxis=dict(
            title=None,  # Remove x-axis title to save space
            gridcolor='rgba(255, 75, 175, 0.15)',
            color='white',
            tickfont=dict(color='white', size=10),
            showticklabels=False  # Hide x-axis labels to save space
        ),
        yaxis=dict(
            color='white',
            tickfont=dict(color='white', size=10)  # Smaller font
        ),
        showlegend=False,
        bargap=0.1  # Reduce gap between bars
    )
    
    # Add subtle glow effect
    fig.update_traces(
        marker_line_width=1.5,
        selector=dict(type="bar")
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Compact summary metrics in one line
    top_genre = max(genre_counts, key=genre_counts.get)
    emoji = genre_emojis.get(top_genre, '🎌')
    diversity = len(genre_counts) / len(df) * 100
    
    st.markdown(f"""
    <div style='display: flex; justify-content: space-around; align-items: center; 
                background: linear-gradient(135deg, rgba(255, 75, 175, 0.1), rgba(138, 43, 226, 0.1));
                border-radius: 10px; padding: 8px; margin: 8px 0; border: 1px solid rgba(255, 75, 175, 0.2);'>
        <span style='color: white; font-size: 0.85em;'>🎯 <b>{len(genre_counts)}</b> genres</span>
        <span style='color: white; font-size: 0.85em;'>👑 <b>{emoji} {top_genre}</b></span>
        <span style='color: white; font-size: 0.85em;'>🌈 <b>{diversity:.0f}%</b> diversity</span>
    </div>
    """, unsafe_allow_html=True)

# Enhanced loading animation
def show_loading_animation():
    with st.spinner(""):
        st.markdown("""
        <div style="display: flex; flex-direction: column; align-items: center; margin-top: 30px;">
            <div class="enhanced-spinner">
                <div class="spinner-ring"></div>
                <div class="spinner-ring"></div>
                <div class="spinner-ring"></div>
                <div class="anime-icon">🎌</div>
            </div>
            <div class="loading-text" style="margin-top: 20px;">
                ✨ Finding the perfect anime for you... ✨
            </div>
            <div style="margin-top: 10px; font-size: 0.9em; color: rgba(255, 255, 255, 0.7);">
                Analyzing your preferences with AI magic 🤖
            </div>
            <style>
                .enhanced-spinner {
                    position: relative;
                    width: 60px;
                    height: 60px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }
                
                .spinner-ring {
                    position: absolute;
                    border: 2px solid transparent;
                    border-radius: 50%;
                    animation: spin 2s linear infinite;
                }
                
                .spinner-ring:nth-child(1) {
                    width: 60px;
                    height: 60px;
                    border-top-color: #ff4baf;
                    border-bottom-color: #ff4baf;
                    animation-duration: 1.5s;
                }
                
                .spinner-ring:nth-child(2) {
                    width: 45px;
                    height: 45px;
                    border-left-color: #ff6b6b;
                    border-right-color: #ff6b6b;
                    animation-duration: 2s;
                    animation-direction: reverse;
                }
                
                .spinner-ring:nth-child(3) {
                    width: 30px;
                    height: 30px;
                    border-top-color: #4ecdc4;
                    border-bottom-color: #4ecdc4;
                    animation-duration: 1s;
                }
                
                .anime-icon {
                    font-size: 1.2em;
                    animation: pulse 2s ease-in-out infinite;
                }
                
                @keyframes spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
            </style>
        </div>
        """, unsafe_allow_html=True)

# Beautiful Animated Background
components.html("""
<style>
    .kawaii-background {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        z-index: -1;
        overflow: hidden;
        background: linear-gradient(-45deg, #0e0e10, #1a1a2e, #16213e, #0f3460);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .sakura-container {
        position: absolute;
        width: 100%;
        height: 100%;
        overflow: hidden;
    }
    
    .sakura {
        position: absolute;
        color: #4ecdc4;
        font-size: 12px;
        animation: fall linear infinite;
        opacity: 0.3;
    }
    
    @keyframes fall {
        0% {
            transform: translateY(-100vh) rotate(0deg);
            opacity: 0.3;
        }
        50% {
            opacity: 0.6;
        }
        100% {
            transform: translateY(100vh) rotate(360deg);
            opacity: 0;
        }
    }
    
    .stars {
        position: absolute;
        width: 100%;
        height: 100%;
    }
    
    .star {
        position: absolute;
        background: #4ecdc4;
        border-radius: 50%;
        animation: twinkle 3s ease-in-out infinite alternate;
    }
    
    @keyframes twinkle {
        0% { opacity: 0.2; transform: scale(1); }
        100% { opacity: 0.8; transform: scale(1.2); }
    }
    
    .floating-shapes {
        position: absolute;
        width: 100%;
        height: 100%;
    }
    
    .shape {
        position: absolute;
        opacity: 0.1;
        animation: float 20s ease-in-out infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px) rotate(0deg); }
        50% { transform: translateY(-20px) rotate(180deg); }
    }
    
    .anime-pattern {
        position: absolute;
        width: 100%;
        height: 100%;
        background-image: 
            radial-gradient(circle at 20% 80%, rgba(78, 205, 196, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 80% 20%, rgba(128, 203, 196, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 40% 40%, rgba(38, 166, 154, 0.05) 0%, transparent 50%);
        animation: patternMove 25s ease infinite;
    }
    
    @keyframes patternMove {
        0%, 100% { transform: scale(1) rotate(0deg); }
        50% { transform: scale(1.1) rotate(5deg); }
    }
</style>

<div class="kawaii-background">
    <!-- Animated gradient base -->
    <div class="anime-pattern"></div>
    
    <!-- Twinkling stars -->
    <div class="stars" id="stars"></div>
    
    <!-- Floating sakura petals -->
    <div class="sakura-container" id="sakura"></div>
    
    <!-- Floating geometric shapes -->
    <div class="floating-shapes" id="shapes"></div>
</div>

<script>
    // Create twinkling stars
    function createStars() {
        const starsContainer = document.getElementById('stars');
        if (!starsContainer) return;
        
        for (let i = 0; i < 50; i++) {
            const star = document.createElement('div');
            star.className = 'star';
            star.style.left = Math.random() * 100 + '%';
            star.style.top = Math.random() * 100 + '%';
            star.style.width = (Math.random() * 3 + 1) + 'px';
            star.style.height = star.style.width;
            star.style.animationDelay = Math.random() * 3 + 's';
            star.style.animationDuration = (Math.random() * 3 + 2) + 's';
            starsContainer.appendChild(star);
        }
    }
    
    // Create falling sakura petals
    function createSakura() {
        const sakuraContainer = document.getElementById('sakura');
        if (!sakuraContainer) return;
        
        const sakuraSymbols = ['🌸', '🌺', '✨', '⭐', '💫'];
        
        for (let i = 0; i < 15; i++) {
            const sakura = document.createElement('div');
            sakura.className = 'sakura';
            sakura.innerHTML = sakuraSymbols[Math.floor(Math.random() * sakuraSymbols.length)];
            sakura.style.left = Math.random() * 100 + '%';
            sakura.style.animationDuration = (Math.random() * 10 + 10) + 's';
            sakura.style.animationDelay = Math.random() * 5 + 's';
            sakura.style.fontSize = (Math.random() * 8 + 8) + 'px';
            sakuraContainer.appendChild(sakura);
        }
    }
    
    // Create floating geometric shapes
    function createShapes() {
        const shapesContainer = document.getElementById('shapes');
        if (!shapesContainer) return;
        
        const shapes = ['◇', '◈', '◉', '◎', '☆', '★'];
        
        for (let i = 0; i < 8; i++) {
            const shape = document.createElement('div');
            shape.className = 'shape';
            shape.innerHTML = shapes[Math.floor(Math.random() * shapes.length)];
            shape.style.left = Math.random() * 100 + '%';
            shape.style.top = Math.random() * 100 + '%';
            shape.style.fontSize = (Math.random() * 20 + 15) + 'px';
            shape.style.color = '#4ecdc4';
            shape.style.animationDuration = (Math.random() * 10 + 15) + 's';
            shape.style.animationDelay = Math.random() * 5 + 's';
            shapesContainer.appendChild(shape);
        }
    }
    
    // Initialize all animations
    setTimeout(() => {
        createStars();
        createSakura();
        createShapes();
    }, 100);
    
    // Add periodic sakura petals
    setInterval(() => {
        const sakuraContainer = document.getElementById('sakura');
        if (sakuraContainer && sakuraContainer.children.length < 20) {
            const sakura = document.createElement('div');
            sakura.className = 'sakura';
            sakura.innerHTML = ['🌸', '🌺', '✨'][Math.floor(Math.random() * 3)];
            sakura.style.left = Math.random() * 100 + '%';
            sakura.style.animationDuration = (Math.random() * 8 + 8) + 's';
            sakura.style.fontSize = (Math.random() * 6 + 10) + 'px';
            sakuraContainer.appendChild(sakura);
            
            // Remove after animation
            setTimeout(() => {
                if (sakura.parentNode) {
                    sakura.parentNode.removeChild(sakura);
                }
            }, 16000);
        }
    }, 3000);
</script>
""", height=0)

# Enhanced Header with Quick Actions
col1, col2, col3 = st.columns([3, 2, 2])

with col1:
    st.markdown("""
    <div style="margin: 0 0 5px 0; padding: 1px 0;">
        <span style="color: #4ecdc4; font-size: 1.1em; font-weight: 600;">🎌 KawaiiRecSys</span>
        <span style="color: #888; font-size: 0.7em; margin-left: 8px;">AI-Powered Anime Recommendation System</span>
    </div>
    """, unsafe_allow_html=True)

with col2:
    # Quick Action Buttons
    quick_col1, quick_col2 = st.columns(2)
    with quick_col1:
        if st.button("🎲 Random", help="Get a random anime recommendation"):
            random_anime = anime_df.sample(n=1).iloc[0]
            st.session_state.recommendations = pd.DataFrame([{
                'name': random_anime['name'],
                'genre': random_anime['genre'],
                'final_score': 8.0 + (hash(random_anime['name']) % 20) / 10  # Random score 8.0-9.9
            }])
            st.session_state.explanations = ["🎲 Random pick just for you!"]
    
    with quick_col2:
        if st.button("✨ Surprise", help="Get surprise recommendations"):
            surprise_anime = anime_df.sample(n=3)
            surprise_recs = []
            for _, anime in surprise_anime.iterrows():
                surprise_recs.append({
                    'name': anime['name'],
                    'genre': anime['genre'],
                    'final_score': 7.5 + (hash(anime['name']) % 25) / 10
                })
            st.session_state.recommendations = pd.DataFrame(surprise_recs)
            st.session_state.explanations = ["✨ Surprise!", "🎊 Discovery!", "🌟 Hidden gem!"]

with col3:
    # Advanced Search
    search_query = st.text_input("🔍 Search anime...", value=st.session_state.search_query, 
                                placeholder="Search by title, genre...", label_visibility="collapsed")
    if search_query != st.session_state.search_query:
        st.session_state.search_query = search_query
        if len(search_query) > 2:
            # Filter anime based on search
            filtered_anime = anime_df[
                anime_df['name'].str.contains(search_query, case=False, na=False) |
                anime_df['genre'].str.contains(search_query, case=False, na=False)
            ].head(5)
            
            if not filtered_anime.empty:
                search_recs = []
                for _, anime in filtered_anime.iterrows():
                    search_recs.append({
                        'name': anime['name'],
                        'genre': anime['genre'],
                        'final_score': 8.0 + (hash(anime['name']) % 20) / 10
                    })
                st.session_state.recommendations = pd.DataFrame(search_recs)
                st.session_state.explanations = [f"🔍 Search result for '{search_query}'" for _ in range(len(search_recs))]

# Session state already initialized at the top

# Advanced Filters Section (Expandable)
with st.expander("🎯 Advanced Filters & Settings", expanded=False):
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    
    with filter_col1:
        st.markdown("**📅 Release Filters**")
        year_range = st.slider("Year Range", 1960, 2024, (1990, 2024), help="Filter anime by release year")
        min_episodes = st.number_input("Min Episodes", 1, 1000, 1, help="Minimum episode count")
        max_episodes = st.number_input("Max Episodes", 1, 1000, 500, help="Maximum episode count")
    
    with filter_col2:
        st.markdown("**⭐ Rating Filters**")
        min_rating = st.slider("Minimum Rating", 1.0, 10.0, 6.0, 0.1, help="Filter by minimum rating")
        max_rating = st.slider("Maximum Rating", 1.0, 10.0, 10.0, 0.1, help="Filter by maximum rating")
        
        # Genre filter
        available_genres = set()
        for genres in anime_df['genre'].dropna():
            for genre in genres.split(','):
                available_genres.add(genre.strip())
        
        selected_genres = st.multiselect("🎭 Required Genres", sorted(list(available_genres)), 
                                       help="Must include ALL selected genres")
    
    with filter_col3:
        st.markdown("**🎨 Preferences**")
        
        # Mood selector
        mood = st.selectbox("🎭 Mood", [
            "Any", "Happy & Uplifting", "Dark & Serious", "Action-Packed", 
            "Romantic", "Mysterious", "Comedy", "Emotional"
        ], help="Filter by mood/tone")
        
        # Length preference
        length_pref = st.selectbox("📺 Length Preference", [
            "Any", "Short (1-12 episodes)", "Medium (13-26 episodes)", 
            "Long (27+ episodes)", "Movies only"
        ], help="Preferred anime length")
        
        # Studio filter (if available in data)
        exclude_watched = st.checkbox("❌ Exclude anime I've rated", 
                                    help="Hide anime you've already rated")

# Store filters in session state
st.session_state.active_filters = {
    'year_range': year_range,
    'episode_range': (min_episodes, max_episodes),
    'rating_range': (min_rating, max_rating),
    'genres': selected_genres,
    'mood': mood,
    'length_pref': length_pref,
    'exclude_watched': exclude_watched
}

# Full-width single row input controls
st.markdown('<div class="input-card" style="display: none;">', unsafe_allow_html=True)

# Single row with all controls - more compact
c1, c2, c3, c4, c5, c6, c7 = st.columns([0.6, 2.1, 0.8, 0.8, 0.8, 1.1, 0.8])

with c1:
    user_id = st.number_input("User ID", min_value=1, max_value=10000, value=42, label_visibility="collapsed")
    st.caption("User ID")

with c2:
    selected_anime = st.multiselect(
        "Anime",
        options=anime_list,
        default=["Fullmetal Alchemist: Brotherhood"],
        label_visibility="collapsed"
    )
    st.caption("Anime You've Enjoyed")

with c3:
    alpha = st.slider("SVD Weight", 0.0, 1.0, 0.4, 0.1, label_visibility="collapsed")
    st.caption("SVD")

with c4:
    beta = st.slider("Neural Weight", 0.0, 1.0, 0.3, 0.1, label_visibility="collapsed")
    st.caption("Neural")

with c5:
    gamma = round(1.0 - alpha - beta, 1)
    if gamma < 0:
        gamma = 0.0
    st.metric("Content Weight", f"{gamma:.1f}", label_visibility="collapsed")
    st.caption("Content")

with c6:
    if st.button("🚀 Recommend", use_container_width=True):
        st.session_state.recommending = True

with c7:
    prof_col, list_col = st.columns(2)
    with prof_col:
        st.session_state.profiling = st.checkbox("📊", value=False, help="Enable Performance Profiling")
    with list_col:
        if st.button("📚", help="View Watchlist"):
            if st.session_state.watchlist:
                watchlist_anime = anime_df[anime_df['name'].isin(st.session_state.watchlist)]
                if not watchlist_anime.empty:
                    watchlist_recs = []
                    for _, anime in watchlist_anime.iterrows():
                        watchlist_recs.append({
                            'name': anime['name'],
                            'genre': anime['genre'],
                            'final_score': 8.5  # Default score for watchlist
                        })
                    st.session_state.recommendations = pd.DataFrame(watchlist_recs)
                    st.session_state.explanations = ["📚 From your watchlist" for _ in range(len(watchlist_recs))]
            else:
                st.toast("Your watchlist is empty!")

st.markdown('</div>', unsafe_allow_html=True)

# Full-width recommendations section below
st.markdown('<div class="output-card">', unsafe_allow_html=True)

if st.session_state.recommending:
    # This will show when the recommendation process is happening
    show_loading_animation()
    
    # Get recommendations
    st.session_state.recommendations = get_recommendations(
        user_id, selected_anime, alpha, beta, gamma, ratings_df, anime_df, 
        enable_profiling=st.session_state.profiling
    )
    
    # Create explanations
    st.session_state.explanations = [
        f"SVD: {alpha:.1f}, NN: {beta:.1f}, Content: {gamma:.1f}" 
        for _ in range(len(st.session_state.recommendations))
    ]
    
    # Reset flag
    st.session_state.recommending = False
    
    # Auto-rerun to update UI
    st.rerun()

if st.session_state.recommendations is not None:
    show_netflix_style_recommendations(st.session_state.recommendations, st.session_state.explanations)
else:
    st.markdown("""
    <div style="text-align: center; color: #888; padding: 10px; font-size: 0.9em;">
        Click '🚀 Get Recommendations' to see anime suggestions here.
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# User Profile Dashboard
if st.session_state.watchlist or st.session_state.favorites or st.session_state.viewing_status:
    with st.expander("📊 Your Profile Dashboard", expanded=False):
        profile_col1, profile_col2, profile_col3, profile_col4 = st.columns(4)
        
        with profile_col1:
            st.metric("📚 Watchlist", len(st.session_state.watchlist), 
                     help="Total anime in your watchlist")
            
        with profile_col2:
            st.metric("❤️ Favorites", len(st.session_state.favorites), 
                     help="Total favorite anime")
        
        with profile_col3:
            watching_count = sum(1 for status in st.session_state.viewing_status.values() 
                               if status == "watching")
            st.metric("👀 Watching", watching_count, 
                     help="Currently watching anime")
        
        with profile_col4:
            completed_count = sum(1 for status in st.session_state.viewing_status.values() 
                                if status == "completed")
            st.metric("✅ Completed", completed_count, 
                     help="Completed anime")
        
        # Export Options
        st.markdown("**💾 Export Options**")
        export_col1, export_col2, export_col3 = st.columns(3)
        
        with export_col1:
            if st.button("📄 Export Watchlist"):
                if st.session_state.watchlist:
                    watchlist_text = "# My Anime Watchlist\n\n"
                    for i, anime in enumerate(st.session_state.watchlist, 1):
                        watchlist_text += f"{i}. {anime}\n"
                    
                    st.download_button(
                        label="⬇️ Download",
                        data=watchlist_text,
                        file_name="my_anime_watchlist.txt",
                        mime="text/plain"
                    )
                else:
                    st.toast("Watchlist is empty!")
        
        with export_col2:
            if st.button("❤️ Export Favorites"):
                if st.session_state.favorites:
                    favorites_text = "# My Favorite Anime\n\n"
                    for i, anime in enumerate(st.session_state.favorites, 1):
                        favorites_text += f"{i}. {anime}\n"
                    
                    st.download_button(
                        label="⬇️ Download",
                        data=favorites_text,
                        file_name="my_favorite_anime.txt",
                        mime="text/plain"
                    )
                else:
                    st.toast("No favorites yet!")
        
        with export_col3:
            if st.button("📊 Export Full Profile"):
                profile_text = "# My Anime Profile\n\n"
                profile_text += f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n\n"
                
                if st.session_state.watchlist:
                    profile_text += "## 📚 Watchlist\n"
                    for anime in st.session_state.watchlist:
                        profile_text += f"- {anime}\n"
                    profile_text += "\n"
                
                if st.session_state.favorites:
                    profile_text += "## ❤️ Favorites\n"
                    for anime in st.session_state.favorites:
                        profile_text += f"- {anime}\n"
                    profile_text += "\n"
                
                if st.session_state.viewing_status:
                    profile_text += "## 📺 Viewing Status\n"
                    status_names = {
                        "watching": "Currently Watching",
                        "completed": "Completed", 
                        "plan_to_watch": "Plan to Watch"
                    }
                    
                    for status_key, status_name in status_names.items():
                        anime_with_status = [anime for anime, status in st.session_state.viewing_status.items() 
                                           if status == status_key]
                        if anime_with_status:
                            profile_text += f"\n### {status_name}\n"
                            for anime in anime_with_status:
                                profile_text += f"- {anime}\n"
                
                st.download_button(
                    label="⬇️ Download",
                    data=profile_text,
                    file_name="my_anime_profile.md",
                    mime="text/markdown"
                )

# Data Management Section
st.markdown("---")
st.markdown("### 💾 Data Management")

manage_col1, manage_col2, manage_col3, manage_col4 = st.columns(4)

with manage_col1:
    if st.button("💾 Save Data", help="Manually save your data"):
        if save_user_data():
            st.toast("✅ Data saved successfully!")
        else:
            st.toast("❌ Failed to save data!")

with manage_col2:
    if st.button("🔄 Reload Data", help="Reload your saved data"):
        loaded, last_saved = load_user_data()
        if loaded:
            st.toast(f"✅ Data reloaded from {last_saved[:19] if last_saved else 'backup'}!")
            st.rerun()
        else:
            st.toast("❌ No saved data found!")

with manage_col3:
    if st.button("📦 Create Backup", help="Create a timestamped backup"):
        backup_path = create_backup()
        if backup_path:
            st.toast(f"✅ Backup created: {os.path.basename(backup_path)}")
        else:
            st.toast("❌ Failed to create backup!")

with manage_col4:
    if st.button("🗑️ Clear All Data", help="⚠️ This will permanently delete all your data!"):
        # Create emergency backup before clearing
        backup_path = create_backup()
        if backup_path:
            st.toast(f"🛡️ Emergency backup created: {os.path.basename(backup_path)}")
        
        # Clear session state
        st.session_state.watchlist = set()
        st.session_state.favorites = set()
        st.session_state.viewing_status = {}
        st.session_state.user_ratings = {}
        
        # Clear saved file
        try:
            data_path = get_user_data_path()
            if os.path.exists(data_path):
                os.remove(data_path)
        except Exception as e:
            st.error(f"Failed to clear saved data: {e}")
        
        st.toast("🗑️ All data cleared! Emergency backup created.")
        st.rerun()

# Show current data status
if st.session_state.watchlist or st.session_state.favorites or st.session_state.viewing_status:
    data_path = get_user_data_path()
    if os.path.exists(data_path):
        try:
            mod_time = datetime.fromtimestamp(os.path.getmtime(data_path))
            st.caption(f"💾 Last saved: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
        except:
            st.caption("💾 Data file exists")
    else:
        st.caption("⚠️ Data not yet saved to disk")

# Footer with random anime quote (more compact)
quote = get_random_quote()
st.markdown(f'<div class="quote-footer">"{quote}"</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    pass 
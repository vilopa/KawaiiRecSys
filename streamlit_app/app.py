import streamlit as st
import pandas as pd
import sys
import os
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
from collections import Counter

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
            padding: 5px !important;
            background-color: #0e0e10;
        }
        h1, h2, h3 {
            font-family: 'Helvetica Neue', sans-serif;
            font-weight: 600;
            margin-top: 0rem;
            margin-bottom: 0rem;
        }
        .stButton button {
            background: linear-gradient(145deg, #ff4baf, #ff0055);
            color: white;
            border-radius: 5px;
            border: none;
            font-weight: bold;
            padding: 5px;
            transition: all 0.3s ease;
            font-size: 1.1em;
            width: 100%;
            margin-top: 0rem;
        }
        .stButton button:hover {
            transform: scale(1.02);
            box-shadow: 0 0 5px rgba(255, 75, 175, 0.5);
        }
        .stSlider > div[data-baseweb="slider"] > div {
            background: linear-gradient(to right, #ff4baf, #ff0055);
        }
        .stNumberInput>div>div>input, .stTextInput>div>div>input {
            background-color: #1c1c1e;
            color: white;
            border-radius: 5px;
            border: 1px solid #ff4baf;
            padding: 5px;
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
            border: 1px solid #ff4baf;
            padding: 5px;
        }
        .stMultiSelect > div > div > div {
            color: white;
        }
        .recommendation-title {
            color: #ff4baf;
            margin-bottom: 5px;
            font-size: 1.3em;
            text-align: center;
        }
        .anime-card {
            background-color: #1c1c1e;
            border-radius: 5px;
            padding: 5px;
            box-shadow: 0 0 5px rgba(0,0,0,0.2);
            transition: 0.3s ease;
            overflow: hidden;
            text-align: center;
            color: white;
            margin: 5px;
            border: 1px solid rgba(255, 75, 175, 0.2);
        }
        .anime-card:hover {
            transform: scale(1.03);
            box-shadow: 0 0 5px rgba(255, 75, 175, 0.3);
            border-color: #ff4baf;
        }
        .anime-image {
            border-radius: 5px;
            width: 100%;
            height: 160px;
            object-fit: cover;
            margin-bottom: 5px;
        }
        .anime-title {
            margin: 0;
            font-size: 1em;
            font-weight: 600;
            color: white;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .anime-score {
            font-size: 0.85em;
            color: #ff4baf;
            margin: 5px 0;
        }
        .genre-tag {
            display: inline-block;
            padding: 5px;
            border-radius: 5px;
            font-size: 0.7em;
            margin: 5px;
            background-color: rgba(255, 75, 175, 0.2);
            color: #ff4baf;
        }
        .input-card {
            background-color: #1c1c1e;
            border-radius: 5px;
            padding: 5px;
            margin-bottom: 5px;
            margin-right: 5px;
            border: 1px solid rgba(255, 75, 175, 0.2);
            box-shadow: 0 0 5px rgba(0,0,0,0.1);
            height: 100%;
        }
        .output-card {
            background-color: #1c1c1e;
            border-radius: 5px;
            padding: 5px;
            margin-bottom: 5px;
            margin-left: 5px;
            border: 1px solid rgba(255, 75, 175, 0.2);
            box-shadow: 0 0 5px rgba(0,0,0,0.1);
            height: 100%;
            min-height: 400px;
        }
        .loading-text {
            text-align: center;
            color: #ff4baf;
            font-size: 1.2em;
            margin: 5px 0;
            animation: pulse 1.5s infinite;
        }
        .feedback-buttons {
            display: flex;
            justify-content: space-between;
            margin-top: 5px;
        }
        .like-button, .dislike-button {
            display: inline-block;
            cursor: pointer;
            padding: 5px;
            border-radius: 5px;
            font-size: 14px;
            text-align: center;
            background-color: rgba(255, 75, 175, 0.1);
            transition: all 0.2s ease;
        }
        .like-button:hover {
            background-color: rgba(0, 255, 0, 0.2);
        }
        .dislike-button:hover {
            background-color: rgba(255, 0, 0, 0.2);
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
            padding: 5px;
        }
        
        .stForm {
            padding: 5px;
        }
        
        .st-emotion-cache-1gulkj5 {
            padding: 5px;
        }
        
        .st-emotion-cache-16txtl3 {
            padding: 5px;
        }
        
        .st-emotion-cache-4z1n4l {
            padding: 5px;
        }
        
        div[data-testid="stVerticalBlock"] {
            gap: 5px;
            padding: 5px;
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
        
    df["bg_color"] = df["genre"].apply(genre_to_color)
    df["image_url"] = df["name"].apply(get_anime_image)
    
    # Create columns for each recommendation
    cols = st.columns(len(df))
    
    for idx, (col, row) in enumerate(zip(cols, df.itertuples())):
        with col:
            anime_name = row.name
            st.markdown(f"""
            <div class="anime-card" style="background-color: {row.bg_color};">
                <img src="{row.image_url}" class="anime-image" alt="{anime_name}" />
                <h5 class="anime-title">{anime_name}</h5>
                <p class="anime-score">⭐ {row.final_score:.2f}</p>
                <div class="genre-tags">
                    {''.join([f'<span class="genre-tag">{genre}</span>' for genre in row.genre.split(',')[:2]])}
                </div>
                <div class="feedback-buttons">
                    <div class="like-button" onclick="
                        var data = new FormData();
                        data.append('data', '{anime_name}|like');
                        fetch('/_stcore/message', {{
                            method: 'POST',
                            body: data
                        }})">👍</div>
                    <div class="dislike-button" onclick="
                        var data = new FormData();
                        data.append('data', '{anime_name}|dislike');
                        fetch('/_stcore/message', {{
                            method: 'POST',
                            body: data
                        }})">👎</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Explanation
            if explanations and idx < len(explanations):
                st.markdown(f"<div class='caption'>📝 {explanations[idx]}</div>", unsafe_allow_html=True)
    
    # Genre distribution chart
    st.markdown("<div class='vertical-chart'>", unsafe_allow_html=True)
    st.markdown("<h3 class='compact-h3'>Genre Distribution</h3>", unsafe_allow_html=True)
    genres = []
    for g in df['genre']:
        genres.extend([x.strip() for x in g.split(',')])
    genre_counts = Counter(genres)
    fig, ax = plt.subplots(figsize=(4, 2.5))
    ax.pie(genre_counts.values(), labels=genre_counts.keys(), autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

# Enhanced loading animation
def show_loading_animation():
    with st.spinner(""):
        st.markdown("""
        <div style="display: flex; justify-content: center; margin-top: 20px;">
            <div class="spinner">
                <style>
                    .spinner {
                        width: 40px;
                        height: 40px;
                        position: relative;
                    }
                    
                    .spinner:before {
                        content: "";
                        box-sizing: border-box;
                        position: absolute;
                        width: 40px;
                        height: 40px;
                        border-radius: 50%;
                        border: 3px solid transparent;
                        border-top-color: #ff4baf;
                        border-bottom-color: #ff4baf;
                        animation: spin 1s linear infinite;
                    }
                    
                    @keyframes spin {
                        0% { transform: rotate(0deg); }
                        100% { transform: rotate(360deg); }
                    }
                </style>
            </div>
        </div>
        <p style="text-align: center; margin-top: 10px; color: #ff4baf;">Finding the perfect anime for you...</p>
        """, unsafe_allow_html=True)

# Animated background
components.html("""
<div style="position: fixed; top: 0; left: 0; width: 100%; height: 100%; 
            background: url('https://media.giphy.com/media/xT0xeJpnrWC4XWblEk/giphy.gif') repeat;
            background-size: 150px; opacity: 0.05; z-index: -1;"></div>
""", height=0)

# Hero Banner (more compact)
st.markdown("""
<div style='text-align: center; '>
    <p style='color: #ff4baf; text-shadow: 0 0 10px rgba(255, 75, 175, 0.5);'>
        🎌 KawaiiRecSys 🎌
    </p>
    <p style='font-size: 1em; color: #cccccc; margin-top: 0;'>
        Your AI-Powered Anime Recommendation System 🌸
    </p>
</div>
""", unsafe_allow_html=True)

# Initialize session state for recommendations if it doesn't exist
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None
if 'explanations' not in st.session_state:
    st.session_state.explanations = None
if 'recommending' not in st.session_state:
    st.session_state.recommending = False
if 'profiling' not in st.session_state:
    st.session_state.profiling = False

# Create the half-half layout
left_col, right_col = st.columns(2)

# Left column - Input card
with left_col:
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    
    # User inputs
    st.markdown('<p class="section-title">🔍 Find Anime</p>', unsafe_allow_html=True)
    
    # User ID
    user_id = st.number_input("User ID", min_value=1, max_value=10000, value=42)
    
    # Anime selection
    selected_anime = st.multiselect(
        "Anime You've Enjoyed",
        options=anime_list,
        default=["Fullmetal Alchemist: Brotherhood"]
    )
    
    # Model weights sliders with clearer labels
    st.markdown("<h3 class='compact-h3'>Recommendation Weights</h3>", unsafe_allow_html=True)
    
    # SVD weight
    alpha = st.slider(
        "SVD Weight",
        min_value=0.0,
        max_value=1.0,
        value=0.4,
        step=0.1,
        help="Higher values prioritize recommendations based on other users with similar taste"
    )
    
    # Neural Network weight
    beta = st.slider(
        "Neural Net Weight",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.1,
        help="Higher values prioritize deep learning recommendations based on complex patterns"
    )
    
    # Content-based weight (automatically calculated)
    gamma = round(1.0 - alpha - beta, 1)
    if gamma < 0:
        gamma = 0.0
        st.warning("Total weight exceeds 1.0. Adjusting Content-Based weight to 0.")
        st.info(f"Current weights: SVD={alpha}, Neural={beta}, Content={gamma}")
    else:
        st.info(f"Content Weight: {gamma}")
    
    # Advanced options (collapsed by default)
    with st.expander("Advanced Options"):
        st.session_state.profiling = st.checkbox("Enable Performance Profiling", value=False)
        if st.session_state.profiling:
            st.warning("Profiling enabled. Performance results will be saved to 'profiles/' directory.")

    # Get recommendations button
    if st.button("Get Recommendations"):
        st.session_state.recommending = True
    
    st.markdown('</div>', unsafe_allow_html=True)

# Right column - Recommendations
with right_col:
    st.markdown('<div class="output-card">', unsafe_allow_html=True)
    st.markdown('<p class="recommendation-title">✨ Your Recommendations ✨</p>', unsafe_allow_html=True)
    
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
        st.info("Click 'Get Recommendations' to see suggestions here.")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer with random anime quote (more compact)
quote = get_random_quote()
st.markdown(f'<div class="quote-footer">"{quote}"</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    pass 
import pandas as pd
import os
import streamlit as st
from .jikan_api import fetch_anime_image, fetch_anime_data
import numpy as np
import random
import requests
import json
import time
from typing import Dict, List, Tuple, Any, Optional
import colorsys
import pickle

# Anime quotes for the footer
ANIME_QUOTES = [
    "I'll take a potato chip... and eat it!",
    "The world isn't perfect. But it's there for us, doing the best it can.",
    "Whatever you lose, you'll find it again. But what you throw away you'll never get back.",
    "People's lives don't end when they die, it ends when they lose faith.",
    "If you don't take risks, you can't create a future!",
    "If you don't share someone's pain, you can never understand them.",
    "Sometimes the questions are complicated and the answers are simple.",
    "I don't want to conquer anything. I just think the guy with the most freedom in this ocean is the Pirate King!",
    "It's not the face that makes someone a monster; it's the choices they make.",
    "Sometimes it's necessary to do unnecessary things.",
    "Being lonely is more painful than getting hurt.",
    "Those who forgive themselves, and are able to accept their true nature... They are the strong ones!"
]

# Constants
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Create cache directories
os.makedirs(os.path.join(PROJECT_ROOT, "cache"), exist_ok=True)
os.makedirs(os.path.join(PROJECT_ROOT, "cache", "images"), exist_ok=True)

# Image cache dictionary
IMAGE_CACHE = {}

def get_random_quote() -> str:
    """Get a random anime quote."""
    return random.choice(ANIME_QUOTES)

def load_anime_data() -> pd.DataFrame:
    """Load and preprocess anime data."""
    # Load anime.csv
    anime_path = os.path.join(PROJECT_ROOT, "data/anime.csv")
    anime_df = pd.read_csv(anime_path)
    
    # Clean the data
    anime_df.dropna(subset=['name'], inplace=True)
    anime_df.fillna({'genre': 'Unknown', 'type': 'Unknown', 'rating': 0, 'members': 0}, inplace=True)
    
    # Filter out anime with very few members
    anime_df = anime_df[anime_df['members'] >= 100]
    
    return anime_df

def hsv_to_hex(h: float, s: float, v: float) -> str:
    """Convert HSV color to hex."""
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return "#{:02x}{:02x}{:02x}".format(int(r*255), int(g*255), int(b*255))

def genre_to_color(genre_string: str) -> str:
    """Convert genre to a background color."""
    if pd.isna(genre_string) or genre_string == '':
        return '#1c1c1e'
        
    genres = genre_string.split(',')
    main_genre = genres[0].strip()
    
    # Map common genres to colors
    genre_colors = {
        'Action': '#ff5e5e',
        'Adventure': '#ffa55e',
        'Comedy': '#ffff5e',
        'Drama': '#ff5eff',
        'Fantasy': '#5effff',
        'Horror': '#8c1a1a',
        'Mystery': '#8c6e1a',
        'Romance': '#ff5ea3',
        'Sci-Fi': '#5e8cff',
        'Slice of Life': '#5eff8c',
        'Sports': '#5effa5',
        'Supernatural': '#a55eff',
        'Thriller': '#8c1a4b'
    }
    
    # Use the mapped color or generate one based on genre name
    if main_genre in genre_colors:
        return genre_colors[main_genre]
    else:
        # Generate a stable color based on the genre name
        hue = sum(ord(c) for c in main_genre) % 360 / 360
        return hsv_to_hex(hue, 0.6, 0.8)

def save_image_to_cache(anime_name: str, image_url: str) -> None:
    """Save image URL to cache."""
    cache_path = os.path.join(PROJECT_ROOT, "cache", "images", "image_cache.pkl")
    
    # Update memory cache
    IMAGE_CACHE[anime_name] = image_url
    
    # Update disk cache
    try:
        # Load existing cache if it exists
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                disk_cache = pickle.load(f)
        else:
            disk_cache = {}
            
        # Update and save
        disk_cache[anime_name] = image_url
        with open(cache_path, 'wb') as f:
            pickle.dump(disk_cache, f)
    except:
        # If caching fails, just continue
        pass

def load_image_cache() -> Dict[str, str]:
    """Load image cache from disk."""
    cache_path = os.path.join(PROJECT_ROOT, "cache", "images", "image_cache.pkl")
    
    # If memory cache is populated, use it
    if IMAGE_CACHE:
        return IMAGE_CACHE
        
    # Load from disk if available
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                IMAGE_CACHE.update(pickle.load(f))
            return IMAGE_CACHE
        except:
            # If loading fails, return empty dict
            return {}
    
    return {}

def get_anime_image(anime_name: str) -> str:
    """Get anime image URL for display."""
    # Load image cache
    image_cache = load_image_cache()
    
    # Check cache first
    if anime_name in image_cache:
        return image_cache[anime_name]
    
    # Fallback images
    fallback_images = [
        "https://cdn.myanimelist.net/images/anime/10/47347.jpg",
        "https://cdn.myanimelist.net/images/anime/5/73199.jpg",
        "https://cdn.myanimelist.net/images/anime/1208/94745.jpg",
        "https://cdn.myanimelist.net/images/anime/13/17405.jpg",
        "https://cdn.myanimelist.net/images/anime/9/9453.jpg"
    ]
    
    try:
        # Try to get image from Jikan API (MyAnimeList)
        search_url = f"https://api.jikan.moe/v4/anime?q={anime_name.replace(' ', '%20')}&limit=1"
        response = requests.get(search_url, timeout=2)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('data') and len(data['data']) > 0:
                image_url = data['data'][0]['images']['jpg']['image_url']
                
                # Cache the result
                save_image_to_cache(anime_name, image_url)
                
                return image_url
        
        # Rate limiting or error - use fallback
        fallback = random.choice(fallback_images)
        save_image_to_cache(anime_name, fallback)
        return fallback
        
    except Exception as e:
        # If API fails, use a fallback image
        fallback = random.choice(fallback_images)
        save_image_to_cache(anime_name, fallback)
        return fallback

def enrich_with_images(recommendations_df: pd.DataFrame) -> pd.DataFrame:
    """Add image URLs to recommendations DataFrame."""
    # Just return as is - image URLs will be fetched by the UI
    return recommendations_df

def get_anime_details(title: str) -> dict:
    """Get detailed anime information using Jikan API."""
    return fetch_anime_data(title)

def load_and_merge_data():
    """
    Loads and merges anime and ratings data from CSV files.
    
    This function:
    1. Checks for existence of required CSV files
    2. Validates required columns in both files
    3. Merges data on anime_id
    4. Handles missing values and data quality issues
    
    Returns:
        tuple: (merged_df, error_message)
            - merged_df: pandas DataFrame with merged data if successful, None if error
            - error_message: str describing any error, None if successful
            
    Required files:
        - data/anime.csv: Must contain anime_id, title, genre columns
        - data/ratings.csv: Must contain user_id, anime_id, rating columns
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    anime_path = os.path.join(project_root, "data/anime.csv")
    ratings_path = os.path.join(project_root, "data/ratings.csv")
    
    # Check for file existence
    if not os.path.exists(anime_path):
        return None, f"Missing required file: {anime_path}"
    if not os.path.exists(ratings_path):
        return None, f"Missing required file: {ratings_path}"
    
    try:
        # Load data with explicit column types
        anime = pd.read_csv(anime_path)
        ratings = pd.read_csv(ratings_path)
        
        # Validate required columns
        required_anime_cols = ['anime_id', 'title', 'genre']
        required_ratings_cols = ['user_id', 'anime_id', 'rating']
        
        missing_anime_cols = [col for col in required_anime_cols if col not in anime.columns]
        missing_ratings_cols = [col for col in required_ratings_cols if col not in ratings.columns]
        
        if missing_anime_cols:
            return None, f"Missing columns in anime.csv: {', '.join(missing_anime_cols)}"
        if missing_ratings_cols:
            return None, f"Missing columns in ratings.csv: {', '.join(missing_ratings_cols)}"
        
        # Clean and validate data
        ratings['rating'] = pd.to_numeric(ratings['rating'], errors='coerce')
        ratings = ratings.dropna(subset=['rating'])
        
        # Merge data
        merged = ratings.merge(anime, on="anime_id", how="left")
        
        # Handle missing values
        merged = merged.dropna(subset=["anime_id", "user_id", "rating", "title"])
        
        # Validate merged data
        if merged.empty:
            return None, "No valid data after merging and cleaning"
            
        # Reset index after cleaning
        merged = merged.reset_index(drop=True)
        
        return merged, None
        
    except pd.errors.EmptyDataError:
        return None, "One or both CSV files are empty"
    except pd.errors.ParserError:
        return None, "Error parsing CSV files. Please check file format"
    except Exception as e:
        return None, f"Unexpected error loading data: {str(e)}"

def check_dataset_status():
    """
    Checks if required data files exist in the project.
    
    Returns:
        dict: Dictionary with file names as keys and boolean existence status as values.
        Example: {"anime.csv": True, "ratings.csv": False}
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data")
    
    return {
        "anime.csv": os.path.exists(os.path.join(data_dir, "anime.csv")),
        "ratings.csv": os.path.exists(os.path.join(data_dir, "ratings.csv"))
    }

def display_project_status():
    """
    Displays a friendly, colorful status report of the KawaiiRecSys project.
    
    This function:
    1. Checks all required files and components
    2. Shows status with emojis and colors
    3. Provides helpful suggestions for missing files
    4. Displays overall project progress
    
    Usage:
        from utils.helpers import display_project_status
        display_project_status()  # Call at the start of your Streamlit app
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Define project components
    components = {
        "📂 Data Files": {
            "anime.csv": {
                "path": os.path.join(project_root, "data/anime.csv"),
                "suggestion": "💡 Please add anime.csv to /data/ with columns: anime_id, title, genre"
            },
            "ratings.csv": {
                "path": os.path.join(project_root, "data/ratings.csv"),
                "suggestion": "💡 Please add ratings.csv to /data/ with columns: user_id, anime_id, rating"
            }
        },
        "🚀 Recommender Logic": {
            "hybrid.py": {
                "path": os.path.join(project_root, "src/hybrid.py"),
                "suggestion": "💡 Please create hybrid.py in /src/ for hybrid recommendations"
            },
            "svd.py": {
                "path": os.path.join(project_root, "src/svd.py"),
                "suggestion": "💡 Please create svd.py in /src/ for SVD-based recommendations"
            }
        },
        "🧰 Utilities": {
            "helpers.py": {
                "path": os.path.join(project_root, "utils/helpers.py"),
                "suggestion": "💡 Please create helpers.py in /utils/ for utility functions"
            },
            "jikan_api.py": {
                "path": os.path.join(project_root, "utils/jikan_api.py"),
                "suggestion": "💡 Please create jikan_api.py in /utils/ for Jikan API integration"
            }
        },
        "🎨 Streamlit App": {
            "app.py": {
                "path": os.path.join(project_root, "streamlit_app/app.py"),
                "suggestion": "💡 Please create app.py in /streamlit_app/ for the main application"
            }
        }
    }
    
    # Track progress
    total_ready = 0
    total_components = 0
    
    # Display status for each component
    for component_name, files in components.items():
        st.subheader(component_name)
        component_ready = 0
        
        for file_name, info in files.items():
            exists = os.path.exists(info["path"])
            total_components += 1
            
            if exists:
                total_ready += 1
                component_ready += 1
                st.success(f"✅ {file_name}: Found at {info['path']}")
            else:
                st.error(f"❌ {file_name}: Missing")
                st.info(info["suggestion"])
        
        # Show component progress
        if len(files) > 1:
            st.write(f"Progress: {component_ready}/{len(files)} files ready")
    
    # Show overall progress
    progress = (total_ready / total_components) * 100
    st.subheader("📊 Overall Progress")
    st.progress(progress)
    st.write(f"Project is {progress:.1f}% ready") 
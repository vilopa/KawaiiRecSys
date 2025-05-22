import pandas as pd
import os
import streamlit as st
from .jikan_api import fetch_anime_image, fetch_anime_data

# Anime quotes for the footer
ANIME_QUOTES = [
    "In our society, letting others find out that you're a nice person is a very risky move. – Hitagi Senjougahara",
    "The world isn't perfect. But it's there for us, doing the best it can... that's what makes it so damn beautiful. – Roy Mustang",
    "Forgetting is like a wound. The wound may heal, but it has already left a scar. – Monkey D. Luffy",
    "If you don't like your destiny, don't accept it. Instead, have the courage to change it the way you want it to be. – Naruto Uzumaki",
    "A lesson without pain is meaningless. That's because no one can gain without sacrificing something. – Edward Elric"
]

def get_anime_image(title: str) -> str:
    """Get the image URL for a given anime title using Jikan API."""
    return fetch_anime_image(title)

def get_anime_details(title: str) -> dict:
    """Get detailed anime information using Jikan API."""
    return fetch_anime_data(title)

def enrich_with_images(df: pd.DataFrame, title_col: str = "name") -> pd.DataFrame:
    """
    Given a DataFrame with a title column, add an 'image_url' column using Jikan API.
    """
    df["image_url"] = df[title_col].apply(fetch_anime_image)
    return df

def genre_to_color(genre: str) -> str:
    """Map anime genres to specific colors."""
    genre = str(genre).lower()
    if "romance" in genre:
        return "#ff4baf"
    elif "action" in genre:
        return "#ff0000"
    elif "comedy" in genre:
        return "#ff9500"
    elif "horror" in genre:
        return "#6e0dd0"
    elif "sci-fi" in genre or "scifi" in genre:
        return "#00bfff"
    elif "drama" in genre:
        return "#ff69b4"
    elif "fantasy" in genre:
        return "#9370db"
    elif "mystery" in genre:
        return "#4b0082"
    else:
        return "#1c1c1e"

def load_anime_data() -> pd.DataFrame:
    """Load the anime data from CSV file."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return pd.read_csv(os.path.join(project_root, "data/anime.csv"))

def get_random_quote() -> str:
    """Get a random anime quote."""
    import random
    return random.choice(ANIME_QUOTES)

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
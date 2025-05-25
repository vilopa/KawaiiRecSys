import pandas as pd
import numpy as np
from typing import List, Dict, Any
from .svd import train_svd_model, get_svd_recommendations
from .neural_net import train_neural_model, get_neural_recommendations
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.helpers import enrich_with_images
import time
import os
import pickle
import cProfile
import pstats
import io

# Create cache directory if it doesn't exist
os.makedirs("cache", exist_ok=True)
os.makedirs("profiles", exist_ok=True)  # Create directory for profile results

# Cache for trained models
MODEL_CACHE = {}

def get_content_based_recommendations(
    anime_df: pd.DataFrame,
    selected_anime: List[str],
    top_n: int = 10
) -> pd.DataFrame:
    """Get content-based recommendations based on selected anime."""
    # Create TF-IDF matrix for genres
    tfidf = TfidfVectorizer(stop_words='english')
    genre_matrix = tfidf.fit_transform(anime_df['genre'].fillna(''))
    
    # Get indices of selected anime
    selected_indices = anime_df[anime_df['name'].isin(selected_anime)].index
    
    if len(selected_indices) == 0:
        return pd.DataFrame()  # Return empty DataFrame if no matches
    
    # Calculate average similarity to selected anime
    selected_similarity = np.mean(cosine_similarity(genre_matrix[selected_indices], genre_matrix), axis=0)
    
    # Create recommendations DataFrame
    recommendations = anime_df.copy()
    recommendations['content_score'] = selected_similarity
    
    # Remove selected anime from recommendations
    recommendations = recommendations[~recommendations['name'].isin(selected_anime)]
    
    return recommendations.sort_values('content_score', ascending=False).head(top_n)

def get_model_cache_key(user_id, model_type):
    """Generate a cache key for models"""
    return f"{model_type}_{user_id}"

def load_cached_model(user_id, model_type):
    """Load model from cache if available"""
    cache_key = get_model_cache_key(user_id, model_type)
    
    # Check memory cache first
    if cache_key in MODEL_CACHE:
        return MODEL_CACHE[cache_key]
    
    # Check disk cache
    cache_file = f"cache/{cache_key}.pkl"
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                model_data = pickle.load(f)
                MODEL_CACHE[cache_key] = model_data
                return model_data
        except:
            # If loading fails, just return None
            return None
    
    return None

def save_model_to_cache(user_id, model_type, model_data):
    """Save model to cache"""
    cache_key = get_model_cache_key(user_id, model_type)
    
    # Save to memory cache
    MODEL_CACHE[cache_key] = model_data
    
    # Save to disk cache
    cache_file = f"cache/{cache_key}.pkl"
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(model_data, f)
    except:
        # If saving fails, just continue
        pass

def hybrid_recommend(
    user_id: int,
    selected_anime: List[str],
    ratings_df: pd.DataFrame,
    anime_df: pd.DataFrame,
    top_n: int = 10,
    alpha: float = 0.4,  # Adjusted weight distribution
    beta: float = 0.3,   # Weight for neural network
    gamma: float = 0.3   # Weight for content-based
) -> pd.DataFrame:
    """Get hybrid recommendations combining SVD, neural network, and content-based approaches."""
    # Limit ratings to improve performance
    start_time = time.time()
    
    # Sample ratings data (at most 100,000 ratings) if the dataset is large
    sample_size = min(500, len(ratings_df))
    if len(ratings_df) > sample_size:
        # Ensure user's ratings are included in the sample
        user_ratings = ratings_df[ratings_df['user_id'] == user_id]
        other_ratings = ratings_df[ratings_df['user_id'] != user_id].sample(
            sample_size - len(user_ratings), 
            random_state=42
        )
        sampled_ratings = pd.concat([user_ratings, other_ratings])
    else:
        sampled_ratings = ratings_df
    
    # Get content-based recommendations (fast, do this first)
    content_recs = get_content_based_recommendations(anime_df, selected_anime, top_n * 2)
    
    # Get SVD recommendations
    svd_cached = load_cached_model(user_id, 'svd')
    if svd_cached:
        svd_model = svd_cached
    else:
        svd_model = train_svd_model(sampled_ratings)
        save_model_to_cache(user_id, 'svd', svd_model)
    
    svd_recs = get_svd_recommendations(svd_model, user_id, anime_df, top_n * 2)
    
    # Get neural network recommendations only if needed (based on beta weight)
    if beta > 0.1:  # Only use neural if weight is significant
        neural_cached = load_cached_model(user_id, 'neural')
        if neural_cached:
            neural_model, user_encoder, anime_encoder = neural_cached
        else:
            neural_model, user_encoder, anime_encoder = train_neural_model(sampled_ratings)
            save_model_to_cache(user_id, 'neural', (neural_model, user_encoder, anime_encoder))
        
        neural_recs = get_neural_recommendations(
            neural_model, user_id, anime_df, user_encoder, anime_encoder, sampled_ratings, top_n * 2
        )
    else:
        neural_recs = pd.DataFrame()  # Empty DF if neural weight is too low
    
    # Handle empty recommendation sets
    if content_recs.empty and neural_recs.empty:
        return svd_recs.head(top_n)
    elif content_recs.empty:
        # If no content recommendations, adjust weights between SVD and neural
        alpha = 0.6
        beta = 0.4
        gamma = 0.0
    elif neural_recs.empty or beta <= 0.1:
        # If no neural recommendations, adjust weights between SVD and content
        alpha = 0.6
        gamma = 0.4
        beta = 0.0
    
    # Prepare DataFrames for merge
    svd_df = svd_recs[['anime_id', 'predicted_rating']].copy()
    
    # Merge recommendations
    all_anime_ids = set(svd_df['anime_id'])
    
    # Add neural recommendations if available
    if not neural_recs.empty:
        neural_df = neural_recs[['anime_id', 'neural_score']].copy()
        all_anime_ids.update(neural_df['anime_id'])
    else:
        neural_df = pd.DataFrame(columns=['anime_id', 'neural_score'])
    
    # Add content recommendations if available
    if not content_recs.empty:
        content_df = content_recs[['anime_id', 'content_score']].copy()
        all_anime_ids.update(content_df['anime_id'])
    else:
        content_df = pd.DataFrame(columns=['anime_id', 'content_score'])
    
    # Create a unified DataFrame with all anime IDs
    hybrid_recs = pd.DataFrame({'anime_id': list(all_anime_ids)})
    
    # Merge with individual recommendation DataFrames
    hybrid_recs = hybrid_recs.merge(svd_df, on='anime_id', how='left')
    hybrid_recs = hybrid_recs.merge(neural_df, on='anime_id', how='left')
    hybrid_recs = hybrid_recs.merge(content_df, on='anime_id', how='left')
    
    # Fill missing values with 0
    hybrid_recs = hybrid_recs.fillna(0)
    
    # Normalize scores
    for col in ['predicted_rating', 'neural_score', 'content_score']:
        if hybrid_recs[col].max() > hybrid_recs[col].min():
            hybrid_recs[col] = (hybrid_recs[col] - hybrid_recs[col].min()) / \
                               (hybrid_recs[col].max() - hybrid_recs[col].min())
    
    # Calculate final score with three components
    hybrid_recs['final_score'] = (alpha * hybrid_recs['predicted_rating'] + 
                                  beta * hybrid_recs['neural_score'] + 
                                  gamma * hybrid_recs['content_score'])
    
    # Sort by final score and get top N
    hybrid_recs = hybrid_recs.sort_values('final_score', ascending=False).head(top_n)
    
    # Add anime details
    hybrid_recs = hybrid_recs.merge(
        anime_df[['anime_id', 'name', 'genre', 'type', 'rating']],
        on='anime_id'
    )
    
    # Get only necessary images to speed up loading
    hybrid_recs_with_images = enrich_with_images(hybrid_recs)
    
    end_time = time.time()
    print(f"Recommendation time: {end_time - start_time:.2f} seconds")
    
    return hybrid_recs_with_images 

def profiled_hybrid_recommend(*args, **kwargs):
    """
    Profile the hybrid_recommend function and save results to a file.
    Uses the same parameters as hybrid_recommend.
    """
    profile_filename = f"profiles/hybrid_recommend_profile_{int(time.time())}.prof"
    
    # Run the profiler
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Execute the function
    result = hybrid_recommend(*args, **kwargs)
    
    # Disable profiler and print stats
    profiler.disable()
    
    # Save profile results to file
    profiler.dump_stats(profile_filename)
    
    # Print stats to console
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(20)  # Print top 20 time-consuming functions
    print(f"Profile saved to {profile_filename}")
    print("\nProfile Results:")
    print(s.getvalue())
    
    return result 
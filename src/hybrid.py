import pandas as pd
import numpy as np
from typing import List, Dict, Any
from .svd import train_svd_model, get_svd_recommendations
from .neural_net import train_neural_model, get_neural_recommendations
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.helpers import enrich_with_images

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
    # Get SVD recommendations
    svd_model = train_svd_model(ratings_df)
    svd_recs = get_svd_recommendations(svd_model, user_id, anime_df, top_n * 2)
    
    # Get neural network recommendations
    neural_model, user_encoder, anime_encoder = train_neural_model(ratings_df)
    neural_recs = get_neural_recommendations(
        neural_model, user_id, anime_df, user_encoder, anime_encoder, ratings_df, top_n * 2
    )
    
    # Get content-based recommendations
    content_recs = get_content_based_recommendations(anime_df, selected_anime, top_n * 2)
    
    # Handle empty recommendation sets
    if content_recs.empty and neural_recs.empty:
        return svd_recs.head(top_n)
    elif content_recs.empty:
        # If no content recommendations, adjust weights between SVD and neural
        alpha = 0.6
        beta = 0.4
        gamma = 0.0
    elif neural_recs.empty:
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
    
    # Add image URLs using Jikan API
    hybrid_recs = enrich_with_images(hybrid_recs)
    
    return hybrid_recs 
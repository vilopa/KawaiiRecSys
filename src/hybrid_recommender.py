import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, Dataset, Reader
import os

# Get the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load datasets with correct paths
anime_df = pd.read_csv(os.path.join(project_root, "data/anime.csv"))
rating_df = pd.read_csv(os.path.join(project_root, "data/rating.csv"))
ratings_df = rating_df[rating_df['rating'] != -1]

# TF-IDF Vectorizer for genres
anime_df['genre'] = anime_df['genre'].fillna('').str.lower()
tfidf = TfidfVectorizer(token_pattern=r'[^,]+')
tfidf_matrix = tfidf.fit_transform(anime_df['genre'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
anime_df = anime_df.reset_index()

# Build SVD model
reader = Reader(rating_scale=(1, 10))
data = Dataset.load_from_df(ratings_df[['user_id', 'anime_id', 'rating']], reader)
trainset = data.build_full_trainset()
svd = SVD()
svd.fit(trainset)

def hybrid_recommend(user_id, anime_titles, top_n=5, alpha=0.6):
    """
    Generate hybrid recommendations combining content-based and collaborative filtering.
    Supports multiple anime titles as input.
    
    Args:
        user_id (int): User ID for collaborative filtering
        anime_titles (str or list): Single anime title or list of anime titles
        top_n (int): Number of recommendations to return
        alpha (float): Weight for collaborative filtering (0-1)
        
    Returns:
        pd.DataFrame: DataFrame containing recommendations and scores
    """
    # Convert single title to list
    if isinstance(anime_titles, str):
        anime_titles = [anime_titles]
    
    # Initialize recommendation scores
    all_recommendations = []
    
    for title in anime_titles:
        # Find anime index
        idx = anime_df[anime_df['name'].str.lower() == title.lower()].index
        
        if len(idx) == 0:
            return f"Anime titled '{title}' not found in dataset."
        
        idx = idx[0]
        
        # Get similarity scores
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+20]
        
        # Calculate hybrid scores
        for i, sim in sim_scores:
            anime_id = anime_df.loc[i, 'anime_id']
            pred = svd.predict(user_id, anime_id).est
            final_score = (1 - alpha) * sim + alpha * (pred / 10)  # Normalize rating (1-10) to 0-1
            all_recommendations.append({
                'index': i,
                'score': final_score,
                'title': title
            })
    
    # Aggregate scores for each anime
    score_dict = {}
    for rec in all_recommendations:
        idx = rec['index']
        if idx not in score_dict:
            score_dict[idx] = {
                'scores': [],
                'titles': []
            }
        score_dict[idx]['scores'].append(rec['score'])
        score_dict[idx]['titles'].append(rec['title'])
    
    # Calculate average scores
    recommendations = []
    for idx, data in score_dict.items():
        avg_score = sum(data['scores']) / len(data['scores'])
        recommendations.append((idx, avg_score))
    
    # Sort and get top N
    recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)[:top_n]
    rec_indices = [i[0] for i in recommendations]
    
    # Prepare result DataFrame
    result = anime_df.loc[rec_indices, ['name', 'genre', 'type', 'rating']].copy()
    result['final_score'] = [round(i[1], 3) for i in recommendations]
    
    # Add placeholder image URLs if not present
    if 'image_url' not in result.columns:
        result['image_url'] = [f"https://via.placeholder.com/120/ff4baf/ffffff?text={name[:10]}" 
                             for name in result['name']]
    
    return result.reset_index(drop=True)

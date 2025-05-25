import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split, cross_validate
from typing import List, Dict, Any
import pickle
import os

class SVDRecSys:
    def __init__(self, anime_df, rating_df):
        """
        Initialize the SVD-based recommendation system.
        
        Args:
            anime_df (pd.DataFrame): DataFrame containing anime information
            rating_df (pd.DataFrame): DataFrame containing user ratings
        """
        self.anime_df = anime_df
        self.rating_df = rating_df
        self.model = None
        self.trainset = None
        
    def prepare_data(self):
        """Prepare the data for the SVD model."""
        # Remove unrated entries
        rating_df = self.rating_df[self.rating_df['rating'] != -1]
        
        # Prepare data for Surprise
        reader = Reader(rating_scale=(1, 10))
        data = Dataset.load_from_df(rating_df[['user_id', 'anime_id', 'rating']], reader)
        
        # Split data into train and test sets
        trainset, _ = train_test_split(data, test_size=0.2, random_state=42)
        self.trainset = trainset
        
        return trainset
    
    def train_model(self, n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02):
        """
        Train the SVD model with the given parameters.
        
        Args:
            n_factors (int): Number of factors for the SVD model
            n_epochs (int): Number of epochs for training
            lr_all (float): Learning rate for all parameters
            reg_all (float): Regularization term for all parameters
        """
        if self.trainset is None:
            self.prepare_data()
            
        # Initialize and train SVD model
        self.model = SVD(
            n_factors=n_factors,
            n_epochs=n_epochs,
            lr_all=lr_all,
            reg_all=reg_all
        )
        self.model.fit(self.trainset)
        
    def get_user_recommendations(self, user_id, top_n=10):
        """
        Get personalized recommendations for a user.
        
        Args:
            user_id (int): ID of the user
            top_n (int): Number of recommendations to return
            
        Returns:
            pd.DataFrame: DataFrame containing top N recommendations
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
            
        # Get list of all anime IDs
        all_anime_ids = self.rating_df['anime_id'].unique()
        
        # Get anime that user has already rated
        user_rated = self.rating_df[self.rating_df['user_id'] == user_id]['anime_id'].values
        
        # Get anime that user hasn't rated yet
        unrated_anime = [aid for aid in all_anime_ids if aid not in user_rated]
        
        # Make predictions for unrated anime
        predictions = []
        for anime_id in unrated_anime:
            pred = self.model.predict(user_id, anime_id)
            predictions.append((anime_id, pred.est))
        
        # Sort predictions by estimated rating
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        # Get top N recommendations
        top_n = predictions[:top_n]
        
        # Create DataFrame with recommendations
        recommendations = pd.DataFrame(top_n, columns=['anime_id', 'predicted_rating'])
        recommendations = recommendations.merge(
            self.anime_df[['anime_id', 'name', 'genre', 'type', 'rating']], 
            on='anime_id'
        )
        
        return recommendations
    
    def save_model(self, filepath='models/svd_model.pkl'):
        """Save the trained model to a file."""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
            
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
            
    def load_model(self, filepath='models/svd_model.pkl'):
        """Load a trained model from a file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found at {filepath}")
            
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
            
        # Recreate trainset for predictions
        self.prepare_data()

def train_svd_model(ratings_df: pd.DataFrame) -> SVD:
    """
    Train an SVD model on the ratings data.
    
    Args:
        ratings_df (pd.DataFrame): DataFrame containing user ratings
        
    Returns:
        SVD: Trained SVD model
    """
    # Filter out ratings with -1 (not rated)
    filtered_ratings = ratings_df[ratings_df['rating'] != -1].copy()
    
    # For large datasets, take a sample to speed up training
    sample_size = min(5000, len(filtered_ratings))
    if len(filtered_ratings) > sample_size:
        filtered_ratings = filtered_ratings.sample(sample_size, random_state=42)
    
    # Create Surprise reader and dataset
    reader = Reader(rating_scale=(1, 10))
    data = Dataset.load_from_df(
        filtered_ratings[['user_id', 'anime_id', 'rating']], 
        reader
    )
    
    # Build full trainset
    trainset = data.build_full_trainset()
    
    # Train SVD model with optimized parameters for speed
    svd = SVD(
        n_factors=50,
        n_epochs=10,  # Reduced from default 20
        lr_all=0.01,  # Increased learning rate for faster convergence
        reg_all=0.02
    )
    svd.fit(trainset)
    
    return svd

def get_svd_recommendations(
    model: SVD,
    user_id: int,
    anime_df: pd.DataFrame,
    top_n: int = 10
) -> pd.DataFrame:
    """
    Get recommendations for a user using the trained SVD model.
    
    Args:
        model (SVD): Trained SVD model
        user_id (int): User ID to get recommendations for
        anime_df (pd.DataFrame): DataFrame containing anime information
        top_n (int): Number of recommendations to return
        
    Returns:
        pd.DataFrame: DataFrame containing top N recommendations
    """
    # Get all anime IDs
    all_anime_ids = anime_df['anime_id'].unique()
    
    # For faster predictions, limit to a maximum of 5000 anime
    if len(all_anime_ids) > 5000:
        # Include some of the most popular anime
        popular_anime = anime_df.nlargest(1000, 'members')['anime_id'].values
        
        # Take a random sample for the rest
        remaining_anime = np.setdiff1d(all_anime_ids, popular_anime)
        random_sample = np.random.choice(remaining_anime, 4000, replace=False)
        
        all_anime_ids = np.concatenate([popular_anime, random_sample])
    
    # Make predictions
    predictions = []
    for anime_id in all_anime_ids:
        try:
            # Use raw prediction for faster performance (no inner algorithm)
            pred = model.predict(user_id, anime_id)
            predictions.append((anime_id, pred.est))
        except:
            # Skip if prediction fails
            continue
    
    # Sort predictions by estimated rating
    predictions.sort(key=lambda x: x[1], reverse=True)
    
    # Get top N recommendations
    top_predictions = predictions[:top_n]
    
    # Create DataFrame with recommendations
    recommendations = pd.DataFrame(top_predictions, columns=['anime_id', 'predicted_rating'])
    
    # Add anime information
    recommendations = recommendations.merge(
        anime_df[['anime_id', 'name', 'genre', 'type', 'rating']], 
        on='anime_id'
    )
    
    return recommendations

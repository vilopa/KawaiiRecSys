import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
import pickle
import os
from typing import List, Dict, Any

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
    """Train SVD model on ratings data."""
    reader = Reader(rating_scale=(1, 10))
    data = Dataset.load_from_df(ratings_df[['user_id', 'anime_id', 'rating']], reader)
    trainset = data.build_full_trainset()
    
    model = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02)
    model.fit(trainset)
    return model

def get_svd_recommendations(
    model: SVD,
    user_id: int,
    anime_df: pd.DataFrame,
    top_n: int = 10
) -> pd.DataFrame:
    """Get recommendations using SVD model."""
    # Get all anime IDs
    all_anime_ids = anime_df['anime_id'].unique()
    
    # Get predictions for all anime
    predictions = []
    for anime_id in all_anime_ids:
        pred = model.predict(user_id, anime_id)
        predictions.append({
            'anime_id': anime_id,
            'predicted_rating': pred.est
        })
    
    # Convert to DataFrame and merge with anime info
    pred_df = pd.DataFrame(predictions)
    recommendations = pred_df.merge(anime_df, on='anime_id')
    
    # Sort by predicted rating and return top N
    return recommendations.sort_values('predicted_rating', ascending=False).head(top_n)

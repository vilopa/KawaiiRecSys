import pandas as pd
import numpy as np
import os
import pickle
from typing import List, Dict, Any, Tuple
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model, save_model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class NeuralRecSys:
    def __init__(self, anime_df, rating_df):
        """
        Initialize the Neural Network-based recommendation system.
        
        Args:
            anime_df (pd.DataFrame): DataFrame containing anime information
            rating_df (pd.DataFrame): DataFrame containing user ratings
        """
        self.anime_df = anime_df
        self.rating_df = rating_df
        self.model = None
        self.user_encoder = LabelEncoder()
        self.anime_encoder = LabelEncoder()
        
    def prepare_data(self):
        """Prepare the data for the neural network model."""
        # Remove unrated entries
        rating_df = self.rating_df[self.rating_df['rating'] != -1].copy()
        
        # Encode user and anime IDs
        rating_df['user_encoded'] = self.user_encoder.fit_transform(rating_df['user_id'])
        rating_df['anime_encoded'] = self.anime_encoder.fit_transform(rating_df['anime_id'])
        
        # Get total number of users and anime
        self.n_users = len(rating_df['user_encoded'].unique())
        self.n_anime = len(rating_df['anime_encoded'].unique())
        
        # Prepare train and test sets
        train_data, test_data = train_test_split(
            rating_df, test_size=0.2, random_state=42
        )
        
        self.train_data = train_data
        self.test_data = test_data
        
        return train_data, test_data
    
    def build_model(self, embedding_size=50):
        """
        Build the neural network model.
        
        Args:
            embedding_size (int): Size of the embedding layers
        """
        # Define user input and embedding
        user_input = Input(shape=(1,), name='user_input')
        user_embedding = Embedding(self.n_users, embedding_size, name='user_embedding')(user_input)
        user_vec = Flatten(name='flatten_users')(user_embedding)
        
        # Define anime input and embedding
        anime_input = Input(shape=(1,), name='anime_input')
        anime_embedding = Embedding(self.n_anime, embedding_size, name='anime_embedding')(anime_input)
        anime_vec = Flatten(name='flatten_anime')(anime_embedding)
        
        # Concatenate user and anime embeddings
        concat = Concatenate()([user_vec, anime_vec])
        
        # Add dense layers
        dense1 = Dense(128, activation='relu')(concat)
        dense2 = Dense(64, activation='relu')(dense1)
        dense3 = Dense(32, activation='relu')(dense2)
        
        # Output layer
        output = Dense(1)(dense3)
        
        # Create model
        model = Model(inputs=[user_input, anime_input], outputs=output)
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mean_squared_error'
        )
        
        self.model = model
        return model
    
    def train_model(self, epochs=20, batch_size=64, validation_split=0.1):
        """
        Train the neural network model.
        
        Args:
            epochs (int): Number of epochs for training
            batch_size (int): Batch size for training
            validation_split (float): Validation split ratio
        """
        if not hasattr(self, 'train_data'):
            self.prepare_data()
            
        if self.model is None:
            self.build_model()
        
        # Prepare training data
        train_users = self.train_data['user_encoded'].values
        train_anime = self.train_data['anime_encoded'].values
        train_ratings = self.train_data['rating'].values
        
        # Define early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )
        
        # Train model
        history = self.model.fit(
            [train_users, train_anime],
            train_ratings,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=1
        )
        
        return history
    
    def evaluate_model(self):
        """Evaluate the model on the test set."""
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
            
        # Prepare test data
        test_users = self.test_data['user_encoded'].values
        test_anime = self.test_data['anime_encoded'].values
        test_ratings = self.test_data['rating'].values
        
        # Evaluate model
        loss = self.model.evaluate([test_users, test_anime], test_ratings, verbose=0)
        print(f"Test MSE: {loss}")
        
        return loss
    
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
            
        # Check if user is in the encoding
        if user_id not in self.user_encoder.classes_:
            print(f"User ID {user_id} not found in the dataset")
            return None
            
        # Get user encoded ID
        user_encoded = self.user_encoder.transform([user_id])[0]
        
        # Get anime that user has already rated
        user_rated = self.rating_df[self.rating_df['user_id'] == user_id]['anime_id'].values
        
        # Get all anime IDs and their encodings
        all_anime = self.anime_df['anime_id'].unique()
        
        # Filter anime IDs that are in the encoding
        valid_anime = [aid for aid in all_anime if aid in self.anime_encoder.classes_]
        
        # Get anime that user hasn't rated yet
        unrated_anime = [aid for aid in valid_anime if aid not in user_rated]
        
        # If no unrated anime, return None
        if not unrated_anime:
            print(f"User {user_id} has rated all available anime")
            return None
            
        # Prepare data for prediction
        user_data = np.array([user_encoded] * len(unrated_anime))
        anime_data = self.anime_encoder.transform(unrated_anime)
        
        # Make predictions
        predictions = self.model.predict([user_data, anime_data], verbose=0)
        
        # Create DataFrame with predictions
        recommendations = pd.DataFrame({
            'anime_id': unrated_anime,
            'predicted_rating': predictions.flatten()
        })
        
        # Sort by predicted rating
        recommendations = recommendations.sort_values('predicted_rating', ascending=False)
        
        # Get top N recommendations
        recommendations = recommendations.head(top_n)
        
        # Add anime information
        recommendations = recommendations.merge(
            self.anime_df[['anime_id', 'name', 'genre', 'type', 'rating']], 
            on='anime_id'
        )
        
        return recommendations
    
    def save_model(self, model_path='models/neural_model', encoders_path='models/neural_encoders.pkl'):
        """Save the trained model and encoders to files."""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        os.makedirs(os.path.dirname(encoders_path), exist_ok=True)
        
        # Save model
        save_model(self.model, model_path)
        
        # Save encoders
        with open(encoders_path, 'wb') as f:
            pickle.dump({
                'user_encoder': self.user_encoder,
                'anime_encoder': self.anime_encoder,
                'n_users': self.n_users,
                'n_anime': self.n_anime
            }, f)
            
    def load_model(self, model_path='models/neural_model', encoders_path='models/neural_encoders.pkl'):
        """Load a trained model and encoders from files."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        if not os.path.exists(encoders_path):
            raise FileNotFoundError(f"Encoders file not found at {encoders_path}")
            
        # Load model
        self.model = load_model(model_path)
        
        # Load encoders
        with open(encoders_path, 'rb') as f:
            encoders_data = pickle.load(f)
            self.user_encoder = encoders_data['user_encoder']
            self.anime_encoder = encoders_data['anime_encoder']
            self.n_users = encoders_data['n_users']
            self.n_anime = encoders_data['n_anime']


def train_neural_model(ratings_df: pd.DataFrame) -> Tuple[Model, LabelEncoder, LabelEncoder]:
    """
    Train a neural network model on ratings data.
    
    Args:
        ratings_df (pd.DataFrame): DataFrame containing user ratings
        
    Returns:
        tuple: (model, user_encoder, anime_encoder) - Trained model and encoders
    """
    # Remove negative ratings
    ratings_df = ratings_df[ratings_df['rating'] != -1].copy()
    
    # Encode user IDs and anime IDs
    user_encoder = LabelEncoder()
    anime_encoder = LabelEncoder()
    
    ratings_df['user_encoded'] = user_encoder.fit_transform(ratings_df['user_id'])
    ratings_df['anime_encoded'] = anime_encoder.fit_transform(ratings_df['anime_id'])
    
    # Get number of unique users and anime
    n_users = len(ratings_df['user_encoded'].unique())
    n_anime = len(ratings_df['anime_encoded'].unique())
    
    # Build model
    embedding_size = 50
    
    # User embedding
    user_input = Input(shape=(1,), name='user_input')
    user_embedding = Embedding(n_users, embedding_size, name='user_embedding')(user_input)
    user_vec = Flatten(name='flatten_users')(user_embedding)
    
    # Anime embedding
    anime_input = Input(shape=(1,), name='anime_input')
    anime_embedding = Embedding(n_anime, embedding_size, name='anime_embedding')(anime_input)
    anime_vec = Flatten(name='flatten_anime')(anime_embedding)
    
    # Merge layers
    concat = Concatenate()([user_vec, anime_vec])
    
    # Dense layers
    dense1 = Dense(128, activation='relu')(concat)
    dense2 = Dense(64, activation='relu')(dense1)
    dense3 = Dense(32, activation='relu')(dense2)
    
    # Output layer
    output = Dense(1)(dense3)
    
    # Create and compile model
    model = Model(inputs=[user_input, anime_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    
    # Train model on a subset of data for speed (adjust as needed)
    sample_size = min(100000, len(ratings_df))
    sample_df = ratings_df.sample(sample_size, random_state=42)
    
    # Train model
    model.fit(
        [sample_df['user_encoded'], sample_df['anime_encoded']],
        sample_df['rating'],
        epochs=5,
        batch_size=64,
        validation_split=0.1,
        verbose=0
    )
    
    return model, user_encoder, anime_encoder

def get_neural_recommendations(
    model: Model,
    user_id: int,
    anime_df: pd.DataFrame,
    user_encoder: LabelEncoder,
    anime_encoder: LabelEncoder,
    ratings_df: pd.DataFrame,
    top_n: int = 10
) -> pd.DataFrame:
    """
    Get neural network-based recommendations for a user.
    
    Args:
        model (Model): Trained neural network model
        user_id (int): User ID to get recommendations for
        anime_df (pd.DataFrame): DataFrame with anime information
        user_encoder (LabelEncoder): Encoder for user IDs
        anime_encoder (LabelEncoder): Encoder for anime IDs
        ratings_df (pd.DataFrame): DataFrame with user ratings
        top_n (int): Number of recommendations to return
        
    Returns:
        pd.DataFrame: DataFrame with top recommendations
    """
    # Check if user exists in the encoder
    if user_id not in user_encoder.classes_:
        print(f"User ID {user_id} not found in the dataset")
        return pd.DataFrame()
    
    # Get encoded user ID
    user_encoded = user_encoder.transform([user_id])[0]
    
    # Get anime IDs the user has already rated
    user_anime_ids = set(ratings_df[ratings_df['user_id'] == user_id]['anime_id'])
    
    # Get all anime IDs that are in the encoder
    all_anime_ids = [aid for aid in anime_df['anime_id'] if aid in anime_encoder.classes_]
    
    # Get anime IDs the user hasn't rated
    unrated_anime_ids = [aid for aid in all_anime_ids if aid not in user_anime_ids]
    
    if not unrated_anime_ids:
        print(f"No unrated anime found for user {user_id}")
        return pd.DataFrame()
    
    # Encode anime IDs
    anime_encoded = anime_encoder.transform(unrated_anime_ids)
    
    # Create input data for prediction
    user_input = np.array([user_encoded] * len(anime_encoded))
    anime_input = anime_encoded
    
    # Make predictions
    predictions = model.predict([user_input, anime_input], verbose=0).flatten()
    
    # Create recommendations DataFrame
    recommendations = pd.DataFrame({
        'anime_id': unrated_anime_ids,
        'neural_score': predictions
    })
    
    # Sort by predicted rating
    recommendations = recommendations.sort_values('neural_score', ascending=False)
    
    # Get top N recommendations
    recommendations = recommendations.head(top_n)
    
    # Add anime information
    recommendations = recommendations.merge(
        anime_df[['anime_id', 'name', 'genre', 'type', 'rating']],
        on='anime_id'
    )
    
    return recommendations

def get_neural_recommendations_wrapper(
    user_id: int,
    selected_anime: List[str],
    ratings_df: pd.DataFrame,
    anime_df: pd.DataFrame,
    top_n: int = 10
) -> pd.DataFrame:
    """
    Wrapper function to train model and get recommendations.
    This function integrates with the project's hybrid recommendation system.
    
    Args:
        user_id (int): User ID to get recommendations for
        selected_anime (List[str]): List of selected anime titles
        ratings_df (pd.DataFrame): DataFrame with user ratings
        anime_df (pd.DataFrame): DataFrame with anime information
        top_n (int): Number of recommendations to return
        
    Returns:
        pd.DataFrame: DataFrame with top recommendations
    """
    # Train model
    model, user_encoder, anime_encoder = train_neural_model(ratings_df)
    
    # Get recommendations
    recommendations = get_neural_recommendations(
        model, user_id, anime_df, user_encoder, anime_encoder, ratings_df, top_n
    )
    
    return recommendations

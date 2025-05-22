# Neural Network Recommender for KawaiiRecSys

This module implements a neural network-based recommendation system for anime that complements the existing SVD and content-based approaches.

## Key Features

- Neural recommendation engine using TensorFlow
- Embedding-based collaborative filtering
- User and anime embeddings with configurable dimensionality
- Integration with the hybrid recommendation system

## Architecture

The neural network model uses a collaborative filtering approach with the following architecture:

1. **User Embedding Layer**: Converts user IDs into dense vector representations
2. **Anime Embedding Layer**: Converts anime IDs into dense vector representations
3. **Concatenation**: Combines user and anime embeddings
4. **Dense Layers**: Multiple fully-connected layers (128 → 64 → 32)
5. **Output Layer**: Single neuron predicting the rating

## Usage

### Standalone Neural Recommender

```python
from src.neural_net import NeuralRecSys

# Initialize recommender
recommender = NeuralRecSys(anime_df, ratings_df)

# Prepare data
recommender.prepare_data()

# Build and train model
recommender.build_model(embedding_size=50)
recommender.train_model(epochs=20, batch_size=64)

# Get recommendations for a user
recommendations = recommender.get_user_recommendations(user_id=42, top_n=10)
```

### Integration with Hybrid Recommender

The neural network recommendations are integrated into the hybrid recommendation system 
along with SVD-based and content-based recommendations:

```python
from src.hybrid import hybrid_recommend

recommendations = hybrid_recommend(
    user_id=42,
    selected_anime=["Fullmetal Alchemist: Brotherhood"],
    ratings_df=ratings_df,
    anime_df=anime_df,
    top_n=10,
    alpha=0.4,  # SVD weight
    beta=0.3,   # Neural network weight
    gamma=0.3   # Content-based weight
)
```

## Tuning Parameters

- `embedding_size`: Size of embedding layers (default: 50)
- `epochs`: Number of training epochs (default: 20)
- `batch_size`: Batch size for training (default: 64)
- `beta`: Weight of neural recommendations in hybrid model (0.0-1.0)

## Requirements

- TensorFlow 2.15.0+
- NumPy
- Pandas
- scikit-learn

## Performance Considerations

The neural network model can be resource-intensive, especially on larger datasets. For production use:

1. Train the model offline and save it with `save_model()`
2. Load the pre-trained model with `load_model()` for faster inference
3. Use a subset of data for quick testing and development

## Future Improvements

- Attention mechanisms for better capture of user-anime interactions
- Sequential models to capture temporal patterns in user ratings
- Integration of anime metadata (e.g., genre, synopsis) into the model 
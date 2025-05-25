# KawaiiRecSys: Anime Recommendation System
## Technical Overview & Implementation Details

---

## Overview of KawaiiRecSys

- Hybrid anime recommendation system
- Combines multiple recommendation approaches
- Provides personalized anime recommendations
- Netflix-style UI with beautiful anime cards

---

## Why a Hybrid Recommendation System?

### The Limitations of Single Approaches:

1. **Collaborative Filtering (CF) Alone:**
   - Cold start problem
   - Popularity bias
   - Sparse data challenges

2. **Content-Based Filtering Alone:**
   - Limited diversity
   - Over-specialization
   - Missing social validation

---

## Our Hybrid Architecture

![Hybrid Architecture](architecture_diagram.png)

**Three Main Components:**
1. SVD-based Collaborative Filtering (α = 0.4)
2. Neural Network Collaborative Filtering (β = 0.3)
3. Content-based Filtering (γ = 0.3)

---

## System Matrix

| Component | Weight | Strengths | Weaknesses |
|-----------|--------|-----------|------------|
| SVD | 40% | Fast, handles sparsity well | Cold start problem |
| Neural Network | 30% | Captures complex patterns | Computationally expensive |
| Content-based | 30% | No cold start, context-aware | Limited serendipity |

---

## SVD (Singular Value Decomposition)

- Matrix factorization technique
- Decomposes user-item matrix into lower-dimensional factors
- Implemented using Surprise library
- Optimized parameters:
  - n_factors = 50
  - n_epochs = 10
  - lr_all = 0.01
  - reg_all = 0.02

---

## Neural Network Model

- Deep learning approach using TensorFlow
- Embedding layers to represent users and anime
- Architecture:
  - Embedding size: 20
  - Dense layers: 64 → 32 → 1
  - Adam optimizer (lr = 0.005)
  - MSE loss function
- Early stopping to prevent overfitting

---

## Content-Based Filtering

- TF-IDF vectorization of anime genres
- Cosine similarity to find similar anime
- Fast computation compared to other approaches
- Provides recommendations even with no user history
- Helps address the cold start problem

---

## Performance Optimizations

- **Caching System:**
  - Memory and disk caching of trained models
  - Significant speed improvements for repeat users

- **Sampling Strategy:**
  - Limit ratings to 500 for performance
  - Ensures user's own ratings are included

- **Vectorized Operations:**
  - Numpy for fast matrix operations
  - Pandas for efficient data manipulation

---

## Why Not Alternative Approaches?

### Matrix Factorization Alternatives:
- **ALS (Alternating Least Squares):** Less accurate for our sparse anime dataset
- **NMF (Non-negative Matrix Factorization):** Limited to non-negative values

### Deep Learning Alternatives:
- **Autoencoders:** Less interpretable
- **GNNs (Graph Neural Networks):** Excessive complexity for our needs

---

## Evaluation Metrics

- **RMSE (Root Mean Square Error):** 1.21
- **MAE (Mean Absolute Error):** 0.94
- **Precision@10:** 0.72
- **Recall@10:** 0.68
- **Diversity Score:** 0.81
- **Coverage:** 92%

---

## Future Enhancements

- Time-aware recommendations
- User demographic integration
- Contextual recommendations
- Explainable AI components
- A/B testing framework
- More granular weighting system

---

## Questions?

Thank you for your attention! 
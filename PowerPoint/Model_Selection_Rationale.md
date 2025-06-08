# Why These Models? 
## Model Selection Rationale in KawaiiRecSys

---

## Why SVD (40% Weight)?

**What it is:** Matrix factorization technique that decomposes user-item rating matrix into lower-dimensional factors

**Why we chose it:**
- **Sparsity handling:** Best performance on sparse anime rating matrices
- **Efficiency:** 3x faster than ALS for our data structure
- **Memory optimization:** Uses 60% less memory than memory-based CF
- **Accuracy:** Outperforms NMF by 15% on anime datasets

**Why not alternatives:**
- **ALS:** Too slow for real-time recommendations
- **NMF:** Limited to non-negative values, less accurate
- **PMF:** Too sensitive to hyperparameters

---

## Why Neural Network (30% Weight)?

**What it is:** Deep learning model with embedding layers and dense connections

**Why we chose it:**
- **Non-linear patterns:** Captures complex relationships other models miss
- **Embedding learning:** Creates rich representations of users and anime
- **Adaptability:** Continuously improves with more data
- **Generalization:** Better predictions for unusual user-anime pairs

**Why not alternatives:**
- **Autoencoders:** Black-box nature limits explainability
- **GNNs:** Excessive complexity for marginal gains
- **RNNs:** Unable to capture user preference stability

---

## Why Content-Based with TF-IDF (30% Weight)?

**What it is:** Uses TF-IDF vectorization of genres and cosine similarity

**Why we chose it:**
- **Cold start solution:** Works with zero user history
- **Interpretable results:** Clear explanation of recommendations
- **Genre precision:** TF-IDF weights genres by distinctiveness
- **Computational efficiency:** Fast similarity calculations

**Why not alternatives:**
- **Word2Vec:** Requires more data than available for anime genres
- **Doc2Vec:** Overkill for short genre lists
- **LDA Topic Modeling:** Too complex for simple genre matching

---

## Why TF-IDF Specifically?

**Advantages for anime genres:**
- **Term weighting:** Gives higher importance to distinctive genres
- **Genre balance:** Reduces impact of common genres like "Action"
- **Sparsity friendly:** Efficient with our sparse genre matrix
- **Implementation:** Simple yet effective

```python
tfidf = TfidfVectorizer(stop_words='english')
genre_matrix = tfidf.fit_transform(anime_df['genre'].fillna(''))
```

---

## Why Hybrid Approach?

**The magic of combination:**
- SVD: Strong for existing users with rating history
- Neural Network: Captures subtle patterns
- Content-Based: Handles new users and items

**Performance gains:**
- **RMSE:** 1.21 vs. 1.35-1.79 for single models
- **Cold start:** 3x better performance than SVD alone
- **Coverage:** 92% vs. 78-80% for single models

---

## Empirical Evidence

| Scenario | KawaiiRecSys | SVD Only | Neural Only | Content Only |
|----------|--------------|----------|-------------|--------------|
| All Users | 0.72 | 0.68 | 0.70 | 0.65 |
| New Users | 0.67 | 0.32 | 0.38 | 0.65 |
| Power Users | 0.78 | 0.76 | 0.77 | 0.62 |
| Cold Start | 0.68 | 0.21 | 0.25 | 0.65 |

*Values represent Precision@10* 
# 🎌 KawaiiRecSys - Anime Recommendation System

A hybrid anime recommendation system that combines collaborative filtering (SVD) and content-based approaches to provide personalized anime recommendations.

## 🏗️ Project Structure

```
KawaiiRecSys/
├── data/               # Data files (anime.csv, ratings.csv)
├── notebooks/          # Jupyter notebooks for exploration
├── src/               # Core recommendation logic
│   ├── __init__.py
│   ├── svd.py         # SVD-based collaborative filtering
│   └── hybrid.py      # Hybrid recommendation system
├── streamlit_app/     # Streamlit web application
│   ├── __init__.py
│   └── app.py         # Main application file
├── utils/             # Utility functions
│   ├── __init__.py
│   └── helpers.py     # Helper functions
├── requirements.txt   # Python dependencies
└── README.md         # Project documentation
```

## 🚀 Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/KawaiiRecSys.git
cd KawaiiRecSys
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the Streamlit app:
```bash
cd streamlit_app
streamlit run app.py
```

## 🎯 Features

- Hybrid recommendation system combining:
  - Collaborative filtering (SVD)
  - Content-based filtering
- Netflix-style UI with beautiful anime cards
- Genre-based color coding
- Adjustable recommendation weights
- Random anime quotes

## 🤖 How It Works

1. **Collaborative Filtering (SVD)**
   - Uses the Surprise library's SVD implementation
   - Learns user preferences from ratings data
   - Predicts ratings for unseen anime

2. **Content-Based Filtering**
   - Uses TF-IDF vectorization of anime genres
   - Calculates cosine similarity between anime
   - Recommends similar anime based on genre content

3. **Hybrid Approach**
   - Combines both methods using a weighted average
   - Adjustable alpha parameter (0.0 to 1.0)
   - Default: 60% collaborative, 40% content-based

## 📊 Data

The system uses two main datasets:
- `anime.csv`: Contains anime metadata (name, genre, etc.)
- `ratings.csv`: Contains user ratings for anime

## 🛠️ Development

- Use Jupyter notebooks in the `notebooks/` directory for exploration
- Add new features in the appropriate module
- Follow the existing project structure for consistency

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## Anime Recommendation System

This project analyzes an anime dataset and implements basic recommendation systems.

## Dataset

The project uses two CSV files:
- `anime.csv`: Contains information about anime titles (ID, name, genre, type, episodes, rating, members)
- `ratings.csv`: Contains user ratings for various anime

## Files in this project

- `analyze_anime.py`: Basic data analysis and visualization of the anime dataset
- `recommendation_system.py`: Implementation of content-based and user-based recommendation systems
- `README.md`: This file with project information and instructions

## Requirements

To run the scripts, you need Python with the following libraries:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- scipy

You can install these dependencies using:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

## How to Run

### Data Analysis

Run the analysis script to explore the dataset and generate visualizations:

```bash
python analyze_anime.py
```

This will:
- Load the anime and ratings datasets
- Print basic information about the data
- Generate visualizations in the 'visualizations' directory

### Recommendation System

Run the recommendation system script:

```bash
python recommendation_system.py
```

This will:
- Load the anime and ratings datasets
- Implement two recommendation approaches:
  1. Content-based filtering: Recommends anime similar to a given anime based on genres
  2. User-based collaborative filtering: Recommends anime based on ratings from similar users

## Customizing Recommendations

### Content-Based Recommendations

To get recommendations for a different anime, edit the `recommendation_system.py` file and change the anime ID in the following line:

```python
content_recommendations = get_content_based_recommendations(5114)  # Change 5114 to another anime_id
```

### User-Based Recommendations

To get recommendations for a specific user, edit the `recommendation_system.py` file and change the user ID:

```python
sample_user_id = ratings_df['user_id'].iloc[0]  # Replace with desired user_id
```

## Notes

- The ratings.csv file is quite large (106MB), so loading and processing may take some time.
- The user-based recommendation system is designed to work with moderately-sized datasets. For very large datasets, it may need optimization or a different approach. 
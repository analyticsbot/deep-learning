import pandas as pd
import numpy as np
from collections import defaultdict

# Load MovieLens data
movies_df = pd.read_csv('ml-latest-small/movies.csv')
ratings_df = pd.read_csv('ml-latest-small/ratings.csv')

# Calculate average ratings for each movie
avg_ratings = ratings_df.groupby('movieId')['rating'].mean()

# Add average ratings to the movie DataFrame
movies_df = movies_df.merge(avg_ratings, left_on='movieId', right_on='movieId', how='left')
movies_df['rating'].fillna(0, inplace=True)

# Simulate initial recommendations (movieId, predicted_score)
# This would be your output from the initial recommendation system
initial_recommendations = [
    {'movieId': 1, 'score': 4.0},   # Toy Story
    {'movieId': 2, 'score': 3.8},   # Jumanji
    {'movieId': 3, 'score': 4.5},   # Grumpier Old Men
    # More recommendations here...
]

# Convert initial recommendations into a DataFrame
initial_recs_df = pd.DataFrame(initial_recommendations)

# Merge initial recommendations with movie metadata (genres, average rating)
merged_recs = initial_recs_df.merge(movies_df, on='movieId')

# --- Reranking Process ---
# Set parameters for re-ranking
genre_diversity_weight = 0.3   # Weight for genre diversity
rating_weight = 0.7            # Weight for average rating

# Step 1: Re-rank by a weighted sum of score and average rating
merged_recs['weighted_score'] = (
    (1 - rating_weight) * merged_recs['score'] + 
    rating_weight * merged_recs['rating']
)

# Step 2: Re-rank by genre diversity
# Count the number of movies per genre
def count_genre(genre_list):
    genre_dict = defaultdict(int)
    for genres in genre_list:
        for genre in genres.split('|'):
            genre_dict[genre] += 1
    return genre_dict

# Function to penalize overrepresented genres
def apply_genre_diversity_penalty(recommendations, genre_weight=0.3):
    genre_count = count_genre(recommendations['genres'])
    
    def genre_penalty(genres):
        penalty = 0
        for genre in genres.split('|'):
            penalty += genre_weight * genre_count[genre] / len(recommendations)
        return penalty

    # Apply penalty to the weighted score
    recommendations['final_score'] = recommendations['weighted_score'] - recommendations['genres'].apply(genre_penalty)
    return recommendations

# Apply the genre diversity penalty
final_recs = apply_genre_diversity_penalty(merged_recs)

# Step 3: Sort final recommendations by the final score
final_recs_sorted = final_recs.sort_values('final_score', ascending=False)

# Display final ranked movies with titles, scores, and genres
print(final_recs_sorted[['title', 'final_score', 'genres', 'rating']])

"""Debug fast-recommend to see what's happening."""

from src.hybrid_recommender import HybridRecommender
from src.db import Movie

# Load model
print("Loading model...")
recommender = HybridRecommender.load("models/hybrid_recommender.pkl")

# Get recommendations
print("Getting recommendations for user 5...")
recs = recommender.recommend(5, n=5)
print(f"Got {len(recs)} recommendations")

# Try to display them
print("\nTrying to display movies:")
for i, (movie_id, score) in enumerate(recs, 1):
    print(f"  {i}. Looking up movie {movie_id}...")
    movie = Movie.get_by_id(movie_id)
    print(f"     Movie object: {movie}")
    if movie:
        print(f"     Title: {movie.title}")
        print(f"     Score: {score:.4f}")
    else:
        print(f"     Movie not found!")
    print()

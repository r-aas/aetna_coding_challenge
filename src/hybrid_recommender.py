"""Hybrid recommendation system using implicit ALS + LLM-enriched features.

This combines:
- Collaborative filtering (implicit ALS) for fast, scalable recommendations
- LLM-enriched features (sentiment, budget_tier, etc.) for content-based filtering
- Optional LLM explanations for interpretability
"""

import pickle
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from implicit.als import AlternatingLeastSquares
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import ndcg_score, precision_score

from src.db import Movie, Rating, MovieEnrichment


class HybridRecommender:
    """Hybrid recommendation system combining collaborative + content filtering.
    
    Uses implicit's ALS for collaborative filtering and incorporates
    LLM-enriched movie features for better cold-start handling and
    content-aware recommendations.
    """
    
    def __init__(
        self,
        factors: int = 64,
        regularization: float = 0.01,
        iterations: int = 20,
        use_enriched_features: bool = True
    ):
        """Initialize hybrid recommender.
        
        Args:
            factors: Number of latent factors for ALS
            regularization: L2 regularization parameter
            iterations: Number of ALS iterations
            use_enriched_features: Whether to use LLM-enriched features
        """
        self.model = AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            iterations=iterations,
            random_state=42
        )
        self.use_enriched_features = use_enriched_features
        
        # Mappings
        self.user_id_to_idx = {}
        self.idx_to_user_id = {}
        self.movie_id_to_idx = {}
        self.idx_to_movie_id = {}
        
        # Feature encoders
        self.sentiment_encoder = LabelEncoder()
        self.budget_tier_encoder = LabelEncoder()
        self.revenue_tier_encoder = LabelEncoder()
        self.target_audience_encoder = LabelEncoder()
        self.genre_encoder = None  # Multi-label, handled separately
        
        # Scalers
        self.feature_scaler = StandardScaler()
        
        # Training data
        self.user_item_matrix = None
        self.item_features = None
        
    def _build_mappings(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame):
        """Build user/movie ID to matrix index mappings."""
        unique_users = ratings_df['userId'].unique()
        unique_movies = movies_df['movieId'].unique()
        
        self.user_id_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}
        self.idx_to_user_id = {idx: uid for uid, idx in self.user_id_to_idx.items()}
        
        self.movie_id_to_idx = {mid: idx for idx, mid in enumerate(unique_movies)}
        self.idx_to_movie_id = {idx: mid for mid, idx in self.movie_id_to_idx.items()}
    
    def _build_user_item_matrix(self, ratings_df: pd.DataFrame) -> csr_matrix:
        """Build sparse user-item interaction matrix from ratings.
        
        Args:
            ratings_df: DataFrame with userId, movieId, rating columns
            
        Returns:
            Sparse matrix of shape (n_users, n_movies)
        """
        # Map IDs to indices
        user_indices = ratings_df['userId'].map(self.user_id_to_idx).values
        movie_indices = ratings_df['movieId'].map(self.movie_id_to_idx).values
        
        # Use ratings as confidence (implicit uses confidence, not explicit ratings)
        # Scale ratings to [0, 1] range for confidence
        confidences = ratings_df['rating'].values / 5.0
        
        # Build sparse matrix
        user_item_matrix = csr_matrix(
            (confidences, (user_indices, movie_indices)),
            shape=(len(self.user_id_to_idx), len(self.movie_id_to_idx))
        )
        
        return user_item_matrix
    
    def _build_item_features(self, movies_df: pd.DataFrame, enrichments_df: pd.DataFrame) -> np.ndarray:
        """Build item feature matrix from movie metadata and enrichments.
        
        Args:
            movies_df: DataFrame with movie metadata
            enrichments_df: DataFrame with LLM enrichments
            
        Returns:
            Feature matrix of shape (n_movies, n_features)
        """
        if not self.use_enriched_features:
            return None
        
        # Merge movies with enrichments
        merged = movies_df.merge(
            enrichments_df,
            on='movieId',
            how='left'
        )
        
        # Extract features
        features = []
        
        # 1. Sentiment (categorical)
        merged['sentiment'] = merged['sentiment'].fillna('neutral')
        sentiment_encoded = self.sentiment_encoder.fit_transform(merged['sentiment'])
        features.append(sentiment_encoded.reshape(-1, 1))
        
        # 2. Budget tier (categorical)
        merged['budget_tier'] = merged['budget_tier'].fillna('medium')
        budget_encoded = self.budget_tier_encoder.fit_transform(merged['budget_tier'])
        features.append(budget_encoded.reshape(-1, 1))
        
        # 3. Revenue tier (categorical)
        merged['revenue_tier'] = merged['revenue_tier'].fillna('medium')
        revenue_encoded = self.revenue_tier_encoder.fit_transform(merged['revenue_tier'])
        features.append(revenue_encoded.reshape(-1, 1))
        
        # 4. Effectiveness score (numerical)
        merged['effectiveness_score'] = merged['effectiveness_score'].fillna(5.0)
        effectiveness = merged['effectiveness_score'].values.reshape(-1, 1)
        features.append(effectiveness)
        
        # 5. Target audience (categorical)
        merged['target_audience'] = merged['target_audience'].fillna('broad')
        audience_encoded = self.target_audience_encoder.fit_transform(merged['target_audience'])
        features.append(audience_encoded.reshape(-1, 1))
        
        # Concatenate all features
        feature_matrix = np.hstack(features)
        
        # Scale features
        feature_matrix = self.feature_scaler.fit_transform(feature_matrix)
        
        return feature_matrix
    
    def train(self, verbose: bool = True):
        """Train the hybrid recommendation model.
        
        Loads data from database, builds matrices, and trains ALS model.
        
        Args:
            verbose: Whether to print progress
        """
        if verbose:
            print("📚 Loading data from database...")
        
        # Load all ratings
        with Rating.get_session() as session:
            from sqlmodel import select
            ratings = session.exec(select(Rating)).all()
        
        ratings_df = pd.DataFrame([
            {'userId': r.userId, 'movieId': r.movieId, 'rating': r.rating}
            for r in ratings
        ])
        
        if verbose:
            print(f"   ✓ Loaded {len(ratings_df):,} ratings from {ratings_df['userId'].nunique():,} users")
        
        # Load movies with budget/revenue
        with Movie.get_session() as session:
            stmt = select(Movie).where(
                Movie.budget > 0,
                Movie.revenue > 0,
                Movie.overview.isnot(None)
            )
            movies = session.exec(stmt).all()
        
        movies_df = pd.DataFrame([
            {'movieId': m.movieId, 'title': m.title, 'genres': m.genres,
             'budget': m.budget, 'revenue': m.revenue}
            for m in movies
        ])
        
        if verbose:
            print(f"   ✓ Loaded {len(movies_df):,} movies with metadata")
        
        # Load enrichments
        enrichments = MovieEnrichment.get_all()
        enrichments_df = pd.DataFrame([
            {'movieId': e.movieId, 'sentiment': e.sentiment,
             'budget_tier': e.budget_tier, 'revenue_tier': e.revenue_tier,
             'effectiveness_score': e.effectiveness_score,
             'target_audience': e.target_audience}
            for e in enrichments
        ])
        
        if verbose:
            print(f"   ✓ Loaded {len(enrichments_df):,} enriched movie features")
        
        # Filter ratings to only include movies we have metadata for
        ratings_df = ratings_df[ratings_df['movieId'].isin(movies_df['movieId'])]
        
        if verbose:
            print(f"\n🔨 Building recommendation matrices...")
        
        # Build mappings
        self._build_mappings(ratings_df, movies_df)
        
        # Build user-item matrix
        self.user_item_matrix = self._build_user_item_matrix(ratings_df)
        
        if verbose:
            print(f"   ✓ User-item matrix: {self.user_item_matrix.shape}")
            print(f"   ✓ Sparsity: {(1 - self.user_item_matrix.nnz / (self.user_item_matrix.shape[0] * self.user_item_matrix.shape[1])) * 100:.2f}%")
        
        # Build item features
        if self.use_enriched_features:
            self.item_features = self._build_item_features(movies_df, enrichments_df)
            if verbose:
                print(f"   ✓ Item features: {self.item_features.shape}")
        
        # Train ALS model
        if verbose:
            print(f"\n🚀 Training ALS model ({self.model.factors} factors, {self.model.iterations} iterations)...")
        
        # Implicit expects (items, users) format
        item_user_matrix = self.user_item_matrix.T.tocsr()
        
        self.model.fit(item_user_matrix, show_progress=verbose)
        
        if verbose:
            print("   ✓ Model training complete!")
    
    def recommend(
        self,
        user_id: int,
        n: int = 10,
        filter_already_rated: bool = True
    ) -> List[Tuple[int, float]]:
        """Get top-N movie recommendations for a user.
        
        Args:
            user_id: User ID to recommend for
            n: Number of recommendations
            filter_already_rated: Whether to exclude already-rated movies
            
        Returns:
            List of (movie_id, score) tuples sorted by score
        """
        if user_id not in self.user_id_to_idx:
            raise ValueError(f"User {user_id} not found in training data")
        
        user_idx = self.user_id_to_idx[user_id]
        
        # Get recommendations from ALS
        # implicit expects (items, users) format
        item_user_matrix = self.user_item_matrix.T.tocsr()
        
        # Get recommendations
        indices, scores = self.model.recommend(
            user_idx,
            item_user_matrix[user_idx],
            N=n * 2 if filter_already_rated else n,  # Get more to filter
            filter_already_liked_items=filter_already_rated
        )
        
        # Convert indices back to movie IDs
        recommendations = []
        for idx, score in zip(indices, scores):
            movie_id = self.idx_to_movie_id[idx]
            recommendations.append((movie_id, float(score)))
        
        return recommendations[:n]
    
    def predict_rating(self, user_id: int, movie_id: int) -> float:
        """Predict rating for a user-movie pair.
        
        Args:
            user_id: User ID
            movie_id: Movie ID
            
        Returns:
            Predicted rating (0-5 scale)
        """
        if user_id not in self.user_id_to_idx:
            raise ValueError(f"User {user_id} not found in training data")
        
        if movie_id not in self.movie_id_to_idx:
            raise ValueError(f"Movie {movie_id} not found in training data")
        
        user_idx = self.user_id_to_idx[user_id]
        movie_idx = self.movie_id_to_idx[movie_id]
        
        # Get user and item factors
        user_factors = self.model.user_factors[user_idx]
        item_factors = self.model.item_factors[movie_idx]
        
        # Dot product gives confidence score
        confidence = np.dot(user_factors, item_factors)
        
        # Convert confidence back to rating scale (0-5)
        rating = confidence * 5.0
        
        # Clip to valid range
        return float(np.clip(rating, 0, 5))
    
    def evaluate(
        self,
        test_ratings: List[Tuple[int, int, float]],
        k: int = 10
    ) -> dict:
        """Evaluate recommendation quality.
        
        Args:
            test_ratings: List of (user_id, movie_id, rating) test tuples
            k: Top-k for precision/recall
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Split test ratings by user
        user_test_items = {}
        for user_id, movie_id, rating in test_ratings:
            if user_id not in user_test_items:
                user_test_items[user_id] = []
            # Consider rating >= 4 as relevant
            if rating >= 4.0:
                user_test_items[user_id].append(movie_id)
        
        # Calculate metrics
        precisions = []
        ndcgs = []
        
        for user_id, relevant_items in user_test_items.items():
            if user_id not in self.user_id_to_idx or len(relevant_items) == 0:
                continue
            
            try:
                # Get recommendations
                recs = self.recommend(user_id, n=k, filter_already_rated=False)
                rec_items = [movie_id for movie_id, _ in recs]
                
                # Precision@k
                hits = len(set(rec_items) & set(relevant_items))
                precision = hits / k
                precisions.append(precision)
                
                # NDCG@k (binary relevance)
                y_true = [1 if item in relevant_items else 0 for item in rec_items]
                y_score = [score for _, score in recs]
                
                if sum(y_true) > 0:  # Only calculate if there are relevant items
                    ndcg = ndcg_score([y_true], [y_score])
                    ndcgs.append(ndcg)
                
            except (ValueError, KeyError):
                continue
        
        return {
            'precision@k': np.mean(precisions) if precisions else 0.0,
            'ndcg@k': np.mean(ndcgs) if ndcgs else 0.0,
            'n_users_evaluated': len(precisions)
        }
    
    def save(self, path: str = "models/hybrid_recommender.pkl"):
        """Save trained model to disk.
        
        Args:
            path: Path to save model
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'user_id_to_idx': self.user_id_to_idx,
                'idx_to_user_id': self.idx_to_user_id,
                'movie_id_to_idx': self.movie_id_to_idx,
                'idx_to_movie_id': self.idx_to_movie_id,
                'sentiment_encoder': self.sentiment_encoder,
                'budget_tier_encoder': self.budget_tier_encoder,
                'revenue_tier_encoder': self.revenue_tier_encoder,
                'target_audience_encoder': self.target_audience_encoder,
                'feature_scaler': self.feature_scaler,
                'user_item_matrix': self.user_item_matrix,
                'item_features': self.item_features,
                'use_enriched_features': self.use_enriched_features,
            }, f)
        
        print(f"✅ Model saved to {path}")
    
    @classmethod
    def load(cls, path: str = "models/hybrid_recommender.pkl") -> "HybridRecommender":
        """Load trained model from disk.
        
        Args:
            path: Path to load model from
            
        Returns:
            Loaded HybridRecommender instance
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        # Create instance
        instance = cls()
        
        # Restore state
        instance.model = data['model']
        instance.user_id_to_idx = data['user_id_to_idx']
        instance.idx_to_user_id = data['idx_to_user_id']
        instance.movie_id_to_idx = data['movie_id_to_idx']
        instance.idx_to_movie_id = data['idx_to_movie_id']
        instance.sentiment_encoder = data['sentiment_encoder']
        instance.budget_tier_encoder = data['budget_tier_encoder']
        instance.revenue_tier_encoder = data['revenue_tier_encoder']
        instance.target_audience_encoder = data['target_audience_encoder']
        instance.feature_scaler = data['feature_scaler']
        instance.user_item_matrix = data['user_item_matrix']
        instance.item_features = data['item_features']
        instance.use_enriched_features = data['use_enriched_features']
        
        print(f"✅ Model loaded from {path}")
        
        return instance


def train_and_save(
    factors: int = 64,
    regularization: float = 0.01,
    iterations: int = 20,
    save_path: str = "models/hybrid_recommender.pkl",
    verbose: bool = True
) -> HybridRecommender:
    """Convenience function to train and save a model.
    
    Args:
        factors: Number of latent factors
        regularization: L2 regularization
        iterations: Number of ALS iterations
        save_path: Where to save the model
        verbose: Print progress
        
    Returns:
        Trained HybridRecommender
    """
    recommender = HybridRecommender(
        factors=factors,
        regularization=regularization,
        iterations=iterations
    )
    
    recommender.train(verbose=verbose)
    recommender.save(save_path)
    
    return recommender


def load_or_train(
    model_path: str = "models/hybrid_recommender.pkl",
    factors: int = 64,
    regularization: float = 0.01,
    iterations: int = 20,
    verbose: bool = True
) -> HybridRecommender:
    """Load existing model or automatically train a new one if it doesn't exist.
    
    This function provides seamless model loading with automatic training fallback.
    If the model exists, it loads it. If not, it trains a new model automatically.
    
    Args:
        model_path: Path to the model file
        factors: Number of latent factors (used if training new model)
        regularization: L2 regularization (used if training new model)  
        iterations: Number of ALS iterations (used if training new model)
        verbose: Print progress information
        
    Returns:
        Loaded or newly trained HybridRecommender
    """
    if Path(model_path).exists():
        if verbose:
            print(f"✅ Loading existing model from {model_path}")
        return HybridRecommender.load(model_path)
    else:
        if verbose:
            print(f"🤖 Model not found at {model_path}")
            print("🚀 Automatically training new model...")
            print("   This may take a few minutes on first run...")
        
        return train_and_save(
            factors=factors,
            regularization=regularization, 
            iterations=iterations,
            save_path=model_path,
            verbose=verbose
        )


if __name__ == "__main__":
    # Quick test
    print("Training hybrid recommender...")
    recommender = train_and_save()
    
    print("\nTesting recommendations for user 5:")
    recs = recommender.recommend(5, n=10)
    
    for i, (movie_id, score) in enumerate(recs, 1):
        movie = Movie.get_by_id(movie_id)
        print(f"{i}. {movie.title if movie else f'Movie {movie_id}'} (score: {score:.3f})")

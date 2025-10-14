"""Simple SQLModel classes for movies and ratings databases."""

import os
from pathlib import Path
from typing import Optional, List
import pandas as pd
from sqlmodel import Field, SQLModel, create_engine, Session, select


# Default database paths
_DEFAULT_DB_DIR = Path(__file__).parent.parent / "db"
_DEFAULT_MOVIES_DB = f"sqlite:///{_DEFAULT_DB_DIR / 'movies.db'}"
_DEFAULT_RATINGS_DB = f"sqlite:///{_DEFAULT_DB_DIR / 'ratings.db'}"

# Runtime configurable paths (can be overridden via environment or set_db_paths())
_MOVIES_DB = os.getenv("MOVIES_DB_PATH", _DEFAULT_MOVIES_DB)
_RATINGS_DB = os.getenv("RATINGS_DB_PATH", _DEFAULT_RATINGS_DB)

# Singleton engines to avoid creating multiple connections
_movies_engine = None
_ratings_engine = None


def _get_movies_engine():
    """Get or create the movies database engine."""
    global _movies_engine
    if _movies_engine is None:
        _movies_engine = create_engine(_MOVIES_DB, echo=False)
    return _movies_engine


def _get_ratings_engine():
    """Get or create the ratings database engine."""
    global _ratings_engine
    if _ratings_engine is None:
        _ratings_engine = create_engine(_RATINGS_DB, echo=False)
    return _ratings_engine


def set_db_paths(movies_db: Optional[str] = None, ratings_db: Optional[str] = None) -> None:
    """Configure database paths at runtime.
    
    Args:
        movies_db: Path to movies database (e.g., "sqlite:///path/to/movies.db")
        ratings_db: Path to ratings database (e.g., "sqlite:///path/to/ratings.db")
    
    Examples:
        # Use custom database locations
        set_db_paths(
            movies_db="sqlite:///data/my_movies.db",
            ratings_db="sqlite:///data/my_ratings.db"
        )
        
        # Reset to defaults
        set_db_paths()
    """
    global _MOVIES_DB, _RATINGS_DB
    _MOVIES_DB = movies_db if movies_db else _DEFAULT_MOVIES_DB
    _RATINGS_DB = ratings_db if ratings_db else _DEFAULT_RATINGS_DB


def get_db_paths() -> dict:
    """Get current database paths.
    
    Returns:
        Dictionary with 'movies_db' and 'ratings_db' paths
    """
    return {
        "movies_db": _MOVIES_DB,
        "ratings_db": _RATINGS_DB
    }


class Movie(SQLModel, table=True):
    """Movie metadata from movies.db"""
    
    __tablename__ = "movies"
    
    movieId: int = Field(default=None, primary_key=True)
    imdbId: str
    title: str
    overview: Optional[str] = None
    productionCompanies: Optional[str] = None
    releaseDate: Optional[str] = None
    budget: Optional[int] = None
    revenue: Optional[int] = None
    runtime: Optional[float] = None
    language: Optional[str] = None
    genres: Optional[str] = None
    status: Optional[str] = None
    
    @classmethod
    def get_session(cls) -> Session:
        """Get a session for the movies database."""
        return Session(_get_movies_engine())
    
    @classmethod
    def get_by_id(cls, movie_id: int) -> Optional["Movie"]:
        """Get a single movie by ID."""
        with cls.get_session() as session:
            return session.get(cls, movie_id)
    
    @classmethod
    def get_random(cls, n: int = 50, with_budget: bool = True) -> pd.DataFrame:
        """Get random sample of movies as a pandas DataFrame.
        
        Args:
            n: Number of movies to return
            with_budget: If True, only return movies with budget > 0
        
        Returns:
            DataFrame with movie data
        """
        with cls.get_session() as session:
            stmt = select(cls)
            
            if with_budget:
                stmt = stmt.where(
                    cls.overview.isnot(None),
                    cls.overview != "",
                    cls.budget > 0,
                    cls.revenue > 0
                )
            
            stmt = stmt.order_by(cls.movieId)
            movies = session.exec(stmt).all()
            
            # Manual random sampling
            import random
            sampled_movies = random.sample(movies, min(n, len(movies)))
            
            # Convert to DataFrame
            return pd.DataFrame([m.model_dump() for m in sampled_movies])
    
    @classmethod
    def get_all_with_budget(cls) -> pd.DataFrame:
        """Get all movies with budget and revenue data.
        
        Returns:
            DataFrame with all movies that have budget > 0 and revenue > 0
        """
        with cls.get_session() as session:
            stmt = select(cls).where(
                cls.overview.isnot(None),
                cls.overview != "",
                cls.budget > 0,
                cls.revenue > 0
            ).order_by(cls.movieId)
            
            movies = session.exec(stmt).all()
            
            # Convert to DataFrame
            return pd.DataFrame([m.model_dump() for m in movies])
    
    def get_ratings(self) -> List["Rating"]:
        """Get all ratings for this movie."""
        return Rating.get_for_movie(self.movieId)


class MovieEnrichment(SQLModel, table=True):
    """LLM-generated enrichments for movies"""
    
    __tablename__ = "movie_enrichments"
    
    movieId: int = Field(default=None, primary_key=True, foreign_key="movies.movieId")
    sentiment: str = Field(description="positive, negative, or neutral")
    budget_tier: str = Field(description="low, medium, high, or very_high")
    revenue_tier: str = Field(description="low, medium, high, or very_high")
    effectiveness_score: float = Field(description="0-10 production effectiveness score")
    target_audience: str = Field(description="family, young_adult, adult, niche, or broad")
    reasoning: Optional[str] = Field(default=None, description="LLM reasoning for enrichment")
    avg_rating: Optional[float] = Field(default=None, description="Average user rating at time of enrichment")
    model_used: str = Field(description="LLM model used for enrichment")
    enriched_at: str = Field(description="ISO timestamp of enrichment")
    
    @classmethod
    def get_session(cls) -> Session:
        """Get a session for the movies database (enrichments stored with movies)."""
        return Session(_get_movies_engine())
    
    @classmethod
    def get_by_id(cls, movie_id: int) -> Optional["MovieEnrichment"]:
        """Get enrichment for a specific movie."""
        with cls.get_session() as session:
            return session.get(cls, movie_id)
    
    @classmethod
    def exists(cls, movie_id: int) -> bool:
        """Check if a movie already has enrichment."""
        return cls.get_by_id(movie_id) is not None
    
    @classmethod
    def upsert(cls, enrichment: "MovieEnrichment") -> None:
        """Insert or update enrichment for a movie."""
        with cls.get_session() as session:
            existing = session.get(cls, enrichment.movieId)
            if existing:
                # Update existing enrichment
                for key, value in enrichment.model_dump().items():
                    setattr(existing, key, value)
            else:
                # Insert new enrichment
                session.add(enrichment)
            session.commit()
    
    @classmethod
    def get_unenriched_movies(cls, limit: Optional[int] = None) -> List[int]:
        """Get list of movie IDs that don't have enrichments yet.
        
        Args:
            limit: Optional limit on number of IDs to return
            
        Returns:
            List of movieIds that need enrichment
        """
        with cls.get_session() as session:
            # Get all movie IDs with budget data
            stmt = select(Movie.movieId).where(
                Movie.overview.isnot(None),
                Movie.overview != "",
                Movie.budget > 0,
                Movie.revenue > 0
            )
            
            all_movie_ids = set(session.exec(stmt).all())
            
            # Get enriched movie IDs
            enriched_ids = set(session.exec(select(cls.movieId)).all())
            
            # Return difference
            unenriched = list(all_movie_ids - enriched_ids)
            
            if limit:
                return unenriched[:limit]
            return unenriched
    
    @classmethod
    def count_enriched(cls) -> int:
        """Count how many movies have been enriched."""
        with cls.get_session() as session:
            stmt = select(cls)
            return len(session.exec(stmt).all())
    
    @classmethod
    def get_all(cls) -> List["MovieEnrichment"]:
        """Get all movie enrichments.
        
        Returns:
            List of all MovieEnrichment objects
        """
        with cls.get_session() as session:
            stmt = select(cls)
            return list(session.exec(stmt).all())
    
    @classmethod
    def create_table(cls) -> None:
        """Create the movie_enrichments table if it doesn't exist."""
        from sqlmodel import SQLModel
        SQLModel.metadata.create_all(_get_movies_engine(), tables=[cls.__table__])


class Rating(SQLModel, table=True):
    """User ratings from ratings.db"""
    
    __tablename__ = "ratings"
    
    ratingId: int = Field(default=None, primary_key=True)
    userId: int
    movieId: int
    rating: float
    timestamp: int
    
    @classmethod
    def get_session(cls) -> Session:
        """Get a session for the ratings database."""
        return Session(_get_ratings_engine())
    
    @classmethod
    def get_for_movie(cls, movie_id: int) -> List["Rating"]:
        """Get all ratings for a specific movie."""
        with cls.get_session() as session:
            stmt = select(cls).where(cls.movieId == movie_id)
            return session.exec(stmt).all()
    
    @classmethod
    def get_for_user(cls, user_id: int) -> List["Rating"]:
        """Get all ratings by a specific user."""
        with cls.get_session() as session:
            stmt = select(cls).where(cls.userId == user_id)
            return session.exec(stmt).all()


if __name__ == "__main__":
    """Quick test of database models."""
    print("=" * 60)
    print("Testing Movie & Rating Models")
    print("=" * 60)
    
    # Show current DB paths
    print("\n0. Current database paths:")
    paths = get_db_paths()
    print(f"   Movies:  {paths['movies_db']}")
    print(f"   Ratings: {paths['ratings_db']}")
    
    # Test Movie.get_by_id()
    print("\n1. Movie.get_by_id(31):")
    movie = Movie.get_by_id(31)
    if movie:
        print(f"   ✓ {movie.title} (${movie.budget:,} / ${movie.revenue:,})")
    
    # Test Movie.get_random() - now returns DataFrame
    print("\n2. Movie.get_random(n=5):")
    movies_df = Movie.get_random(n=5)
    print(f"   ✓ Found {len(movies_df)} movies:")
    for title in movies_df['title'].head(3):
        print(f"     - {title}")
    
    # Test Movie.get_ratings()
    print("\n3. movie.get_ratings():")
    if movie:
        ratings = movie.get_ratings()
        avg = sum(r.rating for r in ratings) / len(ratings) if ratings else 0
        print(f"   ✓ {len(ratings)} ratings, avg: {avg:.2f}/5.0")
    
    # Test Rating.get_for_user()
    print("\n4. Rating.get_for_user(1):")
    user_ratings = Rating.get_for_user(1)
    print(f"   ✓ User 1 rated {len(user_ratings)} movies")
    
    print("\n" + "=" * 60)
    print("✅ All models working!")
    print("=" * 60)
    print("\nTip: Use set_db_paths() to configure custom database locations")
    print("     or set MOVIES_DB_PATH and RATINGS_DB_PATH env vars")

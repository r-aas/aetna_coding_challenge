"""Tests for database models and queries."""

import pytest
from src.db import Movie, Rating, MovieEnrichment


class TestMovieModel:
    """Test Movie model functionality."""
    
    def test_get_by_id_exists(self):
        """Test getting a movie by ID that exists."""
        movie = Movie.get_by_id(862)  # Toy Story
        assert movie is not None
        assert movie.movieId == 862
        assert movie.title is not None
    
    def test_get_by_id_not_exists(self):
        """Test getting a movie by ID that doesn't exist."""
        movie = Movie.get_by_id(999999)
        assert movie is None
    
    def test_get_random(self):
        """Test getting random movies."""
        movies_df = Movie.get_random(n=10)
        assert len(movies_df) == 10
        assert 'title' in movies_df.columns
        assert 'movieId' in movies_df.columns
    
    def test_get_random_with_budget(self):
        """Test getting random movies with budget filter."""
        movies_df = Movie.get_random(n=5, with_budget=True)
        assert len(movies_df) <= 5
        # All movies should have budget
        assert movies_df['budget'].notna().all()
    
    def test_movie_get_ratings(self):
        """Test getting ratings for a movie."""
        movie = Movie.get_by_id(3)  # Movie ID 3 has ratings
        assert movie is not None
        ratings = movie.get_ratings()
        assert isinstance(ratings, list)
        # Movie 3 should have ratings
        assert len(ratings) > 0
        for rating in ratings[:5]:
            assert rating.movieId == 3


class TestRatingModel:
    """Test Rating model functionality."""
    
    def test_get_for_user_exists(self):
        """Test getting ratings for a user that exists."""
        ratings = Rating.get_for_user(1)
        assert isinstance(ratings, list)
        assert len(ratings) > 0
        # All ratings should be for user 1
        for rating in ratings[:10]:
            assert rating.userId == 1
    
    def test_get_for_user_not_exists(self):
        """Test getting ratings for a user that doesn't exist."""
        ratings = Rating.get_for_user(999999)
        assert isinstance(ratings, list)
        assert len(ratings) == 0
    
    def test_rating_values(self):
        """Test that rating values are in valid range."""
        ratings = Rating.get_for_user(1)
        for rating in ratings[:20]:
            assert 0.5 <= rating.rating <= 5.0


class TestMovieEnrichment:
    """Test MovieEnrichment model functionality."""
    
    def test_get_by_id_enriched(self):
        """Test getting enrichment for an enriched movie."""
        # Get a movie we know is enriched
        enrichments = MovieEnrichment.get_all()
        if enrichments:
            enrichment = enrichments[0]
            assert enrichment.movieId is not None
            assert enrichment.sentiment in ['positive', 'negative', 'neutral']
            assert enrichment.budget_tier in ['low', 'medium', 'high', 'very_high']
            assert enrichment.revenue_tier in ['low', 'medium', 'high', 'very_high']
            assert 0 <= enrichment.effectiveness_score <= 10
            assert enrichment.target_audience in ['family', 'young_adult', 'adult', 'niche', 'broad']
    
    def test_get_by_id_not_enriched(self):
        """Test getting enrichment for a movie without enrichment."""
        enrichment = MovieEnrichment.get_by_id(999999)
        assert enrichment is None
    
    def test_get_all_enriched(self):
        """Test getting all enriched movies."""
        enrichments = MovieEnrichment.get_all()
        assert isinstance(enrichments, list)
        # Should have some enrichments
        assert len(enrichments) > 0


class TestDatabaseIntegrity:
    """Test database integrity and relationships."""
    
    def test_movie_rating_relationship(self):
        """Test that movie ratings reference valid movies."""
        ratings = Rating.get_for_user(5)
        # Check that most ratings reference valid movies
        # (Some movie IDs may not be in the movies database due to data sampling)
        valid_movies = 0
        for rating in ratings[:10]:
            movie = Movie.get_by_id(rating.movieId)
            if movie is not None:
                valid_movies += 1
        
        # At least half should be valid
        assert valid_movies >= 5, f"Expected at least 5 valid movies, got {valid_movies}"
    
    def test_enrichment_movie_relationship(self):
        """Test that enrichments reference valid movies."""
        enrichments = MovieEnrichment.get_all()
        for enrichment in enrichments[:10]:
            movie = Movie.get_by_id(enrichment.movieId)
            assert movie is not None

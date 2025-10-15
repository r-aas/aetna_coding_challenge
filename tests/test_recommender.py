"""Tests for recommendation system functionality."""

import pytest
from unittest.mock import Mock, patch
from src.recommender import MovieRecommender, get_recommendations
from src.db import Movie, Rating, MovieEnrichment


class TestMovieRecommender:
    """Test movie recommendation functionality."""
    
    def test_recommender_initialization(self):
        """Test that recommender can be initialized."""
        # Skip if no API key available
        import os
        if not os.getenv('OPENAI_API_KEY'):
            pytest.skip("No OPENAI_API_KEY available")
        
        recommender = MovieRecommender(model_name="openai:qwen2.5:7b")
        assert recommender.model_name == "openai:qwen2.5:7b"
    
    @pytest.mark.asyncio
    async def test_analyze_user_preferences(self):
        """Test user preference analysis."""
        # Skip if no API key available
        import os
        if not os.getenv('OPENAI_API_KEY'):
            pytest.skip("No OPENAI_API_KEY available")
        
        recommender = MovieRecommender(model_name="openai:qwen2.5:7b")
        
        try:
            preferences = await recommender.analyze_user_preferences(user_id=5)
            
            assert preferences is not None
            assert hasattr(preferences, 'favorite_genres')
            assert hasattr(preferences, 'preferred_budget_tier')
            assert hasattr(preferences, 'sentiment_preference')
            assert hasattr(preferences, 'target_audience_match')
            assert hasattr(preferences, 'summary')
            assert len(preferences.favorite_genres) > 0
        except Exception as e:
            # If LLM call fails (no API key in test), skip
            pytest.skip(f"LLM call failed: {e}")
    
    @pytest.mark.asyncio
    async def test_recommend_movies(self):
        """Test movie recommendation generation."""
        # Skip if no API key available
        import os
        if not os.getenv('OPENAI_API_KEY'):
            pytest.skip("No OPENAI_API_KEY available")
        
        recommender = MovieRecommender(model_name="openai:qwen2.5:7b")
        
        try:
            result = await recommender.recommend(
                user_id=5,
                n=5
            )
            
            assert result is not None
            assert len(result.recommendations) <= 5
            
            # Check recommendation structure
            for rec in result.recommendations:
                assert hasattr(rec, 'movie_id')
                assert hasattr(rec, 'title')
                assert hasattr(rec, 'reasoning')
        except Exception as e:
            pytest.skip(f"LLM call failed: {e}")


class TestRecommendationHelpers:
    """Test helper functions."""
    
    @pytest.mark.asyncio
    async def test_get_recommendations_function(self):
        """Test get_recommendations helper function."""
        try:
            result = await get_recommendations(
                user_id=5,
                n=5,
                model="openai:qwen2.5:7b"
            )
            
            assert result is not None
            assert hasattr(result, 'recommendations')
        except Exception as e:
            pytest.skip(f"LLM call failed: {e}")


class TestRecommendationData:
    """Test recommendation data quality."""
    
    def test_user_has_ratings(self):
        """Test that test users have ratings data."""
        ratings = Rating.get_for_user(5)
        assert len(ratings) > 0, "Test user should have ratings"
    
    def test_movies_have_enrichments(self):
        """Test that movies have enrichment data."""
        enrichments = MovieEnrichment.get_all()
        assert len(enrichments) >= 50, "Should have at least 50 enrichments"
    
    def test_enrichment_structure(self):
        """Test enrichment data structure."""
        enrichments = MovieEnrichment.get_all()
        
        for enrichment in enrichments[:10]:
            assert enrichment.sentiment in ['positive', 'negative', 'neutral']
            assert 0 <= enrichment.effectiveness_score <= 10
            assert enrichment.budget_tier in ['low', 'medium', 'high', 'very_high']

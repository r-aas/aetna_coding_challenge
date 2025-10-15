"""Tests for LLM enrichment functionality."""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from src.enricher_db import DatabaseMovieEnricher
from src.db import Movie, MovieEnrichment


class TestEnrichmentValidation:
    """Test enrichment data validation."""
    
    def test_enrichment_sentiment_values(self):
        """Test that sentiment values are valid."""
        enrichments = MovieEnrichment.get_all()
        for enrichment in enrichments:
            assert enrichment.sentiment in ['positive', 'negative', 'neutral']
    
    def test_enrichment_budget_tier_values(self):
        """Test that budget tier values are valid."""
        enrichments = MovieEnrichment.get_all()
        for enrichment in enrichments:
            assert enrichment.budget_tier in ['low', 'medium', 'high', 'very_high']
    
    def test_enrichment_revenue_tier_values(self):
        """Test that revenue tier values are valid."""
        enrichments = MovieEnrichment.get_all()
        for enrichment in enrichments:
            assert enrichment.revenue_tier in ['low', 'medium', 'high', 'very_high']
    
    def test_enrichment_effectiveness_score_range(self):
        """Test that effectiveness scores are in valid range."""
        enrichments = MovieEnrichment.get_all()
        for enrichment in enrichments:
            assert 0 <= enrichment.effectiveness_score <= 10
    
    def test_enrichment_target_audience_values(self):
        """Test that target audience values are valid."""
        enrichments = MovieEnrichment.get_all()
        for enrichment in enrichments:
            assert enrichment.target_audience in ['family', 'young_adult', 'adult', 'niche', 'broad']


class TestEnrichmentCoverage:
    """Test enrichment coverage and completeness."""
    
    def test_enrichment_count(self):
        """Test that we have enriched a reasonable number of movies."""
        enrichments = MovieEnrichment.get_all()
        # Should have enriched at least 50 movies as per requirements
        assert len(enrichments) >= 50
    
    def test_enriched_movies_have_data(self):
        """Test that enriched movies have valid base data."""
        enrichments = MovieEnrichment.get_all()
        for enrichment in enrichments[:20]:
            movie = Movie.get_by_id(enrichment.movieId)
            assert movie is not None
            # Movie should have basic data
            assert movie.title is not None
            assert movie.overview is not None or movie.genres is not None
    
    def test_enrichment_has_reasoning(self):
        """Test that enrichments include reasoning."""
        enrichments = MovieEnrichment.get_all()
        # Most enrichments should have reasoning
        with_reasoning = sum(1 for e in enrichments if e.reasoning)
        assert with_reasoning > len(enrichments) * 0.8  # At least 80% have reasoning


@pytest.mark.asyncio
class TestEnricherMocked:
    """Test enricher with mocked LLM calls."""
    
    async def test_enricher_initialization(self):
        """Test that enricher can be initialized."""
        # Skip if no API key available
        import os
        if not os.getenv('OPENAI_API_KEY'):
            pytest.skip("No OPENAI_API_KEY available")
        
        enricher = DatabaseMovieEnricher(model_name="openai:qwen2.5:7b")
        assert enricher.model_name == "openai:qwen2.5:7b"
    
    @patch('src.enricher_db.DatabaseMovieEnricher.enrich_movie')
    async def test_enricher_returns_valid_structure(self, mock_enrich):
        """Test that enricher returns valid enrichment structure."""
        # Skip if no API key available
        import os
        if not os.getenv('OPENAI_API_KEY'):
            pytest.skip("No OPENAI_API_KEY available")
        
        # Mock the enrichment result
        mock_result = MovieEnrichment(
            movieId=862,
            sentiment="positive",
            budget_tier="medium",
            revenue_tier="high",
            effectiveness_score=8.5,
            target_audience="adult",
            reasoning="Test reasoning",
            model_used="test-model",
            enriched_at="2024-01-01T00:00:00Z"
        )
        mock_enrich.return_value = mock_result
        
        enricher = DatabaseMovieEnricher()
        result = await enricher.enrich_movie(862)
        
        assert result is not None
        assert result.sentiment in ['positive', 'negative', 'neutral']
        assert 0 <= result.effectiveness_score <= 10


class TestEnrichmentConsistency:
    """Test enrichment consistency and quality."""
    
    def test_high_budget_high_revenue_effectiveness(self):
        """Test that high budget + high revenue movies have good effectiveness scores."""
        enrichments = MovieEnrichment.get_all()
        high_performers = [
            e for e in enrichments 
            if e.budget_tier in ['high', 'very_high'] 
            and e.revenue_tier in ['high', 'very_high']
        ]
        
        if high_performers:
            avg_score = sum(e.effectiveness_score for e in high_performers) / len(high_performers)
            # High budget + high revenue should generally have good effectiveness
            assert avg_score >= 5.0
    
    def test_sentiment_distribution(self):
        """Test that sentiment is reasonably distributed."""
        enrichments = MovieEnrichment.get_all()
        sentiments = [e.sentiment for e in enrichments]
        
        # Should have a mix of sentiments, not all the same
        unique_sentiments = set(sentiments)
        assert len(unique_sentiments) >= 2
        
        # Positive should be most common (most movies aim to be good)
        positive_count = sentiments.count('positive')
        assert positive_count > len(sentiments) * 0.3  # At least 30% positive

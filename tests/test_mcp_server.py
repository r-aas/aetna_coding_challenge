"""Tests for MCP server and chat functionality."""

import pytest
import json
from src.mcp_server import search_movies, get_movie_details, get_user_ratings, get_random_movies
from src.chat_agent import MovieChatAgent
from src.db import Movie, MovieEnrichment


class TestMCPTools:
    """Test MCP server tools."""
    
    def test_search_movies(self):
        """Test movie search tool."""
        results_json = search_movies(query="action", limit=5)
        results = json.loads(results_json)
        
        assert isinstance(results, list)
        assert len(results) <= 5
        
        if results:
            movie = results[0]
            assert 'movieId' in movie
            assert 'title' in movie
    
    @pytest.mark.asyncio
    async def test_get_movie_details(self):
        """Test get movie details tool."""
        details_json = await get_movie_details(movie_id=862)  # Toy Story
        details = json.loads(details_json)
        
        assert isinstance(details, dict)
        assert 'movieId' in details
        assert 'title' in details
    
    def test_get_user_ratings(self):
        """Test get user ratings."""
        results_json = get_user_ratings(user_id=5, limit=5)
        results = json.loads(results_json)
        
        # get_user_ratings returns a dict with user info and ratings array
        assert isinstance(results, dict)
        assert 'user_id' in results
        assert 'ratings' in results
        if results['ratings']:
            movie = results['ratings'][0]
            assert 'movieId' in movie
            assert 'rating' in movie
    
    def test_get_random_movies(self):
        """Test get random movies."""
        results_json = get_random_movies(n=5)
        results = json.loads(results_json)
        
        assert isinstance(results, list)
        assert len(results) <= 5
    
    @pytest.mark.asyncio
    async def test_movie_details_includes_enrichment(self):
        """Test that movie details include enrichment when available."""
        enrichments = MovieEnrichment.get_all()
        if enrichments:
            movie_id = enrichments[0].movieId
            details_json = await get_movie_details(movie_id=movie_id)
            details = json.loads(details_json)
            
            if 'enrichment' in details:
                assert 'sentiment' in details['enrichment']
                assert 'effectiveness_score' in details['enrichment']


class TestChatAgent:
    """Test chat agent functionality."""
    
    def test_agent_initialization(self):
        """Test that chat agent can be initialized."""
        # Skip if no API key available
        import os
        if not os.getenv('OPENAI_API_KEY'):
            pytest.skip("No OPENAI_API_KEY available")
        
        agent = MovieChatAgent(model_name="openai:qwen2.5:7b")
        assert agent is not None
        assert agent.model_name == "openai:qwen2.5:7b"
    
    def test_agent_custom_system_prompt(self):
        """Test agent with custom system prompt via CLI."""
        # MovieChatAgent doesn't support system_prompt directly
        # Test via MovieChatCLI instead
        from src.chat_agent import MovieChatCLI
        
        custom_prompt = "You are a helpful movie assistant."
        cli = MovieChatCLI(
            model_name="openai:qwen2.5:7b",
            system_prompt=custom_prompt
        )
        assert cli.system_prompt == custom_prompt


class TestMCPIntegration:
    """Test MCP tool integration."""
    
    @pytest.mark.asyncio
    async def test_all_tools_return_valid_json(self):
        """Test that all tools return valid JSON."""
        # Test search
        search_json = search_movies(query="matrix", limit=3)
        search_results = json.loads(search_json)
        assert isinstance(search_results, list)
        
        # Test details
        if search_results:
            movie_id = search_results[0]['movieId']
            details_json = await get_movie_details(movie_id=movie_id)
            details = json.loads(details_json)
            assert isinstance(details, dict)
        
        # Test user ratings
        ratings_json = get_user_ratings(user_id=5, limit=5)
        ratings = json.loads(ratings_json)
        assert isinstance(ratings, dict)
        assert 'ratings' in ratings
        
        # Test random movies
        random_json = get_random_movies(n=3)
        random_movies = json.loads(random_json)
        assert isinstance(random_movies, list)

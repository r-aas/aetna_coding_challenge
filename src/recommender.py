"""LLM-powered movie recommendation system using pydantic-ai."""

from typing import Optional, List
from pydantic import BaseModel, Field
from pydantic_ai import Agent
import pandas as pd

from src.db import Movie, Rating, MovieEnrichment


# Pydantic models for structured outputs
class UserPreferences(BaseModel):
    """User preference profile extracted from rating history."""
    favorite_genres: List[str] = Field(
        description="Genres user rates highly"
    )
    preferred_budget_tier: str = Field(
        description="Budget tier user prefers (low/medium/high/very_high)"
    )
    sentiment_preference: str = Field(
        description="Preferred sentiment (positive/negative/neutral/mixed)"
    )
    target_audience_match: str = Field(
        description="User's demographic match (family/young_adult/adult)"
    )
    summary: str = Field(
        description="Natural language summary of user's taste"
    )


class MovieRecommendation(BaseModel):
    """Single movie recommendation with reasoning."""
    movie_id: int
    title: str
    match_score: float = Field(
        ge=0.0,
        le=10.0,
        description="How well this movie matches user preferences (0-10)"
    )
    reasoning: str = Field(
        description="Why this movie is recommended for this user"
    )


class RecommendationList(BaseModel):
    """List of movie recommendations."""
    recommendations: List[MovieRecommendation]
    user_profile_summary: str = Field(
        description="Brief summary of user's preferences used for recommendations"
    )


class RatingPrediction(BaseModel):
    """Predicted rating for a movie."""
    predicted_rating: float = Field(
        ge=0.0,
        le=5.0,
        description="Predicted user rating (0-5 scale)"
    )
    confidence: str = Field(
        description="Confidence level: high/medium/low"
    )
    reasoning: str = Field(
        description="Explanation of the prediction"
    )


class MovieComparison(BaseModel):
    """Comparative analysis of multiple movies."""
    comparison_summary: str = Field(
        description="Overall comparison summary"
    )
    strongest_movie: str = Field(
        description="Which movie is strongest and why"
    )
    key_differences: List[str] = Field(
        description="Key differences between the movies"
    )
    recommendations: str = Field(
        description="Which movie to watch based on different preferences"
    )


class MovieRecommender:
    """LLM-powered movie recommendation system.
    
    Provides personalized recommendations, rating predictions,
    preference summaries, and comparative analyses.
    """
    
    def __init__(self, model_name: str = "openai:gpt-4o-mini"):
        """Initialize recommender with pydantic-ai agents.
        
        Args:
            model_name: LLM model to use
        """
        self.model_name = model_name
        
        # Agent for user preference extraction
        self.preference_agent = Agent(
            model_name,
            output_type=UserPreferences,
            system_prompt="""You are an expert at understanding user preferences from their rating history.
            
Analyze the user's movie ratings and extract their preferences considering:
- Genre preferences (which genres they rate highly)
- Budget/production quality preferences
- Sentiment preferences (do they prefer uplifting or dark content?)
- Target audience match (family-friendly vs mature content)

Provide a clear, concise summary of their taste in movies."""
        )
        
        # Agent for generating recommendations
        self.recommendation_agent = Agent(
            model_name,
            output_type=RecommendationList,
            system_prompt="""You are an expert movie recommender.
            
Given a user's preference profile and a set of candidate movies, recommend the best matches.
Consider:
- Genre alignment with user preferences
- Budget/production quality match
- Sentiment alignment
- Target audience appropriateness
- Diversity in recommendations (don't recommend too many similar movies)

Provide clear reasoning for each recommendation and a match score (0-10)."""
        )
        
        # Agent for rating predictions
        self.prediction_agent = Agent(
            model_name,
            output_type=RatingPrediction,
            system_prompt="""You are an expert at predicting user ratings.
            
Given a user's rating history and a new movie, predict what rating they would give.
Consider:
- User's genre preferences vs movie's genres
- User's budget preferences vs movie's production value
- User's sentiment preferences vs movie's tone
- Similar movies the user has rated

Provide a prediction with confidence level and clear reasoning."""
        )
        
        # Agent for movie comparisons
        self.comparison_agent = Agent(
            model_name,
            output_type=MovieComparison,
            system_prompt="""You are a film critic and analyst.
            
Compare multiple movies across various dimensions:
- Budget and production value
- Revenue and commercial success
- Critical reception and ratings
- Target audience and themes
- Cultural impact

Provide a balanced analysis highlighting strengths and differences."""
        )
    
    async def analyze_user_preferences(self, user_id: int) -> UserPreferences:
        """Extract user preferences from their rating history.
        
        Args:
            user_id: User ID to analyze
            
        Returns:
            UserPreferences with extracted profile
        """
        # Get user's ratings
        ratings = Rating.get_for_user(user_id)
        
        if not ratings:
            raise ValueError(f"No ratings found for user {user_id}")
        
        # Get movie details for rated movies
        rated_movies = []
        for rating in ratings[:50]:  # Limit to recent 50 for context length
            movie = Movie.get_by_id(rating.movieId)
            if movie:
                # Get enrichment if available
                enrichment = MovieEnrichment.get_by_id(rating.movieId)
                rated_movies.append({
                    'title': movie.title,
                    'rating': rating.rating,
                    'genres': movie.genres,
                    'budget': movie.budget,
                    'overview': movie.overview[:200] if movie.overview else None,
                    'sentiment': enrichment.sentiment if enrichment else None,
                    'budget_tier': enrichment.budget_tier if enrichment else None,
                    'target_audience': enrichment.target_audience if enrichment else None,
                })
        
        # Build context for LLM
        context = f"""User {user_id} has rated {len(ratings)} movies. Here are their ratings:

"""
        for m in rated_movies[:20]:  # Show top 20 for analysis
            context += f"- {m['title']}: {m['rating']}/5.0"
            if m['genres']:
                context += f" (Genres: {m['genres']})"
            if m['sentiment']:
                context += f" [Sentiment: {m['sentiment']}]"
            context += "\n"
        
        context += f"\n\nAnalyze this user's preferences and provide a profile."
        
        result = await self.preference_agent.run(context)
        return result.output
    
    async def recommend(
        self,
        user_id: int,
        query: Optional[str] = None,
        n: int = 10,
        use_enriched_only: bool = True
    ) -> RecommendationList:
        """Generate personalized movie recommendations.
        
        Args:
            user_id: User ID to recommend for
            query: Optional natural language query to filter (e.g., "action movies with high revenue")
            n: Number of recommendations
            use_enriched_only: Only recommend from enriched movies
            
        Returns:
            RecommendationList with recommendations and reasoning
        """
        # Get user preferences
        user_prefs = await self.analyze_user_preferences(user_id)
        
        # Get candidate movies (exclude already rated)
        user_ratings = Rating.get_for_user(user_id)
        rated_ids = {r.movieId for r in user_ratings}
        
        # Get candidate movies
        if use_enriched_only:
            # Get all enriched movies
            enriched = MovieEnrichment.get_all()
            candidate_ids = [e.movieId for e in enriched if e.movieId not in rated_ids]
        else:
            # Get random sample from all movies
            all_movies = Movie.get_random(n=n*3, with_budget=True)
            candidate_ids = [mid for mid in all_movies['movieId'].tolist() if mid not in rated_ids]
        
        # Get details for candidate movies
        candidates = []
        for movie_id in candidate_ids[:n*2]:  # Get 2x candidates for better selection
            movie = Movie.get_by_id(movie_id)
            enrichment = MovieEnrichment.get_by_id(movie_id)
            
            if movie and enrichment:
                candidates.append({
                    'movie_id': movie.movieId,
                    'title': movie.title,
                    'genres': movie.genres,
                    'overview': movie.overview[:200] if movie.overview else "No overview",
                    'budget': movie.budget,
                    'revenue': movie.revenue,
                    'sentiment': enrichment.sentiment,
                    'budget_tier': enrichment.budget_tier,
                    'revenue_tier': enrichment.revenue_tier,
                    'effectiveness_score': enrichment.effectiveness_score,
                    'target_audience': enrichment.target_audience,
                })
        
        # Build context for LLM
        context = f"""User Preferences:
{user_prefs.summary}
- Favorite genres: {', '.join(user_prefs.favorite_genres)}
- Preferred budget tier: {user_prefs.preferred_budget_tier}
- Sentiment preference: {user_prefs.sentiment_preference}
- Target audience: {user_prefs.target_audience_match}

"""
        
        if query:
            context += f"\nUser Query: {query}\n"
        
        context += f"\n\nCandidate Movies ({len(candidates)} available):\n\n"
        
        for c in candidates[:30]:  # Show top 30 to LLM
            context += f"- ID {c['movie_id']}: {c['title']}\n"
            context += f"  Genres: {c['genres']}\n"
            context += f"  Sentiment: {c['sentiment']}, Budget: {c['budget_tier']}, Audience: {c['target_audience']}\n"
            context += f"  Effectiveness: {c['effectiveness_score']}/10\n"
            if c['overview']:
                context += f"  Overview: {c['overview']}\n"
            context += "\n"
        
        context += f"\nRecommend the top {n} movies for this user."
        if query:
            context += f" Filter and prioritize based on the query: '{query}'"
        
        result = await self.recommendation_agent.run(context)
        return result.output
    
    async def predict_rating(self, user_id: int, movie_id: int) -> RatingPrediction:
        """Predict what rating a user would give a movie.
        
        Args:
            user_id: User ID
            movie_id: Movie ID to predict rating for
            
        Returns:
            RatingPrediction with predicted rating and reasoning
        """
        # Get user preferences
        user_prefs = await self.analyze_user_preferences(user_id)
        
        # Get movie details
        movie = Movie.get_by_id(movie_id)
        enrichment = MovieEnrichment.get_by_id(movie_id)
        
        if not movie:
            raise ValueError(f"Movie {movie_id} not found")
        
        # Build context
        context = f"""User Preferences:
{user_prefs.summary}
- Favorite genres: {', '.join(user_prefs.favorite_genres)}
- Preferred budget tier: {user_prefs.preferred_budget_tier}
- Sentiment preference: {user_prefs.sentiment_preference}

Movie to Predict:
- Title: {movie.title}
- Genres: {movie.genres}
- Overview: {movie.overview[:300] if movie.overview else "No overview"}
- Budget: ${movie.budget:,} ({enrichment.budget_tier if enrichment else 'unknown'})
- Revenue: ${movie.revenue:,} ({enrichment.revenue_tier if enrichment else 'unknown'})
"""
        
        if enrichment:
            context += f"- Sentiment: {enrichment.sentiment}\n"
            context += f"- Target Audience: {enrichment.target_audience}\n"
            context += f"- Effectiveness Score: {enrichment.effectiveness_score}/10\n"
        
        context += "\nPredict what rating (0-5) this user would give this movie."
        
        result = await self.prediction_agent.run(context)
        return result.output
    
    async def summarize_preferences(self, user_id: int) -> str:
        """Generate natural language summary of user's preferences.
        
        Args:
            user_id: User ID
            
        Returns:
            Natural language summary
        """
        prefs = await self.analyze_user_preferences(user_id)
        return prefs.summary
    
    async def compare_movies(self, movie_ids: List[int]) -> MovieComparison:
        """Compare multiple movies across various dimensions.
        
        Args:
            movie_ids: List of movie IDs to compare
            
        Returns:
            MovieComparison with analysis
        """
        if len(movie_ids) < 2:
            raise ValueError("Need at least 2 movies to compare")
        
        # Get movie details
        movies_data = []
        for movie_id in movie_ids:
            movie = Movie.get_by_id(movie_id)
            enrichment = MovieEnrichment.get_by_id(movie_id)
            
            if not movie:
                continue
            
            # Get average rating
            ratings = movie.get_ratings()
            avg_rating = sum(r.rating for r in ratings) / len(ratings) if ratings else None
            
            movies_data.append({
                'id': movie.movieId,
                'title': movie.title,
                'genres': movie.genres,
                'overview': movie.overview[:200] if movie.overview else "No overview",
                'budget': movie.budget,
                'revenue': movie.revenue,
                'runtime': movie.runtime,
                'avg_rating': avg_rating,
                'num_ratings': len(ratings),
                'sentiment': enrichment.sentiment if enrichment else None,
                'budget_tier': enrichment.budget_tier if enrichment else None,
                'revenue_tier': enrichment.revenue_tier if enrichment else None,
                'effectiveness_score': enrichment.effectiveness_score if enrichment else None,
                'target_audience': enrichment.target_audience if enrichment else None,
            })
        
        # Build context
        context = f"Compare these {len(movies_data)} movies:\n\n"
        
        for m in movies_data:
            context += f"**{m['title']}** (ID: {m['id']})\n"
            context += f"- Genres: {m['genres']}\n"
            context += f"- Budget: ${m['budget']:,} ({m['budget_tier']})\n" if m['budget'] else "- Budget: Unknown\n"
            context += f"- Revenue: ${m['revenue']:,} ({m['revenue_tier']})\n" if m['revenue'] else "- Revenue: Unknown\n"
            context += f"- Runtime: {m['runtime']} minutes\n" if m['runtime'] else ""
            context += f"- Avg Rating: {m['avg_rating']:.2f}/5.0 ({m['num_ratings']} ratings)\n" if m['avg_rating'] else ""
            if m['sentiment']:
                context += f"- Sentiment: {m['sentiment']}\n"
            if m['effectiveness_score']:
                context += f"- Effectiveness: {m['effectiveness_score']}/10\n"
            if m['target_audience']:
                context += f"- Target Audience: {m['target_audience']}\n"
            context += f"- Overview: {m['overview']}\n\n"
        
        context += "Provide a comprehensive comparison of these movies."
        
        result = await self.comparison_agent.run(context)
        return result.output


# Convenience functions
async def get_recommendations(
    user_id: int,
    query: Optional[str] = None,
    n: int = 10,
    model: str = "openai:gpt-4o-mini"
) -> RecommendationList:
    """Quick function to get recommendations."""
    recommender = MovieRecommender(model_name=model)
    return await recommender.recommend(user_id, query, n)


async def predict_user_rating(
    user_id: int,
    movie_id: int,
    model: str = "openai:gpt-4o-mini"
) -> RatingPrediction:
    """Quick function to predict rating."""
    recommender = MovieRecommender(model_name=model)
    return await recommender.predict_rating(user_id, movie_id)

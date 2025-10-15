"""MCP Server for Movie Database Operations.

Exposes movie database functionality via MCP protocol for use by AI agents.
Includes both ML-based (fast) and LLM-based (intelligent) recommendations.
"""

import asyncio
from typing import Optional, List
from mcp.server.fastmcp import FastMCP
from src.db import Movie, Rating, MovieEnrichment

# Create MCP server
mcp = FastMCP("Movie Database Server")


@mcp.tool()
def search_movies(query: str, limit: int = 10) -> str:
    """Search for movies by title.
    
    Args:
        query: Search query for movie title
        limit: Maximum number of results (default: 10)
    
    Returns:
        JSON string with list of matching movies
    """
    import json
    from sqlmodel import select, or_
    
    with Movie.get_session() as session:
        stmt = select(Movie).where(
            Movie.title.ilike(f"%{query}%")
        ).limit(limit)
        
        movies = session.exec(stmt).all()
        
        results = []
        for movie in movies:
            results.append({
                "movieId": movie.movieId,
                "title": movie.title,
                "genres": movie.genres,
                "budget": movie.budget,
                "revenue": movie.revenue,
                "overview": movie.overview[:200] + "..." if movie.overview and len(movie.overview) > 200 else movie.overview
            })
        
        return json.dumps(results, indent=2)


@mcp.tool()
def get_movie_details(movie_id: int) -> str:
    """Get detailed information about a specific movie.
    
    Args:
        movie_id: The movie ID to look up
    
    Returns:
        JSON string with movie details including enrichments
    """
    import json
    
    movie = Movie.get_by_id(movie_id)
    if not movie:
        return json.dumps({"error": f"Movie {movie_id} not found"})
    
    # Get enrichment data
    enrichment = MovieEnrichment.get_by_id(movie_id)
    
    # Get ratings
    ratings = movie.get_ratings()
    avg_rating = sum(r.rating for r in ratings) / len(ratings) if ratings else None
    
    result = {
        "movieId": movie.movieId,
        "title": movie.title,
        "overview": movie.overview,
        "genres": movie.genres,
        "budget": movie.budget,
        "revenue": movie.revenue,
        "runtime": movie.runtime,
        "releaseDate": movie.releaseDate,
        "language": movie.language,
        "status": movie.status,
        "ratings": {
            "count": len(ratings),
            "average": round(avg_rating, 2) if avg_rating else None
        }
    }
    
    if enrichment:
        result["enrichment"] = {
            "sentiment": enrichment.sentiment,
            "budget_tier": enrichment.budget_tier,
            "revenue_tier": enrichment.revenue_tier,
            "effectiveness_score": enrichment.effectiveness_score,
            "target_audience": enrichment.target_audience,
            "reasoning": enrichment.reasoning
        }
    
    return json.dumps(result, indent=2)


@mcp.tool()
def get_user_ratings(user_id: int, limit: Optional[int] = None) -> str:
    """Get a user's rating history.
    
    Args:
        user_id: The user ID to look up
        limit: Optional limit on number of ratings to return
    
    Returns:
        JSON string with user's ratings and movie details
    """
    import json
    
    ratings = Rating.get_for_user(user_id)
    
    if not ratings:
        return json.dumps({"error": f"No ratings found for user {user_id}"})
    
    if limit:
        ratings = ratings[:limit]
    
    results = []
    for rating in ratings:
        movie = Movie.get_by_id(rating.movieId)
        if movie:
            results.append({
                "movieId": rating.movieId,
                "title": movie.title,
                "rating": rating.rating,
                "timestamp": rating.timestamp,
                "genres": movie.genres
            })
    
    summary = {
        "user_id": user_id,
        "total_ratings": len(Rating.get_for_user(user_id)),
        "average_rating": sum(r.rating for r in Rating.get_for_user(user_id)) / len(Rating.get_for_user(user_id)),
        "ratings": results
    }
    
    return json.dumps(summary, indent=2)


@mcp.tool()
async def get_movie_recommendations(
    user_id: int,
    n: int = 10,
    use_llm: bool = False,
    query: Optional[str] = None
) -> str:
    """Get movie recommendations for a user.
    
    Supports both fast ML-based (default) and intelligent LLM-based recommendations.
    
    Args:
        user_id: User ID to recommend for
        n: Number of recommendations (default: 10)
        use_llm: Use LLM for intelligent recommendations with reasoning (default: False, uses fast ML)
        query: Optional natural language query for LLM mode (e.g., "action movies with high revenue")
    
    Returns:
        JSON string with recommendations (and reasoning if use_llm=True)
    """
    import json
    from pathlib import Path
    from sqlmodel import select
    
    # LLM mode: Intelligent recommendations with reasoning
    if use_llm:
        try:
            from src.recommender import MovieRecommender
            recommender = MovieRecommender()
            result = await recommender.recommend(user_id, query, n)
            
            recommendations = []
            for rec in result.recommendations:
                recommendations.append({
                    "movieId": rec.movie_id,
                    "title": rec.title,
                    "match_score": rec.match_score,
                    "reasoning": rec.reasoning
                })
            
            return json.dumps({
                "user_id": user_id,
                "mode": "llm",
                "user_profile": result.user_profile_summary,
                "query": query,
                "recommendations": recommendations
            }, indent=2)
            
        except Exception as e:
            return json.dumps({"error": str(e)})
    
    # ML mode: Fast collaborative filtering
    else:
        model_path = "models/hybrid_recommender.pkl"
        
        try:
            from src.hybrid_recommender import load_or_train
            
            # Load existing model or automatically train new one
            recommender = load_or_train(model_path, verbose=False)
            
            # Get recommendations
            recs = recommender.recommend(user_id, n=n)
            
            # Convert np.int64 to Python int for SQL compatibility
            movie_ids = [int(movie_id) for movie_id, _ in recs]
            
            # Fetch all movies in a single session
            with Movie.get_session() as session:
                stmt = select(Movie).where(Movie.movieId.in_(movie_ids))
                movies_list = session.exec(stmt).all()
                movies_dict = {m.movieId: m for m in movies_list}
            
            results = []
            for movie_id, score in recs:
                movie = movies_dict.get(int(movie_id))
                if movie:
                    results.append({
                        "movieId": int(movie_id),  # Convert numpy int64 to Python int
                        "title": movie.title,
                        "score": round(float(score), 4),
                        "genres": movie.genres
                    })
            
            return json.dumps({
                "user_id": user_id,
                "mode": "ml",
                "recommendations": results
            }, indent=2)
            
        except ValueError as e:
            return json.dumps({"error": str(e)})


@mcp.tool()
def compare_movies(movie_ids: List[int]) -> str:
    """Compare multiple movies across various dimensions.
    
    Args:
        movie_ids: List of movie IDs to compare
    
    Returns:
        JSON string with comparison data
    """
    import json
    
    if len(movie_ids) < 2:
        return json.dumps({"error": "Need at least 2 movies to compare"})
    
    movies_data = []
    for movie_id in movie_ids:
        movie = Movie.get_by_id(movie_id)
        if not movie:
            continue
        
        enrichment = MovieEnrichment.get_by_id(movie_id)
        ratings = movie.get_ratings()
        avg_rating = sum(r.rating for r in ratings) / len(ratings) if ratings else None
        
        data = {
            "movieId": movie.movieId,
            "title": movie.title,
            "budget": movie.budget,
            "revenue": movie.revenue,
            "genres": movie.genres,
            "avg_rating": round(avg_rating, 2) if avg_rating else None,
            "rating_count": len(ratings)
        }
        
        if enrichment:
            data["enrichment"] = {
                "sentiment": enrichment.sentiment,
                "effectiveness_score": enrichment.effectiveness_score,
                "target_audience": enrichment.target_audience
            }
        
        movies_data.append(data)
    
    return json.dumps({
        "comparison": movies_data,
        "summary": {
            "highest_budget": max((m["budget"] or 0 for m in movies_data)),
            "highest_revenue": max((m["revenue"] or 0 for m in movies_data)),
            "highest_rated": max((m["avg_rating"] or 0 for m in movies_data))
        }
    }, indent=2)


@mcp.tool()
def get_random_movies(n: int = 10, with_enrichment: bool = False) -> str:
    """Get random movies from the database.
    
    Args:
        n: Number of random movies to return
        with_enrichment: Only return movies with enrichment data
    
    Returns:
        JSON string with random movies
    """
    import json
    
    movies_df = Movie.get_random(n=n, with_budget=False)
    
    results = []
    for _, row in movies_df.iterrows():
        movie_data = {
            "movieId": row['movieId'],
            "title": row['title'],
            "genres": row['genres'],
            "budget": row['budget'],
            "revenue": row['revenue']
        }
        
        if with_enrichment:
            enrichment = MovieEnrichment.get_by_id(row['movieId'])
            if enrichment:
                movie_data["enrichment"] = {
                    "sentiment": enrichment.sentiment,
                    "effectiveness_score": enrichment.effectiveness_score,
                    "target_audience": enrichment.target_audience
                }
            else:
                continue  # Skip movies without enrichment if requested
        
        results.append(movie_data)
    
    return json.dumps(results, indent=2)


@mcp.tool()
async def predict_rating(user_id: int, movie_id: int) -> str:
    """Predict what rating a user would give a movie.
    
    Uses LLM to analyze user's preferences and predict their rating for a movie
    they haven't seen yet.
    
    Args:
        user_id: User ID
        movie_id: Movie ID to predict rating for
    
    Returns:
        JSON string with predicted rating, confidence, and reasoning
    """
    import json
    from src.recommender import MovieRecommender
    
    try:
        recommender = MovieRecommender()
        result = await recommender.predict_rating(user_id, movie_id)
        
        return json.dumps({
            "user_id": user_id,
            "movie_id": movie_id,
            "predicted_rating": result.predicted_rating,
            "confidence": result.confidence,
            "reasoning": result.reasoning
        }, indent=2)
        
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def summarize_user_preferences(user_id: int) -> str:
    """Generate natural language summary of user's movie preferences.
    
    Analyzes user's rating history to understand their taste in movies.
    
    Args:
        user_id: User ID to analyze
    
    Returns:
        JSON string with preference summary
    """
    import json
    from src.recommender import MovieRecommender
    
    try:
        recommender = MovieRecommender()
        summary = await recommender.summarize_preferences(user_id)
        
        return json.dumps({
            "user_id": user_id,
            "preference_summary": summary
        }, indent=2)
        
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def semantic_search(
    query: str,
    k: int = 10,
    genre: Optional[str] = None,
    min_budget: Optional[int] = None,
    max_budget: Optional[int] = None,
    with_enrichment: bool = True
) -> str:
    """Semantic search for movies using natural language.
    
    Uses vector embeddings (sqlite-vec) for semantic similarity search.
    Supports metadata filtering for genre, budget, etc.
    
    Args:
        query: Natural language search query (e.g., "dark psychological thriller")
        k: Number of results to return (default: 10)
        genre: Optional genre filter (e.g., "Action", "Sci-Fi")
        min_budget: Optional minimum budget filter
        max_budget: Optional maximum budget filter
        with_enrichment: Include enrichment data (default: True)
    
    Returns:
        JSON string with search results and similarity scores
    """
    import json
    from src.embeddings import init_embeddings
    
    try:
        # Build filters
        filters = {}
        if genre:
            filters["genres__contains"] = genre
        if min_budget:
            filters["budget__gte"] = min_budget
        if max_budget:
            filters["budget__lte"] = max_budget
        
        # Perform search
        embeddings = init_embeddings()
        results = embeddings.hybrid_search(
            query=query,
            k=k,
            filters=filters if filters else None,
            enrich_results=with_enrichment
        )
        
        return json.dumps({
            "query": query,
            "filters": filters if filters else None,
            "results": results
        }, indent=2)
        
    except Exception as e:
        return json.dumps({"error": str(e)})


if __name__ == "__main__":
    # Run the MCP server
    mcp.run()

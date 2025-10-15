"""CLI interface for Aetna AI Engineer Coding Challenge."""

import typer
from typing import Optional
from pathlib import Path

from src.db import Movie, Rating, set_db_paths, get_db_paths

app = typer.Typer(help="Aetna AI Engineer Coding Challenge - Movie System")


@app.callback()
def main(
    movies_db: Optional[str] = typer.Option(None, "--movies-db", help="Custom path to movies database"),
    ratings_db: Optional[str] = typer.Option(None, "--ratings-db", help="Custom path to ratings database"),
):
    """Configure database paths for all commands."""
    if movies_db or ratings_db:
        set_db_paths(movies_db=movies_db, ratings_db=ratings_db)


@app.command()
def test_db():
    """Test database connectivity and models."""
    typer.echo("=" * 60)
    typer.echo("Testing Movie & Rating Models")
    typer.echo("=" * 60)
    
    # Show current DB paths
    typer.echo("\n0. Current database paths:")
    paths = get_db_paths()
    typer.echo(f"   Movies:  {paths['movies_db']}")
    typer.echo(f"   Ratings: {paths['ratings_db']}")
    
    # Test 1: Get specific movie
    typer.echo("\n1. Movie.get_by_id(31):")
    movie = Movie.get_by_id(31)
    if movie:
        typer.echo(f"   ✓ {movie.title} (Budget: ${movie.budget:,}, Revenue: ${movie.revenue:,})")
    
    # Test 2: Random sample - now returns DataFrame
    typer.echo("\n2. Movie.get_random(n=5):")
    movies_df = Movie.get_random(n=5)
    typer.echo(f"   ✓ Found {len(movies_df)} movies:")
    for title in movies_df['title'].head(5):
        typer.echo(f"     - {title}")
    
    # Test 3: Movie ratings
    typer.echo("\n3. movie.get_ratings():")
    if movie:
        ratings = movie.get_ratings()
        avg = sum(r.rating for r in ratings) / len(ratings) if ratings else 0
        typer.echo(f"   ✓ Movie has {len(ratings)} ratings (avg: {avg:.2f}/5.0)")
    
    # Test 4: User ratings
    typer.echo("\n4. Rating.get_for_user(1):")
    user_ratings = Rating.get_for_user(1)
    typer.echo(f"   ✓ User 1 rated {len(user_ratings)} movies")
    
    typer.echo("\n" + "=" * 60)
    typer.echo("✅ All models working!")
    typer.echo("=" * 60)


@app.command()
def sample(
    n: int = typer.Option(10, help="Number of movies to sample"),
    with_budget: bool = typer.Option(True, help="Filter for movies with budget data")
):
    """Sample random movies from the database."""
    typer.echo(f"Sampling {n} movies" + (" with budget data" if with_budget else "") + "...")
    
    movies_df = Movie.get_random(n=n, with_budget=with_budget)
    
    typer.echo("\n" + "=" * 100)
    typer.echo(f"{'Title':<40} {'Budget':<15} {'Revenue':<15} {'Genres':<30}")
    typer.echo("=" * 100)
    
    for _, row in movies_df.iterrows():
        budget = f"${row['budget']:,}" if row['budget'] else "N/A"
        revenue = f"${row['revenue']:,}" if row['revenue'] else "N/A"
        genres = row['genres'][:27] + "..." if row['genres'] and len(row['genres']) > 30 else (row['genres'] or "N/A")
        typer.echo(f"{row['title']:<40} {budget:<15} {revenue:<15} {genres:<30}")
    
    typer.echo("=" * 100)


@app.command()
def movie(
    movie_id: int = typer.Argument(..., help="Movie ID to look up"),
    auto_enrich: bool = typer.Option(False, "--auto-enrich", help="Automatically enrich if needed")
):
    """Get details for a specific movie."""
    import asyncio
    
    m = Movie.get_by_id(movie_id)
    
    if not m:
        typer.echo(f"❌ Movie {movie_id} not found", err=True)
        raise typer.Exit(1)
    
    async def get_enrichment():
        from src.db import MovieEnrichment
        if auto_enrich:
            from src.enricher_db import enrich_movie_if_needed
            try:
                # Try to use current provider model, fallback to OpenAI
                try:
                    import os
                    provider = os.getenv('LLM_PROVIDER', 'openai')
                    if provider == 'ollama':
                        model_name = "ollama:qwen3:32b"
                    else:
                        model_name = "openai:gpt-4o-mini"
                except:
                    model_name = "openai:gpt-4o-mini"
                return await enrich_movie_if_needed(movie_id, model_name, verbose=True)
            except Exception as e:
                typer.echo(f"⚠️  Auto-enrichment failed: {e}")
                return MovieEnrichment.get_by_id(movie_id)
        else:
            return MovieEnrichment.get_by_id(movie_id)
    
    # Get enrichment data
    enrichment = asyncio.run(get_enrichment())
    
    typer.echo("\n" + "=" * 80)
    typer.echo(f"🎬 {m.title}")
    typer.echo("=" * 80)
    
    typer.echo(f"\n📊 Financials:")
    typer.echo(f"   Budget:  ${m.budget:,}" if m.budget else "   Budget:  N/A")
    typer.echo(f"   Revenue: ${m.revenue:,}" if m.revenue else "   Revenue: N/A")
    
    if m.genres:
        typer.echo(f"\n🎭 Genres: {m.genres}")
    
    if m.overview:
        typer.echo(f"\n📝 Overview:")
        typer.echo(f"   {m.overview[:200]}...")
    
    # Get ratings
    ratings = m.get_ratings()
    if ratings:
        avg = sum(r.rating for r in ratings) / len(ratings)
        typer.echo(f"\n⭐ Ratings: {len(ratings)} ratings (avg: {avg:.2f}/5.0)")
    
    # Show enrichment data if available
    if enrichment:
        typer.echo(f"\n🤖 LLM Enrichment:")
        typer.echo(f"   Sentiment: {enrichment.sentiment}")
        typer.echo(f"   Budget Tier: {enrichment.budget_tier}")
        typer.echo(f"   Revenue Tier: {enrichment.revenue_tier}")
        typer.echo(f"   Effectiveness Score: {enrichment.effectiveness_score}/10")
        typer.echo(f"   Target Audience: {enrichment.target_audience}")
        if enrichment.reasoning:
            typer.echo(f"   Reasoning: {enrichment.reasoning[:100]}...")
    elif auto_enrich:
        typer.echo(f"\n⚠️  Enrichment not available")
    else:
        typer.echo(f"\n💡 Add --auto-enrich to get LLM-generated insights")
    
    typer.echo("=" * 80 + "\n")


@app.command()
def user(user_id: int = typer.Argument(..., help="User ID to look up")):
    """Get ratings for a specific user."""
    ratings = Rating.get_for_user(user_id)
    
    if not ratings:
        typer.echo(f"❌ No ratings found for user {user_id}", err=True)
        raise typer.Exit(1)
    
    typer.echo("\n" + "=" * 80)
    typer.echo(f"👤 User {user_id} - {len(ratings)} ratings")
    typer.echo("=" * 80)
    
    # Get average rating
    avg = sum(r.rating for r in ratings) / len(ratings)
    typer.echo(f"\n⭐ Average rating: {avg:.2f}/5.0")
    
    # Show sample ratings
    typer.echo(f"\n📊 Sample ratings:")
    for rating in ratings[:10]:
        movie = Movie.get_by_id(rating.movieId)
        title = movie.title if movie else f"Movie {rating.movieId}"
        typer.echo(f"   {rating.rating}/5.0 - {title}")
    
    if len(ratings) > 10:
        typer.echo(f"   ... and {len(ratings) - 10} more")
    
    typer.echo("=" * 80 + "\n")


@app.command()
def enrich(
    n: Optional[int] = typer.Option(None, help="Number of movies to enrich (default: all unenriched)"),
    all: bool = typer.Option(False, "--all", help="Enrich ALL movies (not just unenriched)"),
    model: str = typer.Option("openai:gpt-4o-mini", help="LLM model to use"),
    overwrite: bool = typer.Option(False, help="Re-enrich movies that already have enrichments"),
    batch_size: int = typer.Option(50, help="Batch size for processing")
):
    """Enrich movies with LLM-generated attributes and store in database.
    
    Uses pydantic-ai to generate structured enrichments:
    - Sentiment analysis (positive/negative/neutral)
    - Budget tier (low/medium/high/very_high)
    - Revenue tier (low/medium/high/very_high)
    - Effectiveness score (0-10)
    - Target audience (family/young_adult/adult/niche/broad)
    
    Enrichments are stored directly in the movies.db database.
    
    Examples:
        # Enrich all unenriched movies
        python main.py enrich
        
        # Enrich only 100 movies
        python main.py enrich --n 100
        
        # Re-enrich all movies (overwrite existing)
        python main.py enrich --all --overwrite
        
        # Resume enrichment (automatically skips enriched movies)
        python main.py enrich
    """
    import asyncio
    from src.enricher_db import DatabaseMovieEnricher
    
    typer.echo(f"🎬 Enriching movies using {model}...")
    if overwrite:
        typer.echo(f"⚠️  Overwrite mode: will re-enrich existing enrichments")
    typer.echo("=" * 80)
    
    async def run_enrichment():
        enricher = DatabaseMovieEnricher(model_name=model)
        results = await enricher.enrich_all(
            batch_size=batch_size,
            overwrite=overwrite or all,
            limit=n
        )
        
        typer.echo("\n✅ Enrichment complete!")
    
    asyncio.run(run_enrichment())


@app.command()
def enrich_one(
    movie_id: int = typer.Argument(..., help="Movie ID to enrich"),
    model: str = typer.Option("openai:gpt-4o-mini", help="LLM model to use"),
    overwrite: bool = typer.Option(False, help="Re-enrich if already enriched")
):
    """Enrich a single movie with LLM-generated attributes and store in database.
    
    Useful for testing and seeing detailed enrichment output.
    """
    import asyncio
    from src.enricher_db import DatabaseMovieEnricher
    from src.db import MovieEnrichment
    
    movie = Movie.get_by_id(movie_id)
    if not movie:
        typer.echo(f"❌ Movie {movie_id} not found", err=True)
        raise typer.Exit(1)
    
    # Check if already enriched
    existing = MovieEnrichment.get_by_id(movie_id)
    if existing and not overwrite:
        typer.echo(f"\n⚠️  Movie already enriched (use --overwrite to re-enrich)")
        typer.echo(f"\n✅ Existing Enrichment:")
        typer.echo(f"   Sentiment: {existing.sentiment}")
        typer.echo(f"   Budget Tier: {existing.budget_tier}")
        typer.echo(f"   Revenue Tier: {existing.revenue_tier}")
        typer.echo(f"   Effectiveness: {existing.effectiveness_score}/10")
        typer.echo(f"   Target Audience: {existing.target_audience}")
        if existing.reasoning:
            typer.echo(f"\n   💡 Reasoning:")
            typer.echo(f"   {existing.reasoning}")
        typer.echo(f"\n   Model: {existing.model_used}")
        typer.echo(f"   Enriched: {existing.enriched_at}")
        return
    
    typer.echo(f"\n🎬 Enriching: {movie.title}")
    typer.echo("=" * 80)
    
    async def run_single_enrichment():
        enricher = DatabaseMovieEnricher(model_name=model)
        result = await enricher.enrich_movie(movie_id, overwrite=overwrite)
        
        if result:
            typer.echo(f"\n✅ Enrichment Results:")
            typer.echo(f"   Sentiment: {result.sentiment}")
            typer.echo(f"   Budget Tier: {result.budget_tier}")
            typer.echo(f"   Revenue Tier: {result.revenue_tier}")
            typer.echo(f"   Effectiveness: {result.effectiveness_score}/10")
            typer.echo(f"   Target Audience: {result.target_audience}")
            
            if result.reasoning:
                typer.echo(f"\n   💡 Reasoning:")
                typer.echo(f"   {result.reasoning}")
            
            typer.echo("=" * 80 + "\n")
    
    asyncio.run(run_single_enrichment())


# ============================================================================
# Task 2: Movie Recommendation System Commands
# ============================================================================

@app.command()
def recommend(
    user_id: int = typer.Argument(..., help="User ID to recommend for"),
    query: Optional[str] = typer.Option(None, "--query", "-q", help="Natural language filter (e.g., 'action movies high revenue')"),
    n: int = typer.Option(10, help="Number of recommendations"),
    model: str = typer.Option("openai:gpt-4o-mini", help="LLM model to use"),
    use_enriched_only: bool = typer.Option(True, help="Only recommend from enriched movies")
):
    """Get personalized movie recommendations for a user.
    
    Uses LLM to analyze user's rating history and recommend movies they'll like.
    
    Examples:
        # Basic recommendations
        python main.py recommend 5
        
        # With natural language query
        python main.py recommend 5 --query "action movies with high revenue"
        python main.py recommend 5 -q "positive sentiment family films"
        
        # More recommendations
        python main.py recommend 5 --n 20
    """
    import asyncio
    from src.recommender import MovieRecommender
    
    typer.echo(f"\n🎬 Generating recommendations for User {user_id}")
    if query:
        typer.echo(f"   Query: \"{query}\"")
    typer.echo("=" * 80)
    
    async def run_recommendation():
        recommender = MovieRecommender(model_name=model)
        
        try:
            result = await recommender.recommend(user_id, query, n, use_enriched_only)
            
            typer.echo(f"\n👤 User Profile:")
            typer.echo(f"   {result.user_profile_summary}\n")
            
            typer.echo(f"✅ Top {len(result.recommendations)} Recommendations:\n")
            
            for i, rec in enumerate(result.recommendations, 1):
                typer.echo(f"{i}. {rec.title} (ID: {rec.movie_id})")
                typer.echo(f"   Match Score: {rec.match_score}/10")
                typer.echo(f"   💡 {rec.reasoning}")
                typer.echo("")
            
            typer.echo("=" * 80 + "\n")
            
        except ValueError as e:
            typer.echo(f"❌ Error: {e}", err=True)
            raise typer.Exit(1)
    
    asyncio.run(run_recommendation())


@app.command()
def predict(
    user_id: int = typer.Argument(..., help="User ID"),
    movie_id: int = typer.Argument(..., help="Movie ID to predict rating for"),
    model: str = typer.Option("openai:gpt-4o-mini", help="LLM model to use")
):
    """Predict what rating a user would give a movie.
    
    Uses LLM to analyze user's preferences and predict their rating.
    
    Examples:
        # Predict user 5's rating for Fight Club (movie 550)
        python main.py predict 5 550
        
        # Predict using Claude
        python main.py predict 5 550 --model "anthropic:claude-3-5-sonnet-20241022"
    """
    import asyncio
    from src.recommender import MovieRecommender
    
    movie = Movie.get_by_id(movie_id)
    if not movie:
        typer.echo(f"❌ Movie {movie_id} not found", err=True)
        raise typer.Exit(1)
    
    typer.echo(f"\n🔮 Predicting rating for User {user_id}")
    typer.echo(f"   Movie: {movie.title}")
    typer.echo("=" * 80)
    
    async def run_prediction():
        recommender = MovieRecommender(model_name=model)
        
        try:
            result = await recommender.predict_rating(user_id, movie_id)
            
            typer.echo(f"\n✅ Prediction:")
            typer.echo(f"   Predicted Rating: {result.predicted_rating:.1f}/5.0")
            typer.echo(f"   Confidence: {result.confidence}")
            typer.echo(f"\n   💡 Reasoning:")
            typer.echo(f"   {result.reasoning}")
            typer.echo("\n" + "=" * 80 + "\n")
            
        except ValueError as e:
            typer.echo(f"❌ Error: {e}", err=True)
            raise typer.Exit(1)
    
    asyncio.run(run_prediction())


@app.command()
def preferences(
    user_id: int = typer.Argument(..., help="User ID to analyze"),
    model: str = typer.Option("openai:gpt-4o-mini", help="LLM model to use")
):
    """Summarize a user's movie preferences.
    
    Analyzes user's rating history to generate a natural language summary
    of their taste in movies.
    
    Examples:
        # Get user 5's preferences
        python main.py preferences 5
        
        # Use Claude for better analysis
        python main.py preferences 5 --model "anthropic:claude-3-5-sonnet-20241022"
    """
    import asyncio
    from src.recommender import MovieRecommender
    
    typer.echo(f"\n👤 Analyzing preferences for User {user_id}")
    typer.echo("=" * 80)
    
    async def run_analysis():
        recommender = MovieRecommender(model_name=model)
        
        try:
            summary = await recommender.summarize_preferences(user_id)
            
            typer.echo(f"\n✅ User Preference Summary:\n")
            typer.echo(f"   {summary}")
            typer.echo("\n" + "=" * 80 + "\n")
            
        except ValueError as e:
            typer.echo(f"❌ Error: {e}", err=True)
            raise typer.Exit(1)
    
    asyncio.run(run_analysis())


@app.command()
def compare(
    movies: str = typer.Argument(..., help="Comma-separated movie IDs (e.g., '550,603,680')"),
    model: str = typer.Option("openai:gpt-4o-mini", help="LLM model to use")
):
    """Compare multiple movies across various dimensions.
    
    Provides comparative analysis of budget, revenue, ratings, themes, etc.
    
    Examples:
        # Compare Fight Club, The Matrix, and Inception
        python main.py compare "550,603,680"
        
        # Compare with Claude
        python main.py compare "550,603" --model "anthropic:claude-3-5-sonnet-20241022"
    """
    import asyncio
    from src.recommender import MovieRecommender
    
    # Parse movie IDs
    try:
        movie_ids = [int(mid.strip()) for mid in movies.split(",")]
    except ValueError:
        typer.echo("❌ Invalid movie IDs. Use comma-separated numbers (e.g., '550,603,680')", err=True)
        raise typer.Exit(1)
    
    if len(movie_ids) < 2:
        typer.echo("❌ Need at least 2 movies to compare", err=True)
        raise typer.Exit(1)
    
    # Get movie titles
    movie_titles = []
    for mid in movie_ids:
        movie = Movie.get_by_id(mid)
        if movie:
            movie_titles.append(movie.title)
        else:
            typer.echo(f"⚠️  Warning: Movie {mid} not found", err=True)
    
    typer.echo(f"\n🎬 Comparing {len(movie_ids)} movies:")
    for i, title in enumerate(movie_titles, 1):
        typer.echo(f"   {i}. {title}")
    typer.echo("=" * 80)
    
    async def run_comparison():
        recommender = MovieRecommender(model_name=model)
        
        try:
            result = await recommender.compare_movies(movie_ids)
            
            typer.echo(f"\n📊 Comparison Summary:")
            typer.echo(f"   {result.comparison_summary}\n")
            
            typer.echo(f"🏆 Strongest Movie:")
            typer.echo(f"   {result.strongest_movie}\n")
            
            typer.echo(f"🔍 Key Differences:")
            for i, diff in enumerate(result.key_differences, 1):
                typer.echo(f"   {i}. {diff}")
            
            typer.echo(f"\n💡 Recommendations:")
            typer.echo(f"   {result.recommendations}")
            
            typer.echo("\n" + "=" * 80 + "\n")
            
        except ValueError as e:
            typer.echo(f"❌ Error: {e}", err=True)
            raise typer.Exit(1)
    
    asyncio.run(run_comparison())


# ============================================================================
# Hybrid Recommendation System Commands (Fast ML-based)
# ============================================================================

@app.command()
def train(
    factors: int = typer.Option(64, help="Number of latent factors for ALS"),
    regularization: float = typer.Option(0.01, help="L2 regularization parameter"),
    iterations: int = typer.Option(20, help="Number of ALS iterations"),
    save_path: str = typer.Option("models/hybrid_recommender.pkl", help="Where to save trained model")
):
    """Train hybrid recommendation model (ALS + LLM features).
    
    Trains a fast, scalable recommendation model that combines:
    - Collaborative filtering (implicit ALS)
    - LLM-enriched movie features
    
    The model is saved to disk and can be used for fast predictions.
    
    Examples:
        # Train with default settings
        python main.py train
        
        # Train with more factors for better accuracy
        python main.py train --factors 128 --iterations 30
        
        # Save to custom location
        python main.py train --save-path "my_model.pkl"
    """
    from src.hybrid_recommender import HybridRecommender
    
    typer.echo("🚀 Training hybrid recommendation model...")
    typer.echo(f"   Factors: {factors}")
    typer.echo(f"   Regularization: {regularization}")
    typer.echo(f"   Iterations: {iterations}")
    typer.echo("=" * 80)
    
    recommender = HybridRecommender(
        factors=factors,
        regularization=regularization,
        iterations=iterations
    )
    
    recommender.train(verbose=True)
    recommender.save(save_path)
    
    typer.echo("\n" + "=" * 80)
    typer.echo(f"✅ Model trained and saved to {save_path}")
    typer.echo("\n💡 Use 'python main.py fast-recommend <user_id>' for fast predictions")


@app.command()
def fast_recommend(
    user_id: int = typer.Argument(..., help="User ID to recommend for"),
    n: int = typer.Option(10, help="Number of recommendations"),
    model_path: str = typer.Option("models/hybrid_recommender.pkl", help="Path to trained model"),
    with_llm_explanations: bool = typer.Option(False, "--explain", help="Add LLM explanations (slower)")
):
    """Get fast movie recommendations using trained hybrid model.
    
    Uses pre-trained ALS model for instant recommendations (<100ms).
    Optionally adds LLM explanations for interpretability.
    
    Examples:
        # Fast recommendations (milliseconds)
        python main.py fast-recommend 5
        
        # With LLM explanations
        python main.py fast-recommend 5 --explain
        
        # More recommendations
        python main.py fast-recommend 5 --n 20
    """
    from pathlib import Path
    typer.echo(f"\n⚡ Loading model...")
    from src.hybrid_recommender import load_or_train
    recommender = load_or_train(model_path, verbose=True)
    
    typer.echo(f"🎬 Generating recommendations for User {user_id}")
    typer.echo("=" * 80)
    
    try:
        # Get fast recommendations
        import time
        start = time.time()
        recs = recommender.recommend(user_id, n=n)
        elapsed = (time.time() - start) * 1000  # Convert to ms
        
        typer.echo(f"\n✅ Top {len(recs)} Recommendations (generated in {elapsed:.1f}ms):\n")
        
        # Get all movies at once to avoid session issues
        # Convert np.int64 to Python int for SQL compatibility
        movie_ids = [int(movie_id) for movie_id, _ in recs]
        
        # Fetch all movies in a single session
        with Movie.get_session() as session:
            from sqlmodel import select
            stmt = select(Movie).where(Movie.movieId.in_(movie_ids))
            movies_list = session.exec(stmt).all()
            # Create a dict for quick lookup
            movies_dict = {m.movieId: m for m in movies_list}
        
        for i, (movie_id, score) in enumerate(recs, 1):
            movie = movies_dict.get(movie_id)
            
            if movie:
                typer.echo(f"{i}. {movie.title} (ID: {movie_id})")
                typer.echo(f"   Score: {score:.4f}")
                
                # Optional: Add LLM explanation
                if with_llm_explanations and i <= 5:  # Only explain top 5
                    import asyncio
                    from src.recommender import MovieRecommender
                    
                    async def get_explanation():
                        llm = MovieRecommender()
                        result = await llm.predict_rating(user_id, movie_id)
                        return result.reasoning
                    
                    reasoning = asyncio.run(get_explanation())
                    typer.echo(f"   💡 {reasoning}")
                
                typer.echo("")
        
        typer.echo("=" * 80 + "\n")
        
    except ValueError as e:
        typer.echo(f"❌ Error: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def eval_model(
    model_path: str = typer.Option("models/hybrid_recommender.pkl", help="Path to trained model"),
    k: int = typer.Option(10, help="Top-K for evaluation metrics")
):
    """Evaluate trained model performance.
    
    Calculates precision@k and NDCG@k on test data.
    
    Examples:
        python main.py eval-model
        python main.py eval-model --k 20
    """
    from pathlib import Path
    
    typer.echo(f"📊 Evaluating model...")
    typer.echo("=" * 80)
    
    # Load model (or train if needed)
    from src.hybrid_recommender import load_or_train
    recommender = load_or_train(model_path, verbose=True)
    
    # Get test ratings (last 20% of each user's ratings)
    all_ratings = Rating.get_session().__enter__().exec(
        __import__('sqlmodel').select(Rating)
    ).all()
    
    # Split into test set (last 20% per user)
    from collections import defaultdict
    user_ratings = defaultdict(list)
    
    for rating in all_ratings:
        user_ratings[rating.userId].append((rating.movieId, rating.rating, rating.timestamp))
    
    test_ratings = []
    for user_id, ratings in user_ratings.items():
        # Sort by timestamp and take last 20%
        ratings.sort(key=lambda x: x[2])
        n_test = max(1, len(ratings) // 5)
        test_items = ratings[-n_test:]
        
        for movie_id, rating, _ in test_items:
            test_ratings.append((user_id, movie_id, rating))
    
    typer.echo(f"   Test set: {len(test_ratings):,} ratings\n")
    
    # Evaluate
    metrics = recommender.evaluate(test_ratings, k=k)
    
    typer.echo(f"✅ Evaluation Results (k={k}):\n")
    typer.echo(f"   Precision@{k}: {metrics['precision@k']:.4f}")
    typer.echo(f"   NDCG@{k}: {metrics['ndcg@k']:.4f}")
    typer.echo(f"   Users evaluated: {metrics['n_users_evaluated']:,}")
    typer.echo("\n" + "=" * 80 + "\n")


# ============================================================================
# MCP Server and Chat Commands
# ============================================================================

@app.command()
def mcp_server():
    """Launch the MCP server on stdio.
    
    This starts the Movie Database MCP server which exposes movie database
    operations as MCP tools. The server runs on stdio and can be used by
    AI agents or other MCP clients.
    
    Available MCP tools:
    - search_movies: Search for movies by title
    - get_movie_details: Get detailed movie information
    - get_user_ratings: Get user rating history
    - get_movie_recommendations: Get ML-based recommendations
    - compare_movies: Compare multiple movies
    - get_random_movies: Get random movies
    
    Examples:
        # Launch MCP server
        python main.py mcp-server
        
        # In Claude Desktop config:
        {
          "mcpServers": {
            "movie-db": {
              "command": "uv",
              "args": ["run", "python", "main.py", "mcp-server"]
            }
          }
        }
    """
    from src.mcp_server import mcp
    
    typer.echo("🚀 Starting Movie Database MCP Server on stdio...")
    typer.echo("   Available tools: search_movies, get_movie_details, get_user_ratings,")
    typer.echo("                   get_movie_recommendations, compare_movies, get_random_movies")
    typer.echo("")
    
    # Run the MCP server
    mcp.run()


@app.command()
def chat(
    query: Optional[str] = typer.Argument(None, help="Query to ask (non-interactive mode). If not provided, enters interactive mode."),
    model: str = typer.Option("openai:gpt-4o-mini", help="LLM model to use"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Force interactive mode even with query"),
    system_prompt: Optional[str] = typer.Option(None, "--system-prompt", help="Custom system prompt for the agent")
):
    """Chat with movie database assistant (non-interactive by default, interactive with -i).
    
    Default mode: Single query execution (provide query as argument)
    Interactive mode: Continuous conversation (use --interactive or -i flag, or don't provide query)
    
    The assistant has access to the movie database through MCP tools and can:
    - Search for movies
    - Provide detailed movie information
    - Show user preferences and ratings
    - Generate personalized recommendations
    - Compare movies
    - Discover random movies
    
    Examples:
        # Non-interactive: Single query (default)
        python main.py chat "Find movies about space"
        python main.py chat "What are user 5's top rated movies?"
        python main.py chat "Recommend action movies for user 10"
        
        # Interactive: Continuous conversation
        python main.py chat --interactive
        python main.py chat -i
        
        # With custom model
        python main.py chat "Find sci-fi movies" --model "anthropic:claude-3-5-sonnet-20241022"
        
        # With custom system prompt
        python main.py chat "Recommend movies" --system-prompt "You are a movie critic who only likes indie films."
        
        # Force interactive even with query in args
        python main.py chat --interactive --model "openai:gpt-4o-mini"
    """
    from src.chat_agent import MovieChatCLI
    
    # If interactive flag is set, ignore query and enter interactive mode
    if interactive:
        query = None
    
    if query:
        # Non-interactive mode
        cli = MovieChatCLI(model_name=model, system_prompt=system_prompt)
        cli.run(query=query)
    else:
        # Interactive mode
        typer.echo(f"\n🤖 Starting movie chat assistant with {model}...\n")
        cli = MovieChatCLI(model_name=model, system_prompt=system_prompt)
        cli.run()


# ============================================================================
# Embedding Commands (Vector Search with sqlite-vec)
# ============================================================================

@app.command()
def embed_stats():
    """Show statistics about movie embeddings."""
    from src.embeddings import init_embeddings
    
    typer.echo("\n" + "=" * 70)
    typer.echo("📊 Movie Embeddings Statistics")
    typer.echo("=" * 70)
    
    embeddings = init_embeddings()
    stats = embeddings.get_stats()
    
    typer.echo(f"\n📈 Coverage:")
    typer.echo(f"   Total Movies:     {stats['total_movies']}")
    typer.echo(f"   With Embeddings:  {stats['total_embeddings']}")
    typer.echo(f"   Coverage:         {stats['coverage']}")
    
    typer.echo(f"\n🧮 Model:")
    typer.echo(f"   Model:       {stats['model']}")
    typer.echo(f"   Dimensions:  {stats['dimensions']}")
    
    typer.echo("\n" + "=" * 70)


@app.command()
def embed_generate(
    movie_ids: str = typer.Argument(..., help="Comma-separated movie IDs (e.g., '550,603,13')"),
    force: bool = typer.Option(False, "--force", "-f", help="Regenerate even if embedding exists")
):
    """Generate embeddings for specific movies.
    
    Examples:
        python main.py embed-generate 550
        python main.py embed-generate "550,603,13"
        python main.py embed-generate "550,603,13" --force
    """
    from src.embeddings import init_embeddings
    
    # Parse movie IDs
    ids = [int(x.strip()) for x in movie_ids.split(",")]
    
    typer.echo(f"\n🔧 Generating embeddings for {len(ids)} movie(s)...")
    
    embeddings = init_embeddings()
    stats = embeddings.generate_batch(ids, force=force)
    
    typer.echo(f"\n✅ Results:")
    typer.echo(f"   Generated: {stats['generated']}")
    typer.echo(f"   Skipped:   {stats['skipped']}")
    typer.echo(f"   Failed:    {stats['failed']}")


@app.command()
def embed_all(
    force: bool = typer.Option(False, "--force", "-f", help="Regenerate all embeddings"),
    limit: int = typer.Option(None, "--limit", "-n", help="Limit number of movies to process")
):
    """Generate embeddings for all movies with overview data.
    
    This will:
    1. Load the sentence-transformer model (all-MiniLM-L6-v2)
    2. Generate embeddings for title + overview
    3. Store in sqlite-vec virtual table
    
    Examples:
        python main.py embed-all                 # Generate for all unenriched
        python main.py embed-all --limit 100     # Process 100 movies
        python main.py embed-all --force         # Regenerate all
    """
    from src.embeddings import init_embeddings
    from src.db import Movie
    
    typer.echo("\n" + "=" * 70)
    typer.echo("🔧 Generating Movie Embeddings")
    typer.echo("=" * 70)
    
    embeddings = init_embeddings()
    
    # Get movies to process
    movies_df = Movie.get_all_with_budget()
    movie_ids = movies_df['movieId'].tolist()
    
    if limit:
        movie_ids = movie_ids[:limit]
    
    typer.echo(f"\n📝 Processing {len(movie_ids)} movies...")
    
    stats = embeddings.generate_batch(movie_ids, force=force)
    
    typer.echo(f"\n✅ Completed!")
    typer.echo(f"   Generated: {stats['generated']}")
    typer.echo(f"   Skipped:   {stats['skipped']}")
    typer.echo(f"   Failed:    {stats['failed']}")
    
    # Show updated stats
    typer.echo("\n" + "=" * 70)
    final_stats = embeddings.get_stats()
    typer.echo(f"📊 Total Coverage: {final_stats['coverage']} ({final_stats['total_embeddings']}/{final_stats['total_movies']})")
    typer.echo("=" * 70)


@app.command()
def semantic_search(
    query: str = typer.Argument(..., help="Natural language search query"),
    k: int = typer.Option(10, "--k", "-n", help="Number of results to return"),
    genre: Optional[str] = typer.Option(None, "--genre", "-g", help="Filter by genre (e.g., 'Action', 'Sci-Fi')"),
    min_budget: Optional[int] = typer.Option(None, "--min-budget", help="Minimum budget"),
    max_budget: Optional[int] = typer.Option(None, "--max-budget", help="Maximum budget"),
    with_enrichment: bool = typer.Option(True, "--with-enrichment/--no-enrichment", help="Include enrichment data")
):
    """Semantic search using vector embeddings.
    
    Uses sqlite-vec for fast K-nearest neighbor search with optional metadata filters.
    
    Examples:
        # Basic search
        python main.py semantic-search "dark psychological thriller"
        
        # Search with genre filter
        python main.py semantic-search "space adventure" --genre "Sci-Fi"
        
        # Search with budget constraints
        python main.py semantic-search "action movie" --min-budget 50000000
        
        # Search with multiple filters
        python main.py semantic-search "family comedy" --genre "Comedy" --min-budget 20000000 -n 5
    """
    from src.embeddings import init_embeddings
    
    typer.echo(f"\n🔍 Searching for: \"{query}\"")
    
    # Build filters
    filters = {}
    if genre:
        filters["genres__contains"] = genre
    if min_budget:
        filters["budget__gte"] = min_budget
    if max_budget:
        filters["budget__lte"] = max_budget
    
    if filters:
        typer.echo(f"📊 Filters: {filters}")
    
    # Perform search
    embeddings = init_embeddings()
    results = embeddings.hybrid_search(
        query=query,
        k=k,
        filters=filters if filters else None,
        enrich_results=with_enrichment
    )
    
    if not results:
        typer.echo("\n❌ No results found")
        return
    
    # Display results
    typer.echo(f"\n🎬 Found {len(results)} movies:\n")
    typer.echo("=" * 100)
    
    for i, result in enumerate(results, 1):
        typer.echo(f"\n{i}. {result['title']} (ID: {result['movie_id']})")
        typer.echo(f"   Similarity: {result['similarity']:.3f} | Distance: {result['distance']:.3f}")
        
        if result['budget'] and result['revenue']:
            typer.echo(f"   Budget: ${result['budget']:,} | Revenue: ${result['revenue']:,}")
        
        if result['genres']:
            typer.echo(f"   Genres: {result['genres'][:50]}")
        
        if with_enrichment and 'enrichment' in result:
            enr = result['enrichment']
            typer.echo(f"   Enrichment: {enr['sentiment']} | {enr['budget_tier']} budget | {enr['target_audience']} audience")
            typer.echo(f"   Effectiveness: {enr['effectiveness_score']:.1f}/10")
        
        if result['overview']:
            overview = result['overview'][:150] + "..." if len(result['overview']) > 150 else result['overview']
            typer.echo(f"   Overview: {overview}")
    
    typer.echo("\n" + "=" * 100)


if __name__ == "__main__":
    app()

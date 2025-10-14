"""Database-backed movie enrichment using pydantic-ai."""

from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field
from pydantic_ai import Agent

from src.db import Movie, Rating, MovieEnrichment
from src.enricher import MovieEnrichment as EnrichmentSchema, SentimentType, BudgetTier, RevenueTier, TargetAudience


class DatabaseMovieEnricher:
    """LLM-powered movie enricher that writes directly to database."""
    
    def __init__(self, model_name: str = "openai:gpt-4o-mini"):
        """Initialize enricher with pydantic-ai agent.
        
        Args:
            model_name: LLM model to use (e.g., "openai:gpt-4o-mini")
        """
        self.model_name = model_name
        
        # Create pydantic-ai agent with structured output
        self.agent = Agent(
            model_name,
            output_type=EnrichmentSchema,
            system_prompt="""You are an expert film analyst and data scientist.
            
Your task is to analyze movie data and provide structured enrichments including:
1. Sentiment analysis of the overview
2. Budget and revenue tier categorization
3. Production effectiveness scoring
4. Target audience identification

Be analytical, consistent, and provide thoughtful reasoning for your assessments.
Consider industry standards, historical context, and the movie's characteristics."""
        )
        
        # Create table if it doesn't exist
        MovieEnrichment.create_table()
    
    async def enrich_movie(
        self,
        movie_id: int,
        overwrite: bool = False
    ) -> Optional[MovieEnrichment]:
        """Enrich a single movie and store in database.
        
        Args:
            movie_id: Movie ID to enrich
            overwrite: If False, skip if already enriched
            
        Returns:
            MovieEnrichment record or None if skipped
        """
        # Check if already enriched
        if not overwrite and MovieEnrichment.exists(movie_id):
            return None
        
        # Get movie data
        movie = Movie.get_by_id(movie_id)
        if not movie:
            print(f"⚠️  Movie {movie_id} not found")
            return None
        
        # Get average rating
        ratings = movie.get_ratings()
        avg_rating = sum(r.rating for r in ratings) / len(ratings) if ratings else None
        
        # Build context for LLM
        budget_str = f"${movie.budget:,}" if movie.budget else "Unknown"
        revenue_str = f"${movie.revenue:,}" if movie.revenue else "Unknown"
        rating_str = f"{avg_rating:.2f}/5.0" if avg_rating else "No ratings yet"
        
        context = f"""
Movie: {movie.title}

Overview: {movie.overview or "No overview available"}

Budget: {budget_str}
Revenue: {revenue_str}
Genres: {movie.genres or "Unknown"}
Average Rating: {rating_str}

Analyze this movie and provide structured enrichment data.
"""
        
        # Run LLM enrichment
        result = await self.agent.run(context)
        enrichment_data = result.output
        
        # Create database record
        db_enrichment = MovieEnrichment(
            movieId=movie_id,
            sentiment=enrichment_data.sentiment.value,
            budget_tier=enrichment_data.budget_tier.value,
            revenue_tier=enrichment_data.revenue_tier.value,
            effectiveness_score=enrichment_data.effectiveness_score,
            target_audience=enrichment_data.target_audience.value,
            reasoning=enrichment_data.reasoning,
            avg_rating=avg_rating,
            model_used=self.model_name,
            enriched_at=datetime.utcnow().isoformat()
        )
        
        # Store in database
        MovieEnrichment.upsert(db_enrichment)
        
        return db_enrichment
    
    async def enrich_batch(
        self,
        movie_ids: list[int],
        overwrite: bool = False,
        show_progress: bool = True
    ) -> dict:
        """Enrich a batch of movies.
        
        Args:
            movie_ids: List of movie IDs to enrich
            overwrite: If False, skip already enriched movies
            show_progress: Show progress updates
            
        Returns:
            Dict with counts of enriched, skipped, and failed
        """
        enriched = 0
        skipped = 0
        failed = 0
        
        for i, movie_id in enumerate(movie_ids, 1):
            if show_progress and i % 10 == 0:
                print(f"  Progress: {i}/{len(movie_ids)} movies processed")
            
            try:
                result = await self.enrich_movie(movie_id, overwrite=overwrite)
                if result is None:
                    skipped += 1
                else:
                    enriched += 1
            except Exception as e:
                print(f"  ❌ Failed to enrich movie {movie_id}: {e}")
                failed += 1
        
        return {
            "enriched": enriched,
            "skipped": skipped,
            "failed": failed
        }
    
    async def enrich_all(
        self,
        batch_size: int = 100,
        overwrite: bool = False,
        limit: Optional[int] = None
    ) -> dict:
        """Enrich all unenriched movies in batches.
        
        Args:
            batch_size: Number of movies to process per batch
            overwrite: If False, only enrich movies without enrichments
            limit: Optional limit on total movies to enrich
            
        Returns:
            Dict with summary statistics
        """
        print("🎬 Starting database enrichment...")
        print("=" * 80)
        
        # Get unenriched movies
        if overwrite:
            # Get all movies with budget
            df = Movie.get_all_with_budget()
            movie_ids = df['movieId'].tolist()
            print(f"📊 Overwrite mode: re-enriching ALL {len(movie_ids)} movies")
        else:
            movie_ids = MovieEnrichment.get_unenriched_movies(limit=limit)
            enriched_count = MovieEnrichment.count_enriched()
            print(f"📊 Found {enriched_count} already enriched movies")
            print(f"📊 Found {len(movie_ids)} unenriched movies")
        
        if not movie_ids:
            print("✅ All movies already enriched!")
            return {"enriched": 0, "skipped": 0, "failed": 0}
        
        if limit and len(movie_ids) > limit:
            movie_ids = movie_ids[:limit]
            print(f"📊 Limiting to {limit} movies")
        
        print(f"🔄 Processing in batches of {batch_size}...")
        print("=" * 80)
        
        # Process in batches
        total_enriched = 0
        total_skipped = 0
        total_failed = 0
        
        for i in range(0, len(movie_ids), batch_size):
            batch = movie_ids[i:i+batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(movie_ids) + batch_size - 1) // batch_size
            
            print(f"\n🔄 Batch {batch_num}/{total_batches} ({len(batch)} movies)...")
            
            results = await self.enrich_batch(batch, overwrite=overwrite, show_progress=False)
            
            total_enriched += results["enriched"]
            total_skipped += results["skipped"]
            total_failed += results["failed"]
            
            print(f"  ✅ Enriched: {results['enriched']}, "
                  f"Skipped: {results['skipped']}, "
                  f"Failed: {results['failed']}")
        
        print("\n" + "=" * 80)
        print("📊 Final Summary:")
        print(f"  Total enriched: {total_enriched}")
        print(f"  Total skipped: {total_skipped}")
        print(f"  Total failed: {total_failed}")
        print(f"  Database total: {MovieEnrichment.count_enriched()}")
        print("=" * 80)
        
        return {
            "enriched": total_enriched,
            "skipped": total_skipped,
            "failed": total_failed
        }


if __name__ == "__main__":
    """Test the database enricher."""
    import asyncio
    
    async def test():
        enricher = DatabaseMovieEnricher()
        
        # Test single movie
        print("Testing single movie enrichment...")
        result = await enricher.enrich_movie(550, overwrite=True)
        
        if result:
            print(f"✅ Enriched movie {result.movieId}")
            print(f"   Sentiment: {result.sentiment}")
            print(f"   Effectiveness: {result.effectiveness_score}/10")
    
    asyncio.run(test())

"""LLM-powered movie data enrichment using pydantic-ai."""

from typing import Optional, Literal
from enum import Enum
import pandas as pd
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

from src.db import Movie, Rating


# Enums for categorical fields
class SentimentType(str, Enum):
    """Sentiment categories for movie overviews."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class BudgetTier(str, Enum):
    """Budget tier categories."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class RevenueTier(str, Enum):
    """Revenue tier categories."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class TargetAudience(str, Enum):
    """Target audience categories."""
    FAMILY = "family"
    YOUNG_ADULT = "young_adult"
    ADULT = "adult"
    NICHE = "niche"
    BROAD = "broad"


# Pydantic models for structured LLM outputs
class MovieEnrichment(BaseModel):
    """Structured enrichment data for a single movie.
    
    This model ensures type-safe, validated outputs from the LLM.
    """
    sentiment: SentimentType = Field(
        description="Sentiment of the movie overview: positive, negative, or neutral"
    )
    budget_tier: BudgetTier = Field(
        description="Budget tier based on production cost: low (<$10M), medium ($10-50M), high ($50-150M), very_high (>$150M)"
    )
    revenue_tier: RevenueTier = Field(
        description="Revenue tier based on box office: low (<$50M), medium ($50-200M), high ($200-500M), very_high (>$500M)"
    )
    effectiveness_score: float = Field(
        ge=0.0,
        le=10.0,
        description="Production effectiveness score (0-10) based on ROI, ratings, and cultural impact"
    )
    target_audience: TargetAudience = Field(
        description="Primary target audience: family, young_adult, adult, niche, or broad"
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Brief explanation of the enrichment decisions"
    )


class MovieEnricher:
    """LLM-powered movie data enrichment using pydantic-ai.
    
    Uses Pydantic models for structured, type-safe LLM outputs.
    """
    
    def __init__(self, model_name: str = "openai:gpt-4o-mini"):
        """Initialize the enricher with a pydantic-ai agent.
        
        Args:
            model_name: Model to use (e.g., "openai:gpt-4o-mini", "openai:gpt-4-turbo")
        """
        # Create pydantic-ai agent with structured output
        self.agent = Agent(
            model_name,
            output_type=MovieEnrichment,
            system_prompt="""You are an expert film analyst and data scientist.
            
Your task is to analyze movie data and provide structured enrichments including:
1. Sentiment analysis of the overview
2. Budget and revenue tier categorization
3. Production effectiveness scoring
4. Target audience identification

Be analytical, consistent, and provide thoughtful reasoning for your assessments.
Consider industry standards, historical context, and the movie's characteristics."""
        )
    
    async def enrich_movie(
        self,
        title: str,
        overview: str,
        budget: Optional[int],
        revenue: Optional[int],
        genres: Optional[str],
        avg_rating: Optional[float] = None
    ) -> MovieEnrichment:
        """Enrich a single movie with LLM-generated attributes.
        
        Args:
            title: Movie title
            overview: Movie description/overview
            budget: Production budget in USD
            revenue: Box office revenue in USD
            genres: Movie genres (JSON string)
            avg_rating: Average user rating (0-5)
            
        Returns:
            MovieEnrichment with all structured fields
        """
        # Build context for the LLM
        budget_str = f"${budget:,}" if budget else "Unknown"
        revenue_str = f"${revenue:,}" if revenue else "Unknown"
        rating_str = f"{avg_rating:.2f}/5.0" if avg_rating else "No ratings yet"
        
        context = f"""
Movie: {title}

Overview: {overview or "No overview available"}

Budget: {budget_str}
Revenue: {revenue_str}
Genres: {genres or "Unknown"}
Average Rating: {rating_str}

Analyze this movie and provide structured enrichment data.
"""
        
        # Run pydantic-ai agent - automatically parses into MovieEnrichment
        result = await self.agent.run(context)
        return result.output
    
    async def enrich_sample(
        self,
        n: int = 50,
        with_budget: bool = True,
        include_ratings: bool = True
    ) -> pd.DataFrame:
        """Enrich a sample of movies with LLM-generated attributes.
        
        Args:
            n: Number of movies to enrich
            with_budget: Filter for movies with budget data
            include_ratings: Include average ratings in analysis
            
        Returns:
            DataFrame with original movie data + enrichment columns
        """
        # Get sample from database
        df = Movie.get_random(n=n, with_budget=with_budget)
        
        # Add average ratings if requested
        if include_ratings:
            avg_ratings = []
            for movie_id in df['movieId']:
                ratings = Rating.get_for_movie(movie_id)
                avg = sum(r.rating for r in ratings) / len(ratings) if ratings else None
                avg_ratings.append(avg)
            df['avg_rating'] = avg_ratings
        
        # Enrich each movie
        enrichments = []
        for _, row in df.iterrows():
            enrichment = await self.enrich_movie(
                title=row['title'],
                overview=row['overview'],
                budget=row['budget'],
                revenue=row['revenue'],
                genres=row['genres'],
                avg_rating=row.get('avg_rating')
            )
            enrichments.append(enrichment.model_dump())
        
        # Convert enrichments to DataFrame
        enrichment_df = pd.DataFrame(enrichments)
        
        # Merge with original data
        result_df = pd.concat([df.reset_index(drop=True), enrichment_df], axis=1)
        
        return result_df
    
    async def enrich_all(
        self,
        output_path: str = "enriched_movies.csv",
        overwrite: bool = False,
        include_ratings: bool = True,
        batch_size: int = 100
    ) -> pd.DataFrame:
        """Enrich all movies in the database with progress tracking.
        
        Args:
            output_path: Path to save/append CSV
            overwrite: If False, skip movies already in output file
            include_ratings: Include average ratings in analysis
            batch_size: Number of movies to process per batch
            
        Returns:
            Complete enriched DataFrame
        """
        from pathlib import Path
        import asyncio
        
        # Load existing enrichments if file exists and not overwriting
        existing_df = None
        existing_ids = set()
        
        if not overwrite and Path(output_path).exists():
            existing_df = pd.read_csv(output_path)
            existing_ids = set(existing_df['movieId'].tolist())
            print(f"📊 Found {len(existing_ids)} already enriched movies in {output_path}")
        
        # Get all movies with budget data
        all_movies_df = Movie.get_all_with_budget()
        total = len(all_movies_df)
        
        # Filter out already enriched movies
        if existing_ids:
            all_movies_df = all_movies_df[~all_movies_df['movieId'].isin(existing_ids)]
            print(f"🎬 {len(all_movies_df)} movies remaining to enrich (out of {total} total)")
        else:
            print(f"🎬 Enriching all {total} movies")
        
        if len(all_movies_df) == 0:
            print("✅ All movies already enriched!")
            return existing_df
        
        # Add average ratings if requested
        if include_ratings:
            print("📊 Calculating average ratings...")
            avg_ratings = []
            for movie_id in all_movies_df['movieId']:
                ratings = Rating.get_for_movie(movie_id)
                avg = sum(r.rating for r in ratings) / len(ratings) if ratings else None
                avg_ratings.append(avg)
            all_movies_df['avg_rating'] = avg_ratings
        
        # Process in batches with progress tracking
        enriched_batches = []
        for i in range(0, len(all_movies_df), batch_size):
            batch_df = all_movies_df.iloc[i:i+batch_size].copy()
            batch_num = (i // batch_size) + 1
            total_batches = (len(all_movies_df) + batch_size - 1) // batch_size
            
            print(f"\n🔄 Processing batch {batch_num}/{total_batches} ({len(batch_df)} movies)...")
            
            # Enrich each movie in batch
            enrichments = []
            for idx, row in batch_df.iterrows():
                enrichment = await self.enrich_movie(
                    title=row['title'],
                    overview=row['overview'],
                    budget=row['budget'],
                    revenue=row['revenue'],
                    genres=row['genres'],
                    avg_rating=row.get('avg_rating')
                )
                enrichments.append(enrichment.model_dump())
            
            # Convert enrichments to DataFrame
            enrichment_df = pd.DataFrame(enrichments)
            
            # Merge with original data
            batch_result = pd.concat([batch_df.reset_index(drop=True), enrichment_df], axis=1)
            enriched_batches.append(batch_result)
            
            print(f"✅ Batch {batch_num}/{total_batches} complete")
        
        # Combine all batches
        new_enrichments_df = pd.concat(enriched_batches, ignore_index=True)
        
        # Combine with existing enrichments
        if existing_df is not None and not overwrite:
            final_df = pd.concat([existing_df, new_enrichments_df], ignore_index=True)
        else:
            final_df = new_enrichments_df
        
        # Save to CSV
        final_df.to_csv(output_path, index=False)
        print(f"\n✅ Exported {len(final_df)} total enriched movies to {output_path}")
        print(f"   ({len(new_enrichments_df)} newly enriched)")
        
        return final_df
    
    async def enrich_and_export(
        self,
        n: int = 50,
        output_path: str = "enriched_movies.csv",
        overwrite: bool = False,
        batch_size: int = 100,
        **kwargs
    ) -> pd.DataFrame:
        """Enrich movies and export to CSV.
        
        Args:
            n: Number of movies to enrich (None for all)
            output_path: Path to save CSV
            overwrite: If False, skip movies already in output file
            batch_size: Batch size for processing (only used with enrich_all)
            **kwargs: Additional arguments for enrich_sample
            
        Returns:
            Enriched DataFrame
        """
        # If n is None or very large, use enrich_all
        if n is None or n >= 1000:
            return await self.enrich_all(
                output_path=output_path,
                overwrite=overwrite,
                batch_size=batch_size,
                **kwargs
            )
        
        df = await self.enrich_sample(n=n, **kwargs)
        df.to_csv(output_path, index=False)
        print(f"✅ Exported {len(df)} enriched movies to {output_path}")
        return df


# Convenience function for quick enrichment
async def enrich_movies(
    n: int = 50,
    model: str = "openai:gpt-3.5-turbo",
    output: str = "enriched_movies.csv"
) -> pd.DataFrame:
    """Quick function to enrich movies with default settings.
    
    Args:
        n: Number of movies to enrich
        model: LLM model to use
        output: Output CSV path
        
    Returns:
        Enriched DataFrame
    """
    enricher = MovieEnricher(model_name=model)
    return await enricher.enrich_and_export(n=n, output_path=output)


if __name__ == "__main__":
    """Test the enricher with a single movie."""
    import asyncio
    
    async def test_enricher():
        enricher = MovieEnricher()
        
        # Test with Fight Club
        movie = Movie.get_by_id(550)
        if movie:
            print(f"🎬 Enriching: {movie.title}")
            print("=" * 60)
            
            # Get ratings
            ratings = movie.get_ratings()
            avg_rating = sum(r.rating for r in ratings) / len(ratings) if ratings else None
            
            # Enrich
            enrichment = await enricher.enrich_movie(
                title=movie.title,
                overview=movie.overview,
                budget=movie.budget,
                revenue=movie.revenue,
                genres=movie.genres,
                avg_rating=avg_rating
            )
            
            print(f"\n✅ Enrichment Results:")
            print(f"   Sentiment: {enrichment.sentiment}")
            print(f"   Budget Tier: {enrichment.budget_tier}")
            print(f"   Revenue Tier: {enrichment.revenue_tier}")
            print(f"   Effectiveness: {enrichment.effectiveness_score}/10")
            print(f"   Target Audience: {enrichment.target_audience}")
            if enrichment.reasoning:
                print(f"\n   Reasoning: {enrichment.reasoning}")
            print("=" * 60)
    
    asyncio.run(test_enricher())

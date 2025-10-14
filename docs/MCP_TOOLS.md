# MCP Server Tools Reference

## Overview

The Movie Database MCP Server exposes **11 tools** for AI agents to interact with the movie database. These tools cover all requirements from the README:

- ✅ Data enrichment (sentiment, budget/revenue tiers, effectiveness scores)
- ✅ Movie recommendations (both ML-based fast and LLM-based intelligent)
- ✅ Rating predictions with reasoning
- ✅ Natural language querying (semantic search)
- ✅ User preference summaries
- ✅ Comparative analyses

## Tool Categories

### 1. Basic Database Operations

#### `search_movies(query: str, limit: int = 10)`
Search for movies by title (fuzzy matching).

**Example:**
```json
{
  "query": "inception",
  "limit": 5
}
```

#### `get_movie_details(movie_id: int)`
Get comprehensive movie information including enrichments.

**Returns:** Movie metadata, ratings, and enrichment data (sentiment, budget tier, effectiveness score, etc.)

#### `get_user_ratings(user_id: int, limit: Optional[int] = None)`
Retrieve a user's complete rating history.

**Returns:** User profile with average rating and list of all rated movies.

#### `get_random_movies(n: int = 10, with_enrichment: bool = False)`
Get random movie samples, optionally filtered to enriched movies only.

---

### 2. Recommendation Systems

#### `get_movie_recommendations(user_id: int, n: int = 10)` 
**Fast ML-based recommendations** using trained ALS model.

- Speed: ~100ms per user
- Method: Collaborative filtering (implicit ALS)
- Best for: Production use, high-volume requests

**Example:**
```json
{
  "user_id": 5,
  "recommendations": [
    {"movieId": 550, "title": "Fight Club", "score": 0.8432, "genres": "Drama|Thriller"}
  ]
}
```

#### `get_llm_recommendations(user_id: int, query: Optional[str] = None, n: int = 10)`
**Intelligent LLM-powered recommendations** with explanations.

- Speed: ~5-10s per user
- Method: LLM analyzes user preferences + enrichment data
- Best for: Interactive use, when explanations are needed

**Features:**
- Natural language query support ("action movies with high revenue")
- Detailed reasoning for each recommendation
- User profile summary
- Match scores (0-10)

**Example:**
```json
{
  "user_id": 5,
  "query": "dark psychological thrillers",
  "user_profile": "User enjoys complex narratives with dark themes...",
  "recommendations": [
    {
      "movieId": 550,
      "title": "Fight Club",
      "match_score": 9.2,
      "reasoning": "Matches user's preference for psychological depth and dark themes..."
    }
  ]
}
```

---

### 3. Prediction & Analysis

#### `predict_rating(user_id: int, movie_id: int)`
**Predict what rating a user would give a movie** they haven't seen.

Uses LLM to analyze user's rating history and movie attributes.

**Returns:**
- Predicted rating (0-5 scale)
- Confidence level (high/medium/low)
- Detailed reasoning

**Example:**
```json
{
  "user_id": 5,
  "movie_id": 680,
  "predicted_rating": 4.2,
  "confidence": "high",
  "reasoning": "Based on user's love of sci-fi and high ratings for cerebral films..."
}
```

#### `summarize_user_preferences(user_id: int)`
**Generate natural language summary of user's taste** in movies.

Analyzes rating history to extract:
- Favorite genres
- Budget preferences
- Sentiment preferences
- Target audience match

**Example:**
```json
{
  "user_id": 5,
  "preference_summary": "This user strongly prefers thought-provoking dramas and sci-fi films with complex narratives. They favor high-budget productions with darker, more serious tones..."
}
```

#### `compare_movies(movie_ids: List[int])`
**Compare multiple movies** across various dimensions.

Returns comparative analysis with:
- Budget, revenue, ratings comparison
- Enrichment data comparison
- Summary statistics

---

### 4. Semantic Search (NEW)

#### `semantic_search(query: str, k: int = 10, ...)`
**Natural language movie search** using vector embeddings.

Uses sqlite-vec for semantic similarity search with metadata filtering.

**Parameters:**
- `query`: Natural language search (e.g., "dark psychological thriller")
- `k`: Number of results
- `genre`: Filter by genre (optional)
- `min_budget`/`max_budget`: Budget range filters (optional)
- `with_enrichment`: Include enrichment data (default: True)

**Example:**
```json
{
  "query": "space adventure with high production value",
  "genre": "Sci-Fi",
  "min_budget": 50000000,
  "k": 5
}
```

**Returns:**
```json
{
  "query": "space adventure with high production value",
  "filters": {"genre": "Sci-Fi", "min_budget": 50000000},
  "results": [
    {
      "movie_id": 680,
      "title": "Pulp Fiction",
      "similarity": 0.892,
      "distance": 0.108,
      "genres": "Thriller|Crime",
      "budget": 8000000,
      "enrichment": {...}
    }
  ]
}
```

---

## README Requirements Alignment

| Requirement | MCP Tools | Status |
|-------------|-----------|--------|
| Data enrichment (sentiment, tiers, scores) | All tools return enrichment data | ✅ Complete |
| Movie recommendations | `get_movie_recommendations`, `get_llm_recommendations` | ✅ Complete |
| Rating predictions | `predict_rating` | ✅ Complete |
| Natural language querying | `semantic_search` | ✅ Complete |
| User preference summaries | `summarize_user_preferences` | ✅ Complete |
| Comparative analyses | `compare_movies` | ✅ Complete |
| Varied inputs support | All tools support flexible parameters | ✅ Complete |

---

## Usage Example

```bash
# Start MCP server
uv run python main.py mcp-server

# In Claude Desktop config (~/Library/Application Support/Claude/claude_desktop_config.json):
{
  "mcpServers": {
    "movie-db": {
      "command": "uv",
      "args": ["run", "python", "/path/to/main.py", "mcp-server"],
      "cwd": "/path/to/aetna-coding-challenge"
    }
  }
}
```

## Performance Notes

**Fast Tools (< 100ms):**
- `search_movies`
- `get_movie_details`
- `get_user_ratings`
- `get_movie_recommendations` (ML-based)
- `compare_movies`
- `semantic_search`

**Intelligent Tools (5-10s, requires LLM):**
- `get_llm_recommendations`
- `predict_rating`
- `summarize_user_preferences`

Choose the right tool based on your use case:
- Production/high-volume → Fast tools
- Interactive/explanatory → Intelligent tools

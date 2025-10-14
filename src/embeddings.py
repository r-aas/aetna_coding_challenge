"""Vector embeddings support for hybrid RAG retrieval with sqlite-vec.

This module provides:
- Embedding generation using sentence-transformers
- Vector table management with sqlite-vec
- Hybrid RAG retrieval methods combining semantic search + filters
"""

import struct
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
import sqlite3
import sqlite_vec

from .db import Movie, MovieEnrichment, get_db_paths


class EmbeddingModel:
    """Manages embedding model and caching."""
    
    _instance = None
    _model = None
    _model_name = None
    _dimensions = None
    
    def __new__(cls, model_name: str = "all-MiniLM-L6-v2"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize embedding model (singleton).
        
        Args:
            model_name: HuggingFace model name for embeddings
                       Default: all-MiniLM-L6-v2 (384 dims, fast, good quality)
        """
        if self._model is None or self._model_name != model_name:
            print(f"Loading embedding model: {model_name}...")
            self._model = SentenceTransformer(model_name)
            self._model_name = model_name
            self._dimensions = self._model.get_sentence_embedding_dimension()
            print(f"✓ Model loaded ({self._dimensions} dimensions)")
    
    @property
    def dimensions(self) -> int:
        """Get embedding dimensions."""
        return self._dimensions
    
    def encode(self, text: str) -> np.ndarray:
        """Generate embedding vector for text."""
        return self._model.encode(text, convert_to_numpy=True)
    
    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts."""
        return self._model.encode(texts, convert_to_numpy=True, show_progress_bar=True)


def vector_to_blob(vector: np.ndarray) -> bytes:
    """Convert numpy array to binary blob for SQLite storage.
    
    Args:
        vector: Numpy array of floats
        
    Returns:
        Binary blob suitable for SQLite storage
    """
    return struct.pack(f'{len(vector)}f', *vector.astype(np.float32))


def blob_to_vector(blob: bytes) -> np.ndarray:
    """Convert binary blob back to numpy array.
    
    Args:
        blob: Binary data from SQLite
        
    Returns:
        Numpy array of floats
    """
    num_floats = len(blob) // 4
    return np.array(struct.unpack(f'{num_floats}f', blob), dtype=np.float32)


class MovieEmbeddings:
    """Manages movie embeddings with sqlite-vec for hybrid RAG retrieval."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize embedding manager.
        
        Args:
            model_name: Sentence transformer model to use
        """
        self.embedding_model = EmbeddingModel(model_name)
        self.dimensions = self.embedding_model.dimensions
        self._ensure_vec_table()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get SQLite connection with sqlite-vec loaded."""
        db_path = get_db_paths()["movies_db"].replace("sqlite:///", "")
        conn = sqlite3.connect(db_path)
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        return conn
    
    def _ensure_vec_table(self) -> None:
        """Create virtual table for embeddings if it doesn't exist."""
        with self._get_connection() as conn:
            # Check if table exists
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='movie_embeddings'"
            )
            
            if not cursor.fetchone():
                print(f"Creating movie_embeddings vector table ({self.dimensions} dims)...")
                conn.execute(f"""
                    CREATE VIRTUAL TABLE movie_embeddings USING vec0(
                        embedding float[{self.dimensions}]
                    )
                """)
                conn.commit()
                print("✓ Vector table created")
    
    def generate_and_store(self, movie_id: int, force: bool = False) -> bool:
        """Generate and store embedding for a movie.
        
        Args:
            movie_id: Movie ID to generate embedding for
            force: If True, regenerate even if embedding exists
            
        Returns:
            True if embedding was generated/stored, False if skipped
        """
        # Check if already exists
        if not force and self.has_embedding(movie_id):
            return False
        
        # Get movie data
        movie = Movie.get_by_id(movie_id)
        if not movie or not movie.overview:
            return False
        
        # Generate embedding from title + overview
        text = f"{movie.title}. {movie.overview}"
        embedding = self.embedding_model.encode(text)
        
        # Store in vector table
        blob = vector_to_blob(embedding)
        with self._get_connection() as conn:
            # Delete existing if present
            conn.execute("DELETE FROM movie_embeddings WHERE rowid = ?", (movie_id,))
            # Insert new
            conn.execute(
                "INSERT INTO movie_embeddings(rowid, embedding) VALUES (?, ?)",
                (movie_id, blob)
            )
            conn.commit()
        
        return True
    
    def generate_batch(self, movie_ids: List[int], force: bool = False) -> Dict[str, int]:
        """Generate embeddings for multiple movies at once (faster).
        
        Args:
            movie_ids: List of movie IDs
            force: If True, regenerate even if embeddings exist
            
        Returns:
            Dict with 'generated', 'skipped', 'failed' counts
        """
        stats = {"generated": 0, "skipped": 0, "failed": 0}
        
        # Filter out movies that already have embeddings
        if not force:
            movie_ids = [mid for mid in movie_ids if not self.has_embedding(mid)]
        
        if not movie_ids:
            stats["skipped"] = len(movie_ids)
            return stats
        
        # Get all movies in batch
        movies = []
        texts = []
        valid_ids = []
        
        for movie_id in movie_ids:
            movie = Movie.get_by_id(movie_id)
            if movie and movie.overview:
                movies.append(movie)
                texts.append(f"{movie.title}. {movie.overview}")
                valid_ids.append(movie_id)
            else:
                stats["failed"] += 1
        
        if not texts:
            return stats
        
        # Generate all embeddings at once
        print(f"Generating {len(texts)} embeddings...")
        embeddings = self.embedding_model.encode_batch(texts)
        
        # Store all embeddings
        with self._get_connection() as conn:
            for movie_id, embedding in zip(valid_ids, embeddings):
                blob = vector_to_blob(embedding)
                conn.execute("DELETE FROM movie_embeddings WHERE rowid = ?", (movie_id,))
                conn.execute(
                    "INSERT INTO movie_embeddings(rowid, embedding) VALUES (?, ?)",
                    (movie_id, blob)
                )
                stats["generated"] += 1
            conn.commit()
        
        print(f"✓ Stored {stats['generated']} embeddings")
        return stats
    
    def has_embedding(self, movie_id: int) -> bool:
        """Check if movie has an embedding."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT rowid FROM movie_embeddings WHERE rowid = ?",
                (movie_id,)
            )
            return cursor.fetchone() is not None
    
    def get_embedding(self, movie_id: int) -> Optional[np.ndarray]:
        """Retrieve embedding vector for a movie.
        
        Args:
            movie_id: Movie ID
            
        Returns:
            Numpy array or None if not found
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT embedding FROM movie_embeddings WHERE rowid = ?",
                (movie_id,)
            )
            row = cursor.fetchone()
            
            if row:
                return blob_to_vector(row[0])
            return None
    
    def search_by_text(
        self,
        query: str,
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[int, float, Movie]]:
        """Semantic search by text query using embeddings.
        
        Args:
            query: Natural language search query
            k: Number of results to return
            filters: Optional filters (e.g., {"genres__contains": "Action"})
            
        Returns:
            List of (movie_id, distance, Movie) tuples sorted by relevance
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query)
        return self.search_by_vector(query_embedding, k, filters)
    
    def search_by_vector(
        self,
        vector: np.ndarray,
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[int, float, Movie]]:
        """Semantic search by embedding vector.
        
        Args:
            vector: Query embedding vector
            k: Number of results to return
            filters: Optional filters to apply
            
        Returns:
            List of (movie_id, distance, Movie) tuples sorted by relevance
        """
        query_blob = vector_to_blob(vector)
        
        # Build WHERE clause for filters
        where_clauses = []
        params = []  # Don't include query_blob here
        
        if filters:
            for key, value in filters.items():
                if "__contains" in key:
                    field = key.replace("__contains", "")
                    where_clauses.append(f"m.{field} LIKE ?")
                    params.append(f"%{value}%")
                elif "__gte" in key:
                    field = key.replace("__gte", "")
                    where_clauses.append(f"m.{field} >= ?")
                    params.append(value)
                elif "__lte" in key:
                    field = key.replace("__lte", "")
                    where_clauses.append(f"m.{field} <= ?")
                    params.append(value)
                else:
                    where_clauses.append(f"m.{key} = ?")
                    params.append(value)
        
        where_sql = " AND " + " AND ".join(where_clauses) if where_clauses else ""
        
        # Perform vector search with optional filters
        # Note: sqlite-vec requires "k = ?" in WHERE clause, not LIMIT
        query = f"""
            SELECT 
                e.rowid as movieId,
                distance,
                m.title,
                m.overview,
                m.budget,
                m.revenue,
                m.genres
            FROM movie_embeddings e
            JOIN movies m ON m.movieId = e.rowid
            WHERE e.embedding MATCH ? AND k = ?{where_sql}
            ORDER BY distance
        """
        
        # Build final params: [query_blob, k, ...filter_params]
        final_params = [query_blob, k] + params
        
        with self._get_connection() as conn:
            cursor = conn.execute(query, final_params)
            results = []
            
            for row in cursor.fetchall():
                movie_id = row[0]
                distance = row[1]
                movie = Movie.get_by_id(movie_id)
                if movie:
                    results.append((movie_id, distance, movie))
            
            return results
    
    def hybrid_search(
        self,
        query: Optional[str] = None,
        vector: Optional[np.ndarray] = None,
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        enrich_results: bool = True
    ) -> List[Dict[str, Any]]:
        """Hybrid RAG retrieval combining semantic search with structured filters.
        
        This is the main retrieval method supporting:
        - Semantic search via text query or vector
        - Metadata filtering (genre, budget, revenue, etc.)
        - Optional enrichment data inclusion
        
        Args:
            query: Natural language query (mutually exclusive with vector)
            vector: Pre-computed embedding vector (mutually exclusive with query)
            k: Number of results to return
            filters: Metadata filters, e.g.:
                    {
                        "genres__contains": "Action",
                        "budget__gte": 50000000,
                        "revenue__gte": 100000000
                    }
            enrich_results: If True, include LLM enrichment data
            
        Returns:
            List of dicts with movie data, distance, and optional enrichments
            
        Examples:
            # Text search with genre filter
            results = embeddings.hybrid_search(
                query="space adventure with AI",
                k=5,
                filters={"genres__contains": "Sci-Fi"}
            )
            
            # Vector search with budget filter
            vec = embedding_model.encode("action thriller")
            results = embeddings.hybrid_search(
                vector=vec,
                k=10,
                filters={"budget__gte": 50000000}
            )
            
            # Combine multiple filters
            results = embeddings.hybrid_search(
                query="family comedy",
                filters={
                    "budget__gte": 20000000,
                    "revenue__gte": 50000000,
                    "genres__contains": "Comedy"
                }
            )
        """
        if query is None and vector is None:
            raise ValueError("Must provide either 'query' or 'vector'")
        
        if query is not None and vector is not None:
            raise ValueError("Provide either 'query' OR 'vector', not both")
        
        # Perform search
        if query:
            raw_results = self.search_by_text(query, k, filters)
        else:
            raw_results = self.search_by_vector(vector, k, filters)
        
        # Format results
        results = []
        for movie_id, distance, movie in raw_results:
            # sqlite-vec returns cosine distance (0 = identical, 2 = opposite)
            # Convert to similarity score (0-1, where 1 = identical)
            similarity = max(0.0, 1 - (distance / 2))
            
            result = {
                "movie_id": movie_id,
                "title": movie.title,
                "overview": movie.overview,
                "distance": float(distance),
                "similarity": float(similarity),
                "budget": movie.budget,
                "revenue": movie.revenue,
                "genres": movie.genres,
                "release_date": movie.releaseDate,
            }
            
            # Add enrichment data if requested
            if enrich_results:
                enrichment = MovieEnrichment.get_by_id(movie_id)
                if enrichment:
                    result["enrichment"] = {
                        "sentiment": enrichment.sentiment,
                        "budget_tier": enrichment.budget_tier,
                        "revenue_tier": enrichment.revenue_tier,
                        "effectiveness_score": enrichment.effectiveness_score,
                        "target_audience": enrichment.target_audience,
                    }
            
            results.append(result)
        
        return results
    
    def count_embeddings(self) -> int:
        """Count total embeddings in database."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM movie_embeddings")
            return cursor.fetchone()[0]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about embeddings."""
        total_movies = len(Movie.get_all_with_budget())
        total_embeddings = self.count_embeddings()
        
        return {
            "total_movies": total_movies,
            "total_embeddings": total_embeddings,
            "coverage": f"{total_embeddings / total_movies * 100:.1f}%" if total_movies > 0 else "0%",
            "dimensions": self.dimensions,
            "model": "all-MiniLM-L6-v2"
        }


# Convenience functions

def init_embeddings(model_name: str = "all-MiniLM-L6-v2") -> MovieEmbeddings:
    """Initialize embeddings manager (convenience function)."""
    return MovieEmbeddings(model_name)


def embed_all_movies(force: bool = False) -> Dict[str, int]:
    """Generate embeddings for all movies with overview.
    
    Args:
        force: If True, regenerate all embeddings
        
    Returns:
        Statistics dict with counts
    """
    embeddings = init_embeddings()
    
    # Get all movie IDs with data
    movies_df = Movie.get_all_with_budget()
    movie_ids = movies_df['movieId'].tolist()
    
    print(f"Processing {len(movie_ids)} movies...")
    return embeddings.generate_batch(movie_ids, force=force)


if __name__ == "__main__":
    """Test embedding functionality."""
    print("=" * 70)
    print("Testing Vector Embeddings with sqlite-vec")
    print("=" * 70)
    
    # Initialize
    embeddings = init_embeddings()
    
    # Show stats
    print("\n📊 Current Statistics:")
    stats = embeddings.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Test embedding generation
    print("\n🔧 Testing embedding generation...")
    test_movie_id = 550  # Fight Club
    if embeddings.generate_and_store(test_movie_id, force=True):
        print(f"   ✓ Generated embedding for movie {test_movie_id}")
    
    # Test retrieval
    vec = embeddings.get_embedding(test_movie_id)
    if vec is not None:
        print(f"   ✓ Retrieved embedding: {vec.shape}")
    
    # Test semantic search
    print("\n🔍 Testing semantic search...")
    results = embeddings.hybrid_search(
        query="dark psychological thriller",
        k=3
    )
    
    print(f"   Found {len(results)} movies:")
    for r in results:
        print(f"   - {r['title']} (similarity: {r['similarity']:.3f})")
    
    print("\n✅ All tests passed!")

"""Interactive chat agent using pydantic-ai with MCP server integration.

Provides conversational interface to the movie database via MCP tools.
"""

import asyncio
import json
import sys
from typing import Optional
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel


class MovieChatAgent:
    """Interactive chat agent for movie database queries."""
    
    def __init__(self, model_name: str = "openai:gpt-4o-mini"):
        """Initialize chat agent.
        
        Args:
            model_name: LLM model to use
        """
        self.model_name = model_name
        
        # Create agent with system prompt
        self.agent = Agent(
            model_name,
            system_prompt="""You are a helpful movie database assistant. You have access to a comprehensive movie database with:
            
- Movie information (titles, budgets, revenue, genres, overviews)
- User ratings and preferences
- LLM-enriched movie attributes (sentiment, effectiveness scores, target audience)
- Machine learning-based recommendation system

You can:
1. Search for movies by title
2. Get detailed movie information including enrichments
3. View user rating histories and preferences
4. Generate personalized recommendations
5. Compare movies across various dimensions
6. Discover random movies

When users ask about movies, use the available tools to provide accurate, helpful information.
Be conversational and friendly. When showing movie data, format it nicely for readability.
If a user asks about recommendations, you can use the ML-based recommendation system.
            """
        )
        
        # We'll add tools dynamically when MCP server is available
        self._mcp_tools = []
    
    def register_mcp_tool(self, tool_name: str, tool_func: callable, description: str):
        """Register an MCP tool with the agent.
        
        Args:
            tool_name: Name of the tool
            tool_func: Function that implements the tool
            description: Description of what the tool does
        """
        self._mcp_tools.append({
            "name": tool_name,
            "func": tool_func,
            "description": description
        })
        
        # Register with pydantic-ai agent
        self.agent.tool(tool_func)
    
    async def chat(self, message: str) -> str:
        """Process a chat message and return response.
        
        Args:
            message: User's message
            
        Returns:
            Agent's response
        """
        result = await self.agent.run(message)
        return result.output


# Default system prompt
DEFAULT_SYSTEM_PROMPT = """You are a helpful movie database assistant. You have access to a comprehensive movie database with:

- Movie information (titles, budgets, revenue, genres, overviews)
- User ratings and preferences
- LLM-enriched movie attributes (sentiment, effectiveness scores, target audience, budget/revenue tiers)
- Machine learning-based recommendation system

You can:
1. Search for movies by title
2. Get detailed movie information including enrichments
3. View user rating histories and preferences
4. Generate personalized recommendations
5. Compare movies across various dimensions
6. Discover random movies

When users ask about movies, use the available tools to provide accurate, helpful information.
Be conversational and friendly. When showing movie data, format it nicely for readability.
If a user asks about recommendations, you can use the ML-based recommendation system.

For queries like "recommend action movies with high revenue and positive sentiment", you should:
1. Search for action movies
2. Get their enrichment data
3. Filter for high revenue tier and positive sentiment
4. Present the results

For user preference summaries, analyze their rating history and genres they rate highly.
"""


class MovieChatCLI:
    """CLI for interactive and non-interactive chat with movie database."""
    
    def __init__(
        self,
        model_name: str = "openai:gpt-4o-mini",
        system_prompt: Optional[str] = None
    ):
        """Initialize chat CLI.
        
        Args:
            model_name: LLM model to use
            system_prompt: Optional custom system prompt (uses default if not provided)
        """
        self.model_name = model_name
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.agent = None
        self.mcp_process = None
        
    async def _start_mcp_server(self):
        """Start the MCP server as a subprocess."""
        import subprocess
        
        # Start MCP server on stdio
        self.mcp_process = subprocess.Popen(
            ["uv", "run", "python", "-m", "src.mcp_server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        print("🚀 MCP server started")
        
    def _stop_mcp_server(self):
        """Stop the MCP server."""
        if self.mcp_process:
            self.mcp_process.terminate()
            self.mcp_process.wait()
            print("🛑 MCP server stopped")
    
    async def _call_mcp_tool(self, tool_name: str, **kwargs):
        """Call an MCP tool via the subprocess.
        
        Args:
            tool_name: Name of the MCP tool
            **kwargs: Tool arguments
            
        Returns:
            Tool result
        """
        # For now, we'll import and call directly since MCP stdio integration
        # requires more complex setup. In production, this would go through MCP protocol.
        from src.mcp_server import (
            search_movies, get_movie_details, get_user_ratings,
            get_movie_recommendations, compare_movies, get_random_movies
        )
        
        tool_map = {
            "search_movies": search_movies,
            "get_movie_details": get_movie_details,
            "get_user_ratings": get_user_ratings,
            "get_movie_recommendations": get_movie_recommendations,
            "compare_movies": compare_movies,
            "get_random_movies": get_random_movies
        }
        
        if tool_name not in tool_map:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})
        
        return tool_map[tool_name](**kwargs)
    
    async def setup_agent(self):
        """Set up the chat agent with MCP tools."""
        print("🤖 Initializing movie chat agent...")
        
        # Import MCP functions directly
        from src.mcp_server import (
            search_movies, get_movie_details, get_user_ratings,
            get_movie_recommendations, compare_movies, get_random_movies
        )
        
        # Create agent with tools registered directly
        self.agent = Agent(
            self.model_name,
            system_prompt=self.system_prompt,
            retries=2
        )
        
        # Register tools using the tool decorator with proper context
        from pydantic_ai import RunContext
        
        @self.agent.tool_plain
        def search_movies_tool(query: str, limit: int = 10) -> str:
            """Search for movies by title.
            
            Args:
                query: Search query for movie title
                limit: Maximum number of results
                
            Returns:
                JSON string with matching movies
            """
            return search_movies(query, limit)
        
        @self.agent.tool_plain
        def get_movie_details_tool(movie_id: int) -> str:
            """Get detailed information about a movie including LLM enrichments.
            
            Args:
                movie_id: The movie ID
                
            Returns:
                JSON string with movie details
            """
            return get_movie_details(movie_id)
        
        @self.agent.tool_plain
        def get_user_ratings_tool(user_id: int, limit: Optional[int] = None) -> str:
            """Get a user's rating history and preferences.
            
            Args:
                user_id: The user ID
                limit: Optional limit on number of ratings
                
            Returns:
                JSON string with user ratings
            """
            return get_user_ratings(user_id, limit)
        
        @self.agent.tool
        async def get_recommendations_tool(ctx: RunContext[None], user_id: int, n: int = 10) -> str:
            """Get movie recommendations for a user using ML model.
            
            Args:
                ctx: Run context
                user_id: User ID to recommend for
                n: Number of recommendations
                
            Returns:
                JSON string with recommendations
            """
            return await get_movie_recommendations(user_id, n)
        
        @self.agent.tool_plain
        def compare_movies_tool(movie_ids: list[int]) -> str:
            """Compare multiple movies across budget, revenue, ratings, and enrichments.
            
            Args:
                movie_ids: List of movie IDs to compare
                
            Returns:
                JSON string with comparison
            """
            return compare_movies(movie_ids)
        
        @self.agent.tool_plain
        def get_random_movies_tool(n: int = 10, with_enrichment: bool = False) -> str:
            """Get random movies from the database.
            
            Args:
                n: Number of movies to return
                with_enrichment: Only return enriched movies
                
            Returns:
                JSON string with random movies
            """
            return get_random_movies(n, with_enrichment)
        
        print("✅ Agent ready! Tools registered:")
        print("   - search_movies_tool")
        print("   - get_movie_details_tool")
        print("   - get_user_ratings_tool")
        print("   - get_recommendations_tool")
        print("   - compare_movies_tool")
        print("   - get_random_movies_tool")
        print()
    
    async def run_interactive(self):
        """Run interactive chat session."""
        await self.setup_agent()
        
        print("=" * 70)
        print("🎬 Movie Database Chat Assistant")
        print("=" * 70)
        print()
        print("Ask me anything about movies! Examples:")
        print("  • 'Find movies about space'")
        print("  • 'Tell me about The Matrix'")
        print("  • 'What are user 5's favorite movies?'")
        print("  • 'Recommend movies for user 5'")
        print("  • 'Compare Fight Club and The Matrix'")
        print("  • 'Show me some random movies'")
        print("  • 'Recommend action movies with high revenue and positive sentiment'")
        print()
        print("Type 'exit' or 'quit' to end the conversation.")
        print("=" * 70)
        print()
        
        conversation_history = []
        
        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    print("\n👋 Goodbye!")
                    break
                
                # Add to conversation history
                conversation_history.append({"role": "user", "content": user_input})
                
                # Get agent response
                print("🤖 Assistant: ", end="", flush=True)
                
                result = await self.agent.run(user_input)
                response = result.output
                print(response)
                print()
                
                # Add to conversation history
                conversation_history.append({"role": "assistant", "content": response})
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                import traceback
                traceback.print_exc()
                print()
    
    async def run_query(self, query: str) -> str:
        """Run a single query and return the response.
        
        Args:
            query: The query to ask
            
        Returns:
            The agent's response
        """
        await self.setup_agent()
        result = await self.agent.run(query)
        return result.output
    
    def run(self, query: Optional[str] = None):
        """Run the chat CLI in either interactive or non-interactive mode.
        
        Args:
            query: Optional query for non-interactive mode. If provided, 
                   runs once and exits. If not provided, enters interactive mode.
        """
        try:
            if query:
                # Non-interactive mode: run single query and exit
                response = asyncio.run(self.run_query(query))
                print(response)
            else:
                # Interactive mode: continuous conversation
                asyncio.run(self.run_interactive())
        except Exception as e:
            print(f"❌ Failed to start chat: {e}")
        finally:
            self._stop_mcp_server()


async def main():
    """Main entry point for chat agent."""
    import sys
    
    model_name = "openai:gpt-4o-mini"
    
    # Check for model arg
    if len(sys.argv) > 1 and sys.argv[1].startswith("--model="):
        model_name = sys.argv[1].split("=", 1)[1]
    
    cli = MovieChatCLI(model_name=model_name)
    cli.run()


if __name__ == "__main__":
    asyncio.run(main())

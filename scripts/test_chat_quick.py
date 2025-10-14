"""Quick test of chat agent functionality."""

import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from src.chat_agent import MovieChatCLI

async def test_chat():
    """Test that chat agent can respond to a simple query."""
    print("🧪 Testing chat agent...\n")
    
    cli = MovieChatCLI(model_name="openai:gpt-4o-mini")
    await cli.setup_agent()
    
    # Test a simple query
    test_queries = [
        "Find movies about robots",
        "Tell me about movie ID 218",
        "What are user 5's top rated movies?"
    ]
    
    for query in test_queries:
        print(f"Query: {query}")
        result = await cli.agent.run(query)
        print(f"Response: {result.output[:200]}...\n")
        print("✅ Test passed!\n")

if __name__ == "__main__":
    asyncio.run(test_chat())

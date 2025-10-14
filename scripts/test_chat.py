#!/usr/bin/env python
"""Quick test of the chat interface."""

import asyncio
from src.chat_agent import MovieChatCLI

async def test_chat():
    """Test chat with a few queries."""
    from src.chat_agent import MovieChatAgent
    
    print("Testing MovieChatAgent...")
    
    # Create agent
    cli = MovieChatCLI()
    await cli.setup_agent()
    
    # Test queries
    queries = [
        "Find movies about robots",
        "Tell me about movie ID 550"
    ]
    
    for query in queries:
        print(f"\n🧪 Testing: {query}")
        print("=" * 60)
        response = await cli.agent.chat(query)
        print(f"Response: {response}\n")

if __name__ == "__main__":
    asyncio.run(test_chat())

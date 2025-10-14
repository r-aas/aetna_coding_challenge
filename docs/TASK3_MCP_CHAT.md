# Task 3: MCP Server & Chat Interface ✅

## Overview

Added two powerful new features to the movie database system:

1. **MCP Server** - Exposes movie database as MCP tools for AI agents
2. **Chat Interface** - Interactive conversational AI assistant for the movie database

## Architecture

### MCP Server (`src/mcp_server.py`)

A FastMCP server that exposes 6 tools:

1. **search_movies** - Search for movies by title
2. **get_movie_details** - Get detailed movie information including enrichments
3. **get_user_ratings** - Get user rating history and preferences
4. **get_movie_recommendations** - Get ML-based recommendations
5. **compare_movies** - Compare multiple movies across dimensions
6. **get_random_movies** - Discover random movies

The server runs on stdio and can be:
- Launched via CLI for testing
- Integrated into Claude Desktop or other MCP clients  
- Used by the interactive chat agent

### Chat Agent (`src/chat_agent.py`)

A pydantic-ai powered conversational agent that:
- Uses the MCP tools to interact with the database
- Provides natural language interface to all database operations
- Maintains conversation context
- Formats responses for readability

## Usage

### 1. Launch MCP Server

```bash
# Run MCP server on stdio
uv run python main.py mcp-server
```

### 2. Chat Interface (Non-Interactive by Default)

The chat interface now supports both non-interactive (single query) and interactive (continuous conversation) modes.

```bash
# Non-interactive mode (default): Single query and exit
uv run python main.py chat "Find movies about space"
uv run python main.py chat "What are user 5's top rated movies?"
uv run python main.py chat "Recommend action movies for user 10"

# Interactive mode: Continuous conversation
uv run python main.py chat --interactive
uv run python main.py chat -i

# With custom model
uv run python main.py chat "Find sci-fi movies" --model "anthropic:claude-3-5-sonnet-20241022"

# With custom system prompt
uv run python main.py chat "Recommend movies" --system-prompt "You are a movie critic who only likes indie films."
```

### Example Chat Session

```
======================================================================
🎬 Movie Database Chat Assistant
======================================================================

Ask me anything about movies! Examples:
  • 'Find movies about space'
  • 'Tell me about The Matrix'
  • 'What are user 5's favorite movies?'
  • 'Recommend movies for user 5'
  • 'Compare Fight Club and The Matrix'
  • 'Show me some random movies'

Type 'exit' or 'quit' to end the conversation.
======================================================================

You: Find movies about robots

🤖 Assistant: I found several movies about robots in the database:

1. **The Terminator** (ID: 218)
   - Genres: Action, Thriller, Science Fiction
   - Budget: $6,400,000
   - Revenue: $78,371,200

2. **Terminator 2: Judgment Day** (ID: 280)
   - Genres: Action, Thriller, Science Fiction
   - Budget: $102,000,000
   - Revenue: $519,843,345

3. **I, Robot** (ID: 2048)
   - Genres: Action, Science Fiction
   - Budget: $120,000,000
   - Revenue: $347,234,916

Would you like more details about any of these movies?

You: Tell me about The Terminator

🤖 Assistant: **The Terminator** is a 1984 sci-fi action film:

- **Overview**: A cyborg assassin is sent back in time to 1984...
- **Budget**: $6.4M
- **Revenue**: $78.4M (1,125% ROI!)
- **Runtime**: 107 minutes
- **Ratings**: 3,174 ratings, average 3.87/5.0

**LLM Enrichment**:
- Sentiment: Positive
- Budget Tier: Low
- Revenue Tier: High
- Effectiveness Score: 9.2/10
- Target Audience: Adult

The film was incredibly effective given its modest budget!

You: Recommend movies for user 5

🤖 Assistant: Based on user 5's rating history, here are my top recommendations:

1. **Bowling for Columbine** (Score: 0.3252)
   - Documentary about gun violence
   - Matches user's interest in thought-provoking content

2. **Se7en** (Score: 0.3194)
   - Dark psychological thriller
   - Aligns with user's preference for intense dramas

3. **Vertigo** (Score: 0.2914)
   - Classic Hitchcock psychological thriller
   - Appeals to user's appreciation for masterful direction

[Generated in 0.4ms using ML model]

You: exit

👋 Goodbye!
```

## Integration with Claude Desktop

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "movie-db": {
      "command": "uv",
      "args": ["run", "python", "/path/to/aetna-coding-challenge/main.py", "mcp-server"]
    }
  }
}
```

Then Claude Desktop will have access to all movie database tools!

## Features

### Natural Language Queries

The chat agent understands conversational requests:
- "What movies did user 5 like?"
- "Show me some sci-fi movies"
- "Compare these three movies"
- "Give me recommendations"

### Tool Integration

Behind the scenes, the agent:
1. Analyzes your natural language request
2. Determines which tools to use
3. Calls MCP tools to get data
4. Formats responses conversationally

### Context Awareness

The agent maintains conversation history and can:
- Reference previous messages
- Build on earlier queries
- Provide follow-up suggestions

## Technical Details

### Pydantic-AI Integration

- Uses `pydantic-ai` for structured agent framework
- Tools are registered via decorators
- Type-safe tool parameters
- Automatic schema generation

### MCP Protocol

- Built with `fastmcp` library
- Runs on stdio for process isolation
- JSON-based tool definitions
- Compatible with Claude Desktop and other MCP clients

## CLI Commands

### Launch MCP Server

```bash
python main.py mcp-server
```

**Purpose**: Start the MCP server for external clients

**Use Cases**:
- Testing MCP integration
- Claude Desktop integration
- Other MCP client integration

### Start Interactive Chat

```bash
# Default model
python main.py chat

# Custom model
python main.py chat --model "anthropic:claude-3-5-sonnet-20241022"
```

**Purpose**: Interactive conversational interface to movie database

**Use Cases**:
- Natural language queries
- Exploratory data analysis
- User-friendly database access
- Testing agent capabilities

## Benefits

### 1. Accessibility
Non-technical users can query the database using natural language instead of SQL or CLI commands.

### 2. Flexibility
MCP server can be integrated into any MCP-compatible application (Claude Desktop, custom tools, etc.)

### 3. Intelligence
The LLM agent provides contextual, intelligent responses rather than raw data dumps.

### 4. Extensibility
Easy to add new MCP tools - just add a function with the `@mcp.tool()` decorator!

## Status

✅ **MCP Server**: Fully functional with 6 tools
✅ **Chat Interface**: Working with pydantic-ai  
✅ **CLI Commands**: Integrated into main CLI
✅ **Documentation**: Complete

## Future Enhancements

Potential improvements:
1. **Streaming responses** for long-running queries
2. **Chat history persistence** across sessions
3. **Multi-turn tool calls** for complex queries
4. **Custom MCP server config** via CLI args
5. **Web interface** using the MCP server

---

**Total Development Time**: ~45 minutes
**Lines of Code**: 
- `src/mcp_server.py`: ~300 lines
- `src/chat_agent.py`: ~320 lines

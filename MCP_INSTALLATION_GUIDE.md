# 🤖 Movie Recommender MCP Installation Guide

**Complete setup guide for integrating the Movie Recommendation System with Claude Desktop and Claude Code via Model Context Protocol (MCP).**

---

## 📋 Prerequisites

- Python 3.11+ installed
- [uv package manager](https://docs.astral.sh/uv/) (will be installed automatically)
- OpenAI API key (for LLM features)
- Claude Desktop or Claude Code CLI

---

## 🚀 Step 1: Install the Movie Recommendation System

### 1.1 Clone and Setup

```bash
# Clone the repository
git clone https://github.com/r-aas/aetna_coding_challenge.git
cd aetna-coding-challenge

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc  # or restart terminal

# Install dependencies
uv sync

# Verify installation
uv run pytest tests/ -v
```

### 1.2 Configure Environment

```bash
# Set your OpenAI API key
export OPENAI_API_KEY="your-openai-api-key-here"

# Or create a .env file
echo "OPENAI_API_KEY=your-openai-api-key-here" > .env
```

### 1.3 Test the System

```bash
# Run a quick test
uv run python main.py sample --n 5

# Test MCP server directly
uv run python -m src.mcp_server
# Should start without errors
```

---

## 🖥️ Step 2: Claude Desktop Integration

### 2.1 Locate Configuration File

**macOS:**
```bash
open ~/Library/Application\ Support/Claude/
# Look for claude_desktop_config.json
```

**Windows:**
```bash
# Navigate to: %APPDATA%\Claude\
# Look for claude_desktop_config.json
```

**Linux:**
```bash
# Usually: ~/.config/claude/claude_desktop_config.json
```

### 2.2 Update Configuration

Edit `claude_desktop_config.json` and add the movie-recommender server:

```json
{
  "mcpServers": {
    "movie-recommender": {
      "command": "/usr/local/bin/uv",
      "args": [
        "--directory",
        "/FULL/PATH/TO/aetna-coding-challenge",
        "run",
        "python",
        "-m",
        "src.mcp_server"
      ],
      "env": {
        "OPENAI_API_KEY": "your-openai-api-key-here"
      }
    }
  }
}
```

**⚠️ Important Notes:**
- Replace `/FULL/PATH/TO/aetna-coding-challenge` with the absolute path to your project
- Replace `/usr/local/bin/uv` with your uv installation path (run `which uv` to find it)
- If you already have other MCP servers, add movie-recommender as an additional entry

### 2.3 Find Your uv Path

```bash
# Find uv installation path
which uv

# Common locations:
# macOS/Linux: /usr/local/bin/uv or ~/.local/bin/uv
# Windows: C:\Users\YourName\.local\bin\uv.exe
```

### 2.4 Get Absolute Project Path

```bash
# In your project directory
pwd
# Copy this full path for the configuration
```

### 2.5 Complete Configuration Example

```json
{
  "mcpServers": {
    "movie-recommender": {
      "command": "/Users/yourname/.local/bin/uv",
      "args": [
        "--directory",
        "/Users/yourname/projects/aetna-coding-challenge",
        "run",
        "python",
        "-m",
        "src.mcp_server"
      ],
      "env": {
        "OPENAI_API_KEY": "sk-proj-abc123...",
        "PATH": "/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
      }
    }
  }
}
```

### 2.6 Restart Claude Desktop

1. Completely quit Claude Desktop
2. Restart the application
3. Start a new conversation

---

## 💻 Step 3: Claude Code CLI Integration

### 3.1 Install Claude Code

```bash
# If not already installed
npm install -g @anthropics/claude-code
# or follow official installation instructions
```

### 3.2 Add MCP Server

```bash
# Navigate to your project directory
cd /path/to/aetna-coding-challenge

# Add the MCP server
claude mcp add --transport stdio movie-recommender -- uv --directory $(pwd) run python -m src.mcp_server
```

### 3.3 Verify Connection

```bash
# List MCP servers
claude mcp list

# Should show:
# movie-recommender: uv --directory /path/to/aetna-coding-challenge run python -m src.mcp_server - ✓ Connected
```

### 3.4 Remove Server (if needed)

```bash
# To remove and reconfigure
claude mcp remove movie-recommender -s local
```

---

## 🧪 Step 4: Verify Installation

### 4.1 Test in Claude Desktop

Start a conversation and try:

```
"Can you search for movies with 'matrix' in the title and show me details about The Matrix?"
```

You should see Claude automatically use the movie search tools.

### 4.2 Test in Claude Code

```bash
# Start Claude Code in your project
claude code

# In the conversation, try:
# "Search for action movies and recommend some to user 5"
```

### 4.3 Independent Testing

```bash
# Install MCP testing client
uv tool install mcp-client-for-testing

# Test movie search
mcp-client-for-testing \
  --config '[{
    "name": "movie-recommender",
    "command": "uv",
    "args": ["--directory", ".", "run", "python", "-m", "src.mcp_server"],
    "env": {"OPENAI_API_KEY": "your-key"}
  }]' \
  --tool_call '{"name": "search_movies", "arguments": {"query": "matrix", "limit": 3}}'

# Test movie details
mcp-client-for-testing \
  --config '[{
    "name": "movie-recommender", 
    "command": "uv",
    "args": ["--directory", ".", "run", "python", "-m", "src.mcp_server"]
  }]' \
  --tool_call '{"name": "get_movie_details", "arguments": {"movie_id": 603}}'
```

---

## 🛠️ Available Tools

Once connected, Claude can use these 11 movie tools:

| Tool | Description | Example |
|------|-------------|---------|
| `search_movies` | Find movies by title keyword | Search for "star wars" |
| `get_movie_details` | Get full movie info + enrichments | Details for movie ID 603 |
| `get_user_ratings` | User's rating history and preferences | User 5's movie preferences |
| `get_random_movies` | Random movie samples | Show me 5 random movies |
| `get_movie_recommendations` | ML-based fast recommendations | Quick recs for user 5 |
| `get_movie_recommendations` | LLM-based smart recommendations | Intelligent recs with reasoning |
| `semantic_search_movies` | Natural language movie search | "Uplifting family comedies" |
| `compare_movies` | Side-by-side movie comparison | Compare Matrix vs Matrix Reloaded |
| `predict_rating` | Predict user rating for movie | What would user 5 rate Inception? |
| `get_movies_by_criteria` | Filter by budget, genre, year, etc. | High-budget 90s sci-fi |
| `get_similar_movies` | Content-based similarity search | Movies similar to The Matrix |

---

## 🔧 Troubleshooting

### Common Issues

#### 1. "MCP server not connecting"

**Check uv installation:**
```bash
which uv
uv --version
```

**Verify project setup:**
```bash
cd /path/to/aetna-coding-challenge
uv sync
uv run python -c "import src.mcp_server; print('OK')"
```

**Test MCP server directly:**
```bash
uv run python -m src.mcp_server
# Should start without errors
```

#### 2. "Tools not appearing in Claude"

**Restart Claude Desktop completely:**
- Quit application entirely
- Restart and start new conversation

**Check configuration syntax:**
```bash
# Validate JSON syntax
python -m json.tool ~/Library/Application\ Support/Claude/claude_desktop_config.json
```

**Verify paths are absolute:**
```bash
# Wrong: "./aetna-coding-challenge"
# Right: "/Users/yourname/projects/aetna-coding-challenge"
```

#### 3. "API key errors"

**Set environment variable:**
```bash
export OPENAI_API_KEY="your-key"
# Add to ~/.bashrc or ~/.zshrc for persistence
```

**Or add to MCP config:**
```json
"env": {
  "OPENAI_API_KEY": "your-key-here"
}
```

**Test without LLM features:**
Most tools work without API key (except LLM recommendations)

#### 4. "Permission denied" errors

**Make uv executable:**
```bash
chmod +x $(which uv)
```

**Check directory permissions:**
```bash
ls -la /path/to/aetna-coding-challenge
# Should be readable by your user
```

### Debug Mode

**Run MCP server with logging:**
```bash
cd /path/to/aetna-coding-challenge
PYTHONPATH=. uv run python -m src.mcp_server --debug
```

**Test specific functions:**
```bash
uv run pytest tests/test_mcp_server.py -v
```

**Check MCP client logs:**
```bash
# In Claude Code
claude mcp get movie-recommender
```

---

## 🎯 Example Usage Scenarios

### 1. Movie Discovery

```
You: "I'm looking for some good sci-fi movies. Can you help?"

Claude: [Uses search_movies and get_movie_details]
I found some excellent sci-fi movies in the database:

1. **The Matrix (1999)**
   - Budget: $63M, Revenue: $463M
   - Sentiment: Positive
   - Effectiveness Score: 8/10
   - A groundbreaking film about reality and consciousness

2. **Blade Runner 2049 (2017)**
   - High-budget production with stunning visuals
   - Target Audience: Adult
   - Sequel that honors the original while expanding the story
```

### 2. Personalized Recommendations

```
You: "Can you recommend movies for user 5 based on their preferences?"

Claude: [Uses get_user_ratings and get_movie_recommendations]
Looking at user 5's rating history:
- Average rating: 3.91/5 across 100 movies
- Enjoys action/drama combinations
- Prefers character-driven stories

Here are my personalized recommendations with reasoning:
[Detailed recommendations with explanations]
```

### 3. Movie Analysis

```
You: "Compare The Matrix and The Matrix Reloaded"

Claude: [Uses compare_movies tool]
Here's a detailed comparison:

**The Matrix (1999) vs The Matrix Reloaded (2003)**
- Budget: $63M vs $150M (137% increase)
- Revenue: $463M vs $739M (59% increase)
- Effectiveness: Both highly effective but original had better ROI
- Sentiment: Both positive, but original more groundbreaking
```

---

## 📞 Support

If you encounter issues:

1. **Check the logs:** `uv run python -m src.mcp_server`
2. **Test independently:** Use `mcp-client-for-testing`
3. **Verify installation:** `uv run pytest tests/ -v`
4. **Contact:** russ771@gmail.com

---

## 🎬 Ready to Use!

Your movie recommendation system is now fully integrated with Claude! You can have natural conversations about movies while Claude has real-time access to:

- 🎭 27,000+ movies with rich metadata
- 🤖 LLM-powered enrichments and analysis  
- 📊 User preferences and rating history
- 🔍 Semantic search and recommendations
- 📈 ML and LLM recommendation engines

**Enjoy exploring movies with AI assistance! 🍿**
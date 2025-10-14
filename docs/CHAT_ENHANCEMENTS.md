# Chat Interface Enhancements

## Summary

Enhanced the chat interface with non-interactive mode (default) and custom system prompts.

## New Features

### 1. Non-Interactive Mode (Default)

**What it does**: Execute single queries without entering interactive loop

**Why it's useful**:
- Perfect for scripting and automation
- Fast execution (no interactive overhead)
- Pipe-friendly output
- Easy to integrate into workflows

**Usage**:
```bash
# Simple query
uv run python main.py chat "Find movies about robots"

# With model selection
uv run python main.py chat "Recommend movies for user 5" --model "anthropic:claude-3-5-sonnet-20241022"

# Pipe output to file
uv run python main.py chat "List sci-fi movies" > scifi_movies.txt
```

### 2. Custom System Prompts

**What it does**: Override the default system prompt to customize agent behavior

**Why it's useful**:
- Create specialized agents (e.g., family-friendly filter, genre expert)
- Control tone and style (e.g., formal, casual, humorous)
- Add constraints (e.g., only recommend certain types of movies)
- Test different prompting strategies

**Usage**:
```bash
# Movie critic persona
uv run python main.py chat "Recommend a movie" --system-prompt "You are a harsh movie critic who only respects art house films."

# Family-friendly filter
uv run python main.py chat "Find movies for kids" --system-prompt "You only recommend family-friendly movies suitable for children under 10."

# Genre specialist
uv run python main.py chat "Suggest a thriller" --system-prompt "You are a thriller movie expert who analyzes films through a psychological lens."

# Sarcastic assistant
uv run python main.py chat "What's good?" --system-prompt "You are a sarcastic movie critic who makes witty observations."
```

### 3. Interactive Mode (Optional)

**What it does**: Enter continuous conversation mode

**When to use**:
- Exploratory analysis
- Multi-turn conversations
- Back-and-forth dialogue
- When you need context from previous exchanges

**Usage**:
```bash
# Enter interactive mode
uv run python main.py chat --interactive
uv run python main.py chat -i

# Interactive mode with custom model
uv run python main.py chat -i --model "anthropic:claude-3-5-sonnet-20241022"

# Interactive mode with custom system prompt
uv run python main.py chat -i --system-prompt "You are a film noir expert."
```

## Command Reference

```bash
# Non-interactive (default)
python main.py chat <QUERY> [OPTIONS]

# Options:
#   --model TEXT               LLM model to use (default: openai:gpt-4o-mini)
#   --system-prompt TEXT       Custom system prompt
#   --interactive, -i          Enter interactive mode
#   --help                     Show help message
```

## Examples

### Scripting Example
```bash
#!/bin/bash
# Get recommendations for multiple users
for user_id in 1 2 3 4 5; do
  echo "User $user_id recommendations:"
  uv run python main.py chat "Recommend 5 movies for user $user_id"
  echo "---"
done
```

### Pipeline Example
```bash
# Get high-revenue action movies and save to file
uv run python main.py chat "Find action movies with high revenue tier" > action_movies.txt
```

### Custom Agent Example
```bash
# Create a family movie recommender
uv run python main.py chat \
  "Recommend movies" \
  --system-prompt "You are a family movie expert. Only recommend G and PG rated movies suitable for children. Focus on positive messages and educational value." \
  --model "anthropic:claude-3-5-sonnet-20241022"
```

## Benefits

1. **Flexibility**: Choose between quick queries and deep conversations
2. **Automation**: Easy to script and integrate into workflows
3. **Customization**: Tailor agent behavior for specific use cases
4. **Performance**: Non-interactive mode is faster for single queries
5. **Compatibility**: Pipe-friendly output works with Unix tools

## Technical Implementation

- **Non-interactive mode**: Runs a single query through `run_query()` async method
- **Interactive mode**: Uses `run_interactive()` for continuous conversation loop
- **System prompt**: Passed to `Agent()` constructor from pydantic-ai
- **Default behavior**: Query argument provided → non-interactive, no query → interactive

---

**Updated**: October 14, 2025
**Status**: ✅ Production Ready

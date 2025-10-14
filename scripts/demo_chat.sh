#!/bin/bash
# Demo script for new chat features

echo "=============================================="
echo "Chat Interface Enhancement Demo"
echo "=============================================="
echo ""

echo "1. Non-Interactive Mode (Default - Single Query)"
echo "----------------------------------------------"
echo "$ uv run python main.py chat 'Find movies about robots'"
echo ""
uv run python main.py chat "Find movies about robots"
echo ""
echo "Press Enter to continue..."
read

echo ""
echo "2. Custom System Prompt Example"
echo "----------------------------------------------"
echo "$ uv run python main.py chat 'Recommend a movie' --system-prompt 'You are a sarcastic movie critic.'"
echo ""
uv run python main.py chat "Recommend a movie" --system-prompt "You are a sarcastic movie critic who makes witty observations."
echo ""
echo "Press Enter to continue..."
read

echo ""
echo "3. Different Model Example"
echo "----------------------------------------------"
echo "$ uv run python main.py chat 'What are user 5s favorite genres?' --model openai:gpt-4o-mini"
echo ""
uv run python main.py chat "What are user 5's favorite genres?" --model "openai:gpt-4o-mini"
echo ""
echo "Press Enter to continue..."
read

echo ""
echo "4. Scripting Example - Multiple Users"
echo "----------------------------------------------"
for user_id in 1 2 3; do
  echo "Getting recommendations for user $user_id..."
  uv run python main.py chat "Give me 3 movie recommendations for user $user_id" | head -20
  echo ""
done
echo "Press Enter to continue..."
read

echo ""
echo "5. Interactive Mode Demo"
echo "----------------------------------------------"
echo "Now launching interactive mode..."
echo "Try these commands:"
echo "  - Find movies about space"
echo "  - Tell me about The Matrix"
echo "  - Type 'exit' to quit"
echo ""
echo "$ uv run python main.py chat --interactive"
echo ""
uv run python main.py chat --interactive

echo ""
echo "=============================================="
echo "Demo Complete!"
echo "=============================================="

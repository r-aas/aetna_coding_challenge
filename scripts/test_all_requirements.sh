#!/bin/bash
# Test all README requirements non-interactively

set -e  # Exit on error

echo "======================================================================"
echo "🧪 Testing All README Requirements"
echo "======================================================================"
echo ""

cd /Users/r/code/aetna-coding-challenge

# 1. Movie Recommendations (ML + LLM)
echo "1️⃣  Testing Movie Recommendations..."
echo "----------------------------------------------------------------------"
echo "📊 ML-based recommendations (fast):"
uv run python main.py recommend 5 --n 5
echo ""
echo "🤖 LLM-enhanced recommendations with reasoning:"
uv run python main.py chat "recommend 5 movies for user 5 based on their taste"
echo ""

# 2. Rating Predictions
echo "2️⃣  Testing Rating Predictions..."
echo "----------------------------------------------------------------------"
uv run python main.py chat "predict what rating user 5 would give to movie 260 (Star Wars Episode IV) and explain why"
echo ""

# 3. Natural Language Querying (Semantic Search)
echo "3️⃣  Testing Semantic Search..."
echo "----------------------------------------------------------------------"
uv run python main.py semantic-search "dark psychological thriller" -n 5
echo ""

# 4. User Preference Summaries
echo "4️⃣  Testing User Preference Summary..."
echo "----------------------------------------------------------------------"
uv run python main.py chat "summarize user 5's movie preferences based on their rating history"
echo ""

# 5. Comparative Analysis
echo "5️⃣  Testing Movie Comparison..."
echo "----------------------------------------------------------------------"
uv run python main.py chat "compare movies 1 and 260 and tell me which is better"
echo ""

# 6. Data Enrichment
echo "6️⃣  Testing Data Enrichment Status..."
echo "----------------------------------------------------------------------"
uv run python main.py enrich-stats
echo ""

echo "======================================================================"
echo "✅ All README Requirements Tested!"
echo "======================================================================"

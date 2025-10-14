#!/bin/bash
set -e

echo "🧪 Running Aetna Movie System Tests"
echo "===================================="
echo ""

# Activate virtual environment if using uv
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Run tests with coverage
echo "📊 Running tests with coverage..."
uv run pytest tests/ \
    --cov=src \
    --cov-report=term-missing \
    --cov-report=html:htmlcov \
    --cov-report=json:coverage.json \
    -v

echo ""
echo "✅ Tests complete!"
echo ""
echo "📈 Coverage Reports:"
echo "  - Terminal: (shown above)"
echo "  - HTML: open htmlcov/index.html"
echo "  - JSON: coverage.json"
echo ""

# Parse coverage percentage from json
if [ -f coverage.json ]; then
    coverage_pct=$(python3 -c "import json; data=json.load(open('coverage.json')); print(f\"{data['totals']['percent_covered']:.1f}%\")")
    echo "📊 Overall Coverage: $coverage_pct"
fi

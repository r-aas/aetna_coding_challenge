# Movie Recommendation System - Makefile
# Production-grade development automation

.PHONY: help install test test-watch test-coverage clean lint format check docker-build docker-run docker-compose-up docker-compose-down backup restore train eval demo setup-env use-openai use-ollama provider-status test-provider

# Default target
help: ## Show this help message
	@echo "Movie Recommendation System - Development Commands"
	@echo "=================================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Development Setup
install: ## Install dependencies with uv
	@echo "🚀 Installing dependencies..."
	curl -LsSf https://astral.sh/uv/install.sh | sh || echo "uv already installed"
	uv sync
	@echo "✅ Dependencies installed!"

install-dev: install ## Install with development dependencies
	@echo "🔧 Installing development dependencies..."
	uv sync --dev
	@echo "✅ Development environment ready!"

# Testing
test: ## Run all tests
	@echo "🧪 Running test suite..."
	uv run pytest tests/ -v

test-api: ## Run tests with API key (requires OPENAI_API_KEY)
	@echo "🧪 Running tests with API integration..."
	@if [ -z "$$OPENAI_API_KEY" ]; then \
		echo "⚠️  OPENAI_API_KEY not set - loading from .env"; \
		set -a && source .env && set +a; \
	fi
	OPENAI_API_KEY=$$OPENAI_API_KEY uv run pytest tests/ -v

test-coverage: ## Run tests with coverage report
	@echo "📊 Running tests with coverage..."
	uv run pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing
	@echo "📈 Coverage report generated in htmlcov/"

test-watch: ## Run tests in watch mode
	@echo "👀 Running tests in watch mode..."
	uv run pytest-watch tests/ -- -v

# Database Operations
db-test: ## Test database connectivity
	@echo "🗄️  Testing database connectivity..."
	uv run python main.py test-db

db-backup: ## Backup databases with timestamp
	@echo "💾 Creating database backup..."
	@mkdir -p backups
	@timestamp=$$(date +"%Y%m%d_%H%M%S"); \
	cp db/movies.db "backups/movies_$$timestamp.db"; \
	cp db/ratings.db "backups/ratings_$$timestamp.db"; \
	echo "✅ Backup created: backups/*_$$timestamp.db"

db-restore: ## Restore databases from latest backup
	@echo "🔄 Restoring from latest backup..."
	@latest_movies=$$(ls -t backups/movies_*.db 2>/dev/null | head -1); \
	latest_ratings=$$(ls -t backups/ratings_*.db 2>/dev/null | head -1); \
	if [ -n "$$latest_movies" ] && [ -n "$$latest_ratings" ]; then \
		cp "$$latest_movies" db/movies.db; \
		cp "$$latest_ratings" db/ratings.db; \
		echo "✅ Restored from $$latest_movies and $$latest_ratings"; \
	else \
		echo "❌ No backups found in backups/"; \
	fi

# Model Training & Evaluation
train: ## Train hybrid recommendation model
	@echo "🤖 Training recommendation model..."
	uv run python main.py train
	@echo "✅ Model trained and saved!"

eval: ## Evaluate trained model performance
	@echo "📊 Evaluating model performance..."
	uv run python main.py eval-model

# Data Operations
enrich: ## Enrich movies with LLM attributes (requires API key)
	@echo "🎬 Enriching movies with LLM attributes..."
	@if [ -z "$$OPENAI_API_KEY" ]; then \
		echo "⚠️  OPENAI_API_KEY not set - loading from .env"; \
		set -a && source .env && set +a; \
	fi
	uv run python main.py enrich --n 50

embed: ## Generate vector embeddings for semantic search
	@echo "🔍 Generating vector embeddings..."
	uv run python main.py embed-generate 550 680 13 603 11
	@echo "✅ Embeddings generated!"

embed-all: ## Generate embeddings for all movies (long running)
	@echo "🔍 Generating embeddings for all movies..."
	@echo "⚠️  This will take a long time - press Ctrl+C to cancel"
	@sleep 3
	uv run python main.py embed-all

# Demo & Usage
demo: ## Run interactive demo
	@echo "🎭 Starting interactive demo..."
	@echo "Try these commands:"
	@echo "  - uv run python main.py sample --n 5"
	@echo "  - uv run python main.py movie 550"
	@echo "  - uv run python main.py fast-recommend 5"
	@echo "  - uv run python main.py chat -i"
	@echo ""
	uv run python main.py sample --n 5

chat: ## Start interactive chat agent
	@echo "💬 Starting interactive chat agent..."
	uv run python main.py chat -i

mcp-server: ## Start MCP server for Claude Desktop
	@echo "🔌 Starting MCP server..."
	@echo "Configure Claude Desktop with this server"
	uv run python main.py mcp-server

# Provider Management
setup-env: ## Set up environment files from examples if they don't exist
	@echo "🔧 Setting up environment files..."
	@if [ ! -f .env.openai ]; then \
		cp example.openai.env .env.openai; \
		echo "📋 Created .env.openai from example - please add your OpenAI API key"; \
	else \
		echo "✅ .env.openai already exists"; \
	fi
	@if [ ! -f .env.ollama ]; then \
		cp example.ollama.env .env.ollama; \
		echo "📋 Created .env.ollama from example"; \
	else \
		echo "✅ .env.ollama already exists"; \
	fi
	@echo "💡 Edit .env.openai to add your OpenAI API key, then run 'make use-openai'"

use-openai: setup-env ## Switch to OpenAI provider
	@echo "🤖 Switching to OpenAI provider..."
	@if [ -f .env.openai ]; then \
		cp .env.openai .env; \
		echo "✅ Switched to OpenAI (gpt-4o-mini)"; \
		if grep -q "your-openai-api-key-here" .env.openai; then \
			echo "⚠️  Please edit .env.openai with your actual OpenAI API key"; \
		else \
			echo "📊 API Key configured"; \
		fi; \
	else \
		echo "❌ .env.openai not found after setup"; \
		exit 1; \
	fi

use-ollama: setup-env ## Switch to Ollama provider (Qwen3)
	@echo "🏠 Switching to Ollama provider..."
	@if [ -f .env.ollama ]; then \
		cp .env.ollama .env; \
		echo "✅ Switched to Ollama (qwen3:32b)"; \
		echo "🔗 Base URL: http://localhost:11434"; \
		echo "💡 Make sure Ollama is running: ollama serve"; \
	else \
		echo "❌ .env.ollama not found after setup"; \
		exit 1; \
	fi

provider-status: ## Show current provider configuration
	@echo "📋 Current Provider Configuration:"
	@echo "=================================="
	@if [ -f .env ]; then \
		echo "🏷️  Provider: $$(grep LLM_PROVIDER .env 2>/dev/null | cut -d'=' -f2 || echo 'Not set')"; \
		echo "🤖 Model: $$(grep MODEL_NAME .env 2>/dev/null | cut -d'=' -f2 || echo 'Not set')"; \
		echo "🔗 Base URL: $$(grep '^BASE_URL=' .env 2>/dev/null | cut -d'=' -f2 || echo 'Not set')"; \
		echo "🔑 API Key: $$(grep '^API_KEY=' .env 2>/dev/null | cut -d'=' -f2 | head -c 20)..."; \
		echo "📐 Model: $$(grep '^MODEL=' .env 2>/dev/null | cut -d'=' -f2 || echo 'Not set')"; \
		echo "🎛️  Max Tokens: $$(grep '^MAX_TOKENS=' .env 2>/dev/null | cut -d'=' -f2 || echo 'Not set')"; \
		echo "🌡️  Temperature: $$(grep '^TEMPERATURE=' .env 2>/dev/null | cut -d'=' -f2 || echo 'Not set')"; \
	else \
		echo "❌ No .env file found - run 'make use-openai' or 'make use-ollama'"; \
	fi

test-provider: provider-status ## Test current provider with sample request
	@echo ""
	@echo "🧪 Testing current provider..."
	@if [ -f .env ]; then \
		uv run python -c "from src.config import get_model_name; print('✅ Provider working correctly')"; \
	else \
		echo "❌ No provider configured"; \
	fi

# Code Quality
lint: ## Run linting with ruff
	@echo "🔍 Linting code..."
	uv run ruff check src/ tests/ --fix || echo "Install ruff: uv add --dev ruff"

format: ## Format code with ruff
	@echo "🎨 Formatting code..."
	uv run ruff format src/ tests/ || echo "Install ruff: uv add --dev ruff"

check: ## Run all code quality checks
	@echo "✅ Running code quality checks..."
	$(MAKE) lint
	$(MAKE) format
	$(MAKE) test

# Docker Operations
docker-build: ## Build Docker image
	@echo "🐳 Building Docker image..."
	docker build -t movie-recommender:latest .
	@echo "✅ Docker image built!"

docker-run: ## Run Docker container
	@echo "🐳 Running Docker container..."
	docker run -it --rm \
		-v $$(pwd)/db:/app/db \
		-v $$(pwd)/.env:/app/.env \
		movie-recommender:latest

docker-shell: ## Get shell in Docker container
	@echo "🐳 Opening shell in Docker container..."
	docker run -it --rm \
		-v $$(pwd)/db:/app/db \
		-v $$(pwd)/.env:/app/.env \
		movie-recommender:latest /bin/bash

docker-compose-up: ## Start services with docker-compose
	@echo "🐳 Starting services with docker-compose..."
	docker-compose up -d
	@echo "✅ Services started!"

docker-compose-down: ## Stop docker-compose services
	@echo "🐳 Stopping docker-compose services..."
	docker-compose down
	@echo "✅ Services stopped!"

docker-compose-logs: ## View docker-compose logs
	docker-compose logs -f

# Cleanup
clean: ## Clean temporary files and caches
	@echo "🧹 Cleaning up..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf coverage.json
	rm -rf dist/
	rm -rf build/
	@echo "✅ Cleanup complete!"

clean-models: ## Clean trained models
	@echo "🧹 Cleaning trained models..."
	rm -rf models/*.pkl
	@echo "✅ Models cleaned!"

clean-all: clean clean-models ## Clean everything including models

# Production
backup-prod: db-backup ## Create production backup
	@echo "🏭 Creating production backup..."
	@timestamp=$$(date +"%Y%m%d_%H%M%S"); \
	tar -czf "backups/full_backup_$$timestamp.tar.gz" \
		db/ models/ .env.example README.md main.py src/ tests/; \
	echo "✅ Full backup: backups/full_backup_$$timestamp.tar.gz"

health-check: ## Run system health check
	@echo "🏥 Running health check..."
	@echo "Testing database..."
	$(MAKE) db-test
	@echo "Testing recommendations..."
	uv run python main.py fast-recommend 5 --n 3
	@echo "Testing embeddings..."
	uv run python main.py embed-stats
	@echo "✅ Health check complete!"

# CI/CD
ci: ## Run CI pipeline locally
	@echo "🔄 Running CI pipeline..."
	$(MAKE) install
	$(MAKE) check
	$(MAKE) test-coverage
	$(MAKE) docker-build
	@echo "✅ CI pipeline complete!"

# Quick commands for development
quick-test: ## Quick test run (no coverage)
	uv run pytest tests/ -x -v

quick-demo: ## Quick demo with sample data
	uv run python main.py sample --n 3
	uv run python main.py movie 550
	uv run python main.py fast-recommend 5 --n 3

# Help for specific areas
help-docker: ## Show Docker-specific help
	@echo "Docker Commands:"
	@echo "==============="
	@echo "make docker-build     - Build the Docker image"
	@echo "make docker-run       - Run container interactively"
	@echo "make docker-shell     - Get bash shell in container"
	@echo "make docker-compose-up - Start all services"

help-ml: ## Show ML/AI-specific help
	@echo "ML/AI Commands:"
	@echo "==============="
	@echo "make train           - Train recommendation model"
	@echo "make eval            - Evaluate model performance"
	@echo "make enrich          - Enrich movies with LLM"
	@echo "make embed           - Generate vector embeddings"
	@echo "make chat            - Interactive chat agent"

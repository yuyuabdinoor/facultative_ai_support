# Makefile for AI Reinsurance System Development

.PHONY: help setup-local install dev dev-local build clean logs

help: ## Show this help message
	@echo "AI Reinsurance System - Development Commands"
	@echo "==========================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup-local: ## Setup and validate local Python environment (optional - for custom paths)
	@echo "🔧 Validating local development environment..."
	python setup-local-dev.py

install: ## Install Python dependencies locally
	@echo "📦 Installing Python dependencies..."
	pip install -r backend/requirements.txt

dev: ## Start development environment (uses your local Python packages - default)
	@echo "🚀 Starting development environment (using local Python packages)..."
	docker-compose up --build

dev-full: ## Start development environment (builds containers with all packages)
	@echo "🚀 Starting development environment (with full package installation)..."
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml up --build

build: ## Build all containers
	@echo "🔨 Building containers..."
	docker-compose build

clean: ## Clean up containers and volumes
	@echo "🧹 Cleaning up..."
	docker-compose down -v
	docker system prune -f

logs: ## Show logs from all services
	docker-compose logs -f

logs-backend: ## Show logs from backend service only
	docker-compose logs -f backend

logs-celery: ## Show logs from celery services only
	docker-compose logs -f celery_worker celery_beat
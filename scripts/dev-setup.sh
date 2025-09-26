#!/bin/bash

# Development setup script for AI Facultative Reinsurance System

set -e

echo "🚀 Setting up AI Facultative Reinsurance System development environment..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if Docker Compose is available
if ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose is not available. Please install Docker Compose and try again."
    exit 1
fi

# Create environment files from examples
echo "📝 Creating environment files..."
if [ ! -f backend/.env ]; then
    cp backend/.env.example backend/.env
    echo "✅ Created backend/.env from example"
fi

if [ ! -f frontend/.env.local ]; then
    cp frontend/.env.example frontend/.env.local
    echo "✅ Created frontend/.env.local from example"
fi

# Create uploads directory
echo "📁 Creating uploads directory..."
mkdir -p uploads
chmod 755 uploads

# Build and start services
echo "🏗️  Building Docker containers..."
docker compose build

echo "🚀 Starting services..."
docker compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 15

# Check if services are running
echo "🔍 Checking service health..."
if curl -f http://localhost:8005/health > /dev/null 2>&1; then
    echo "✅ Backend is running at http://localhost:8005"
else
    echo "⚠️  Backend may not be ready yet. Check logs with: make logs"
fi

if curl -f http://localhost:3000 > /dev/null 2>&1; then
    echo "✅ Frontend is running at http://localhost:3000"
else
    echo "⚠️  Frontend may not be ready yet. Check logs with: make logs"
fi

echo ""
echo "🎉 Development environment setup complete!"
echo ""
echo "📋 Next steps:"
echo "   • Frontend: http://localhost:3000"
echo "   • Backend API: http://localhost:8005"
echo "   • API Docs: http://localhost:8005/docs"
echo ""
echo "🛠️  Useful commands:"
echo "   • View logs: make logs"
echo "   • Stop services: make down"
echo "   • Run tests: make test"
echo "   • Format code: make format"
echo ""
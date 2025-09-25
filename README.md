# AI-Powered Facultative Reinsurance Decision Support System

An intelligent system that automates the analysis and decision-making process for facultative reinsurance applications using AI technologies including OCR, document processing, and risk analysis models.

## Features

- **Multi-format Document Processing**: Supports PDFs, scanned PDFs, emails (.msg), and Excel files
- **AI-Powered Data Extraction**: Uses open-source models for intelligent data extraction and structuring
- **Risk Analysis**: Automated risk assessment with loss history analysis and catastrophe exposure modeling
- **Decision Support**: Intelligent recommendations with rationale and confidence scoring
- **Market Grouping**: Automatic classification and grouping of applications by market segments
- **Modern Web Interface**: Built with NextJS and Tailwind CSS for optimal user experience

## Technology Stack

### Backend
- **FastAPI**: Modern Python web framework
- **PostgreSQL**: Primary database
- **Redis**: Caching and message broker
- **Celery**: Distributed task processing
- **Hugging Face Transformers**: Open-source AI models
- **DOCTR**: OCR processing

### Frontend
- **NextJS 14**: React framework with TypeScript
- **Tailwind CSS**: Utility-first CSS framework
- **Shadcn/ui**: Modern UI components
- **Zustand**: State management

### Infrastructure
- **Docker**: Containerization
- **Docker Compose**: Development orchestration

## Quick Start

### Prerequisites
- Docker and Docker Compose
- Python 3.12.4 (for local development)
- Git

### Fast Development Setup (Recommended)

This project is optimized for fast development by mounting your local Python packages into Docker containers, avoiding the need to reinstall heavy ML libraries like PyTorch.

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ai-facultative-reinsurance-system
   ```

2. **Set up Python virtual environment**
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Start the development environment**
   ```bash
   # From project root
   make dev
   ```

4. **Access the applications**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

### Alternative Setup Options

#### Full Container Build (Slower)
If you prefer complete isolation or don't want to install packages locally:
```bash
make dev-full
```

#### Validate Local Environment
To check your Python environment and generate custom configurations:
```bash
make setup-local
```

## Development Commands

### Available Make Commands
```bash
# Development (uses local Python packages - fast!)
make dev                    # Start with local Python environment
make dev-full              # Start with full container builds (slower)

# Setup and validation
make setup-local           # Validate local Python environment
make install               # Install Python dependencies locally

# Container management
make build                 # Build all containers
make clean                 # Clean up containers and volumes

# Monitoring
make logs                  # Show logs from all services
make logs-backend          # Show backend logs only
make logs-celery           # Show celery logs only

# Help
make help                  # Show all available commands
```

### Development Workflow

#### Fast Development (Recommended)
1. Install packages locally once: `make install`
2. Start development: `make dev`
3. Your local packages are mounted into containers - no reinstallation needed!

#### Full Isolation Development
1. Use full container builds: `make dev-full`
2. Packages are installed inside containers (takes 30+ minutes initially)

### Local Development (Without Docker)
If you prefer to run services locally:

```bash
# Start PostgreSQL and Redis with Docker
docker-compose up db redis

# Backend (in separate terminal)
cd backend
source venv/bin/activate
uvicorn app.main:app --reload

# Frontend (in separate terminal)
cd frontend
npm install
npm run dev
```

## Development Architecture

### Optimized Development Setup

This project uses an innovative approach to speed up development by mounting local Python packages into Docker containers:

- **Local Python Environment**: Install heavy ML packages (PyTorch, transformers) once locally
- **Docker Services**: Run PostgreSQL, Redis, and application containers
- **Volume Mounting**: Mount local `backend/venv/lib/python3.12/site-packages` into containers
- **Fast Iteration**: No package reinstallation, instant startup, hot reload

### Benefits

- ⚡ **Fast startup**: Containers start in seconds instead of 30+ minutes
- 🔄 **Hot reload**: Code changes reflect immediately
- 💾 **Disk space**: No duplicate package installations
- 🔧 **Easy debugging**: Use local IDE with installed packages

## Project Structure

```
├── backend/                 # FastAPI backend
│   ├── venv/               # Python virtual environment (mounted into containers)
│   ├── app/
│   │   ├── agents/         # AI processing agents
│   │   ├── api/            # API routes
│   │   ├── core/           # Core configuration
│   │   ├── models/         # Data models
│   │   └── services/       # Business logic
│   ├── tests/              # Backend tests
│   ├── alembic/            # Database migrations
│   ├── Dockerfile.local    # Lightweight dev container
│   ├── Dockerfile.dev      # Full dev container
│   ├── Dockerfile          # Production container
│   └── requirements.txt    # Python dependencies
├── frontend/               # NextJS frontend
│   ├── src/
│   │   ├── app/           # Next.js app directory
│   │   ├── components/    # React components
│   │   └── lib/           # Utilities
│   └── package.json       # Node.js dependencies
├── docker-compose.yml      # Main development configuration
├── docker-compose.override.yml  # Development overrides
├── docker-compose.local.yml     # Alternative local config
├── docker-compose.prod.yml      # Production configuration
├── Makefile               # Development commands
├── setup-local-dev.py     # Environment setup script
└── docs/                  # Documentation
```

## API Documentation

Once the backend is running, visit http://localhost:8000/docs for interactive API documentation.

## Testing

### Backend Tests
```bash
cd backend
python -m pytest tests/ -v
```

### Frontend Tests
```bash
cd frontend
npm test
```

## Troubleshooting

### Common Issues

#### Slow Container Builds
**Problem**: Docker builds taking 30+ minutes
**Solution**: Use the optimized development setup
```bash
# Install packages locally first
cd backend && pip install -r requirements.txt
# Then use fast development mode
make dev
```

#### Package Import Errors
**Problem**: Python packages not found in containers
**Solution**: Ensure virtual environment exists and is properly mounted
```bash
# Check if venv exists
ls backend/venv/lib/python3.12/site-packages/
# If not, create it
cd backend && python -m venv venv && pip install -r requirements.txt
```

#### Container Permission Issues
**Problem**: Permission denied errors with mounted volumes
**Solution**: Check file permissions and Docker setup
```bash
# On Linux, ensure proper permissions
sudo chown -R $USER:$USER backend/venv/
```

#### Database Connection Errors
**Problem**: Cannot connect to PostgreSQL
**Solution**: Ensure database container is running
```bash
# Check container status
docker-compose ps
# Restart database if needed
docker-compose restart db
```

### Getting Help

- Check the [backend README](backend/README.md) for detailed backend setup
- Run `make help` to see all available commands
- Use `make logs` to view container logs
- Check Docker container status with `docker-compose ps`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes following the development setup
4. Run tests: `make test` (when implemented)
5. Submit a pull request

### Development Guidelines

- Use the optimized development setup for faster iteration
- Follow Python PEP 8 style guidelines
- Add type hints to all functions
- Write tests for new features
- Update documentation as needed

## License

This project is licensed under the MIT License - see the LICENSE file for details.
# Backend - AI Facultative Reinsurance System

FastAPI-based backend service for the AI-Powered Facultative Reinsurance Decision Support System.

## Architecture

The backend follows a multi-agent architecture using Celery for distributed processing:

- **Document Processing Agent**: OCR and text extraction
- **Data Extraction Agent**: NER and structured data extraction  
- **Risk Analysis Agent**: Risk assessment and scoring
- **Decision Engine Agent**: Intelligent recommendations
- **Market Grouping Agent**: Market classification and grouping
- **Limits Validation Agent**: Business rules validation

## Technology Stack

- **FastAPI**: Modern Python web framework with automatic API documentation
- **SQLAlchemy**: ORM for database operations
- **Alembic**: Database migration management
- **Celery**: Distributed task queue for agent processing
- **Redis**: Message broker and caching
- **PostgreSQL**: Primary database
- **Pydantic**: Data validation and serialization

### AI/ML Libraries

- **PyTorch**: Deep learning framework
- **Transformers**: Hugging Face transformer models
- **Sentence Transformers**: Text embeddings
- **DOCTR**: Document OCR processing
- **scikit-learn**: Traditional ML algorithms

## Development Setup

### Prerequisites

- Python 3.12.4
- Docker and Docker Compose (for services)

### Quick Start

1. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start with Docker (recommended)**
   ```bash
   # From project root
   make dev
   ```

   This will:
   - Start PostgreSQL and Redis in containers
   - Mount your local Python packages into backend containers
   - Start FastAPI with hot reload
   - Start Celery worker and beat scheduler

### Alternative Development Options

#### Full Container Development
```bash
# Installs all packages inside containers (slower)
make dev-full
```

#### Local Services Only
```bash
# Start only database and Redis
docker-compose up db redis

# Run backend locally
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Run Celery worker (separate terminal)
celery -A app.celery worker --loglevel=info

# Run Celery beat (separate terminal)  
celery -A app.celery beat --loglevel=info
```

## Configuration

### Environment Variables

Create a `.env` file in the backend directory:

```bash
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/reinsurance

# Redis
REDIS_URL=redis://localhost:6379
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0

# Development
DEBUG=true
LOG_LEVEL=DEBUG

# Security (generate secure keys for production)
SECRET_KEY=your-secret-key-here
```

### Docker Configuration

The backend uses multiple Dockerfile configurations:

- **Dockerfile.local**: Lightweight container that mounts local Python packages
- **Dockerfile.dev**: Full development container with all packages installed
- **Dockerfile**: Production-ready multi-stage build

## API Documentation

Once running, visit:
- **Interactive docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

## Database

### Migrations

```bash
# Create new migration
alembic revision --autogenerate -m "Description"

# Apply migrations
alembic upgrade head

# Rollback migration
alembic downgrade -1
```

### Models

Key database models:
- `Application`: Main reinsurance application
- `Document`: Uploaded documents
- `RiskParameters`: Extracted risk data
- `RiskAnalysis`: Analysis results
- `Recommendation`: Decision recommendations

## Testing

### Run Tests
```bash
# All tests
python -m pytest

# With coverage
python -m pytest --cov=app tests/

# Specific test file
python -m pytest tests/test_main.py -v
```

### Test Structure
```
tests/
├── conftest.py          # Test configuration and fixtures
├── test_main.py         # API endpoint tests
├── test_agents/         # Agent-specific tests
├── test_models/         # Database model tests
└── test_services/       # Business logic tests
```

## Code Quality

### Linting and Formatting
```bash
# Format code
black app/ tests/

# Check linting
flake8 app/ tests/

# Type checking
mypy app/
```

### Pre-commit Hooks
```bash
# Install pre-commit
pip install pre-commit
pre-commit install

# Run manually
pre-commit run --all-files
```

## Agents and Processing

### Document Processing Flow

1. **Upload**: Document uploaded via API
2. **OCR Agent**: Extract text from PDFs/images
3. **Data Extraction Agent**: Extract structured risk data
4. **Risk Analysis Agent**: Analyze risk factors
5. **Limits Validation Agent**: Check business constraints
6. **Decision Engine Agent**: Generate recommendations
7. **Market Grouping Agent**: Classify by market

### Celery Tasks

Key Celery tasks:
- `process_document`: Main document processing pipeline
- `extract_risk_data`: Data extraction from text
- `analyze_risk`: Risk assessment
- `generate_recommendation`: Decision generation

### Monitoring Tasks
```bash
# Monitor Celery workers
celery -A app.celery inspect active

# Monitor task queue
celery -A app.celery inspect reserved

# Celery flower (web monitoring)
pip install flower
celery -A app.celery flower
```

## AI Models

### Hugging Face Models Used

- **NER**: `dbmdz/bert-large-cased-finetuned-conll03-english`
- **Financial Analysis**: `ProsusAI/finbert`
- **Document Layout**: `microsoft/layoutlmv3-base`
- **Zero-shot Classification**: `facebook/bart-large-mnli`
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
- **Decision Generation**: `microsoft/DialoGPT-medium`

### Model Management

Models are automatically downloaded on first use and cached locally. For production:

1. Pre-download models during container build
2. Use model registry for version management
3. Implement model warm-up procedures

## Performance Optimization

### Docker Layer Caching

The development setup uses Docker layer caching to avoid reinstalling packages:

1. **requirements.txt** copied first for better caching
2. **Local packages mounted** to avoid installation time
3. **Multi-stage builds** for production optimization

### Database Optimization

- Use database indexes for frequent queries
- Implement connection pooling
- Use read replicas for heavy read workloads

### Celery Optimization

- Configure worker concurrency based on CPU cores
- Use task routing for different agent types
- Implement task result expiration

## Deployment

### Production Considerations

1. **Environment Variables**: Use secure secret management
2. **Database**: Use managed PostgreSQL service
3. **Redis**: Use managed Redis service  
4. **Monitoring**: Implement health checks and metrics
5. **Logging**: Centralized logging with ELK stack
6. **Security**: HTTPS, authentication, input validation

### Docker Production Build
```bash
# Build production image
docker build -f Dockerfile -t reinsurance-backend:latest .

# Run with production settings
docker run -e DATABASE_URL=... -e REDIS_URL=... reinsurance-backend:latest
```

## Troubleshooting

### Common Issues

1. **Package Installation Slow**: Use local development setup with `make dev`
2. **Import Errors**: Ensure PYTHONPATH is set correctly in containers
3. **Database Connection**: Check PostgreSQL is running and accessible
4. **Celery Tasks Failing**: Check Redis connection and worker logs
5. **Model Download Errors**: Ensure internet access for Hugging Face models

### Debug Mode

Enable debug logging:
```bash
export LOG_LEVEL=DEBUG
export DEBUG=true
```

### Container Debugging
```bash
# Access running container
docker exec -it reinsurance_backend bash

# Check Python path
docker exec -it reinsurance_backend python -c "import sys; print(sys.path)"

# Check installed packages
docker exec -it reinsurance_backend pip list
```

## Contributing

1. Follow PEP 8 style guidelines
2. Add type hints to all functions
3. Write tests for new features
4. Update documentation
5. Use conventional commit messages

### Code Structure

```
app/
├── __init__.py
├── main.py              # FastAPI application
├── celery.py           # Celery configuration
├── agents/             # AI processing agents
│   ├── ocr_agent.py
│   ├── extraction_agent.py
│   ├── risk_agent.py
│   └── decision_agent.py
├── api/                # API routes
│   └── v1/
│       ├── documents.py
│       ├── applications.py
│       └── recommendations.py
├── core/               # Core configuration
│   ├── config.py
│   ├── database.py
│   └── security.py
├── models/             # SQLAlchemy models
│   ├── application.py
│   ├── document.py
│   └── risk.py
└── services/           # Business logic
    ├── document_service.py
    ├── risk_service.py
    └── recommendation_service.py
```
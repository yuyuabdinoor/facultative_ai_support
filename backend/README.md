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

### Quick Start (Containerized)

1. **Build containers** (from project root)
   ```bash
   docker compose build backend celery_worker celery_beat
   ```

2. **Start services**
   ```bash
   docker compose up -d backend celery_worker celery_beat db redis
   ```

3. **Access**
   - API: http://localhost:8005
   - Docs: http://localhost:8005/docs
   - ReDoc: http://localhost:8005/redoc

### Alternative Development Options

#### Run locally without Docker (advanced)
```bash
# Start db and redis in docker
docker compose up -d db redis

# Then run backend locally
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Celery workers (separate terminals)
celery -A app.celery worker --loglevel=info
celery -A app.celery beat --loglevel=info
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

# Currency
# If set, the backend will fetch live FX rates and convert to KES in reports
EXCHANGE_RATE_API_KEY=your-exchangerateapi-key
# Optional base used when calling provider (default USD)
EXCHANGE_RATE_BASE=USD
```

### Docker Configuration

The backend provides multiple Dockerfiles:

- **Dockerfile.local**: Local development image (installs dependencies in-container)
- **Dockerfile**: Production-ready multi-stage build

## API Overview (Demo-relevant)

- **Documents**
  - List: `GET /api/v1/documents/`
  - Upload: `POST /api/v1/documents/upload?application_id=...`
  - Trigger OCR: `POST /api/v1/documents/{document_id}/ocr`
  - OCR status: `GET /api/v1/documents/{document_id}/ocr/status`
  - OCR text: `GET /api/v1/documents/{document_id}/ocr/text`
  - Download: `GET /api/v1/documents/{document_id}/download`

- **Email Processing**
  - Process now: `POST /api/v1/email/emails/process-now`
  - Enable polling: `POST /api/v1/email/emails/polling/enable`
  - Disable polling: `POST /api/v1/email/emails/polling/disable`
  - Polling status: `GET /api/v1/email/emails/polling/status`

- **Applications / Workflow**
  - Create application: `POST /api/v1/applications/`
  - Application status: `GET /api/v1/applications/{application_id}/status`
  - Start workflow: `POST /api/v1/task-monitoring/workflows/start?application_id=...&document_ids=...`

- **Reports**
  - Generate analysis report: `POST /api/v1/reports/applications/{application_id}/analysis-report`
  - Download report: `GET /api/v1/reports/download/{filename}`
  - Get currency rates (KES per unit): `GET /api/v1/reports/currency-rates`
  - Manually update currency rates: `POST /api/v1/reports/update-currency-rates` (admin)
  - Refresh currency rates from ExchangeRate-API: `POST /api/v1/reports/refresh-currency-rates`

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

- Models download on first use and cache to `/app/.cache/huggingface` in containers.
- For offline or production builds, prefetch during image build and mount a persistent cache volume.

## Performance Optimization

### Notes on Dependencies (PaddleOCR & Transformers)

- Backend images install `paddlepaddle==2.6.1` and `paddleocr==2.7.0.3` to enable OCR.
- Transformers pipelines are initialized by preloading models/tokenizers with `cache_dir`, then constructing pipelines without the `cache_dir` parameter (compat with current transformers).

### Currency Conversion to KES

- Reports convert financial amounts to KES using an internal rates table.
- If `EXCHANGE_RATE_API_KEY` is set, rates are refreshed from ExchangeRate-API on startup and can be refreshed via:
  - `POST /api/v1/reports/refresh-currency-rates`
- Rates are persisted to `/app/static/currency_rates.csv`.

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
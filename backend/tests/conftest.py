"""
Pytest configuration and fixtures for the AI Facultative Reinsurance System
"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.main import app
from app.core.config import settings

# Test database URL (use SQLite for testing)
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture
def client():
    """Create a test client for the FastAPI application"""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def db_session():
    """Create a test database session"""
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def sample_document_data():
    """Sample document data for testing"""
    return {
        "filename": "test_document.pdf",
        "document_type": "pdf",
        "metadata": {"size": 1024, "pages": 5}
    }


@pytest.fixture
def sample_risk_parameters():
    """Sample risk parameters for testing"""
    return {
        "asset_value": 1000000.0,
        "coverage_limit": 800000.0,
        "asset_type": "Commercial Building",
        "location": "New York, NY",
        "industry_sector": "Manufacturing"
    }
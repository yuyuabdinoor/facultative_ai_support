"""
Database connection and session management
"""
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from typing import Generator, AsyncGenerator
import logging

from app.core.config import settings
from app.models.database import Base

logger = logging.getLogger(__name__)

# Create async database engine
async_engine = create_async_engine(
    settings.database_url.replace("postgresql://", "postgresql+asyncpg://"),
    pool_pre_ping=True,
    echo=False,  # Set to True for SQL query logging in development
    connect_args={
        "server_settings": {
            "application_name": "reinsurance_backend",
        },
        # Disable SSL for local development to avoid certificate issues
        "ssl": False,
    }
)

# Create async session factory
AsyncSessionLocal = async_sessionmaker(
    async_engine,
    class_=AsyncSession,
    expire_on_commit=False
)

# Create synchronous database engine (for migrations and other sync operations)
engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    echo=False,  # Set to True for SQL query logging in development
)

# Create sync session factory (for migrations)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def create_tables():
    """Create all database tables"""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")
        raise


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency to get async database session
    
    Yields:
        AsyncSession: SQLAlchemy async database session
    """
    async with AsyncSessionLocal() as session:
        yield session

def get_sync_db() -> Generator[Session, None, None]:
    """
    Dependency to get synchronous database session (for migrations, etc.)
    
    Yields:
        Session: SQLAlchemy database session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_db_session() -> Session:
    """
    Get a database session for use outside of FastAPI dependency injection
    
    Returns:
        Session: SQLAlchemy database session
    """
    return SessionLocal()


class DatabaseManager:
    """Database management utilities"""
    
    @staticmethod
    def init_db():
        """Initialize database with tables and seed data"""
        try:
            create_tables()
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
    
    @staticmethod
    def reset_db():
        """Reset database by dropping and recreating all tables"""
        try:
            Base.metadata.drop_all(bind=engine)
            Base.metadata.create_all(bind=engine)
            logger.info("Database reset successfully")
        except Exception as e:
            logger.error(f"Error resetting database: {e}")
            raise
    
    @staticmethod
    def check_connection():
        """Check database connection"""
        try:
            with engine.connect() as connection:
                connection.execute("SELECT 1")
            logger.info("Database connection successful")
            return True
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return False
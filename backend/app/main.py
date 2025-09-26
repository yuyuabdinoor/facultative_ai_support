"""
FastAPI main application entry point
"""
# Import compatibility patch first to fix Python 3.12+ collections issues
from . import compat_startup

from fastapi import FastAPI, Request, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.openapi.utils import get_openapi
import logging
import os
import asyncio
from .api.v1 import api_router
from .agents.email_processing_agent import EmailProcessingAgent, EmailProcessingConfig
from .core.config import settings
from .middleware.security import (
    SecurityHeadersMiddleware,
    RateLimitMiddleware,
    RequestLoggingMiddleware,
    InputValidationMiddleware,
    MaintenanceModeMiddleware,
    maintenance_middleware
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Custom OpenAPI schema
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="AI Facultative Reinsurance System API",
        version="1.0.0",
        description="""
        ## AI-Powered Facultative Reinsurance Decision Support System
        
        This API provides comprehensive endpoints for managing facultative reinsurance applications
        through an AI-powered decision support system.
        
        ### Features
        - **Document Processing**: Upload and process various document formats (PDF, Excel, Email)
        - **AI Analysis**: Automated risk analysis and decision recommendations
        - **Application Management**: Complete application lifecycle management
        - **Analytics**: Comprehensive reporting and analytics
        - **Administration**: System administration and monitoring
        
        ### Authentication
        The API supports multiple authentication methods:
        - **JWT Bearer Tokens**: For web application access
        - **API Keys**: For programmatic access (use X-API-Key header)
        
        ### Rate Limiting
        API requests are rate limited to 60 requests per minute per IP address.
        
        ### Security
        All endpoints implement comprehensive security measures including:
        - Input validation and sanitization
        - SQL injection protection
        - XSS protection
        - CORS policy enforcement
        - Security headers
        """,
        routes=app.routes,
    )
    
    # Add security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT"
        },
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key"
        }
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

# Create FastAPI application
app = FastAPI(
    title="AI Facultative Reinsurance System",
    description="AI-Powered Facultative Reinsurance Decision Support System",
    version="1.0.0",
    docs_url=None,  # Disable default docs
    redoc_url=None,  # Disable default redoc
    openapi_url="/api/openapi.json"
)

# Set custom OpenAPI schema
app.openapi = custom_openapi

# Configure CORS with security considerations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    # During local development allow all headers (keeps CORS flexible for the frontend).
    # In production this should be restricted to required headers.
    allow_headers=["*"],
    expose_headers=["X-Process-Time", "X-RateLimit-Remaining"]
)

# Add security middleware (order matters!)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RateLimitMiddleware, requests_per_minute=60)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(InputValidationMiddleware)

# Add maintenance mode middleware
maintenance_middleware.app = app
app.add_middleware(MaintenanceModeMiddleware)


# Global exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.status_code,
                "message": exc.detail,
                "type": "http_error"
            },
            "request_id": getattr(request.state, "request_id", None)
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled exceptions"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": 500,
                "message": "Internal server error",
                "type": "internal_error"
            },
            "request_id": getattr(request.state, "request_id", None)
        }
    )

# Include API routes
app.include_router(api_router, prefix=settings.api_v1_prefix)

# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint providing API information
    """
    return {
        "message": "AI Facultative Reinsurance System API",
        "version": "1.0.0",
        "documentation": "/docs",
        "health_check": "/health",
        "api_prefix": settings.api_v1_prefix
    }

# Custom documentation endpoints
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    """Custom Swagger UI with enhanced security"""
    return get_swagger_ui_html(
        openapi_url="/api/openapi.json",
        title="AI Reinsurance API - Documentation"
    )

@app.get("/redoc", include_in_schema=False)
async def redoc_html():
    """Custom ReDoc documentation"""
    return get_redoc_html(
        openapi_url="/api/openapi.json",
        title="AI Reinsurance API - Documentation"
    )

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Application startup tasks"""
    logger.info("Starting AI Facultative Reinsurance System API")
    logger.info(f"API version: 1.0.0")
    logger.info(f"Environment: {getattr(settings, 'environment', 'development')}")
    
    # Initialize any required services here
    # - Database connections
    # - Redis connections
    # - Model loading
    # - Cache warming
    # Optionally enable background email polling
    try:
        enable_polling = getattr(settings, "email_polling_enabled", False)
        if not enable_polling:
            logger.info("Email polling disabled (email_polling_enabled is False)")
            return

        cfg = EmailProcessingConfig(
            imap_server=getattr(settings, "email_imap_server", "") or "",
            email_user=getattr(settings, "email_user", "") or "",
            email_password=getattr(settings, "email_password", "") or "",
            processed_folder=getattr(settings, "email_processed_folder", "Processed"),
            error_folder=getattr(settings, "email_error_folder", "Error"),
            download_dir=getattr(settings, "email_download_dir", "/tmp/email_attachments"),
            check_interval=int(getattr(settings, "check_interval", 300)),
            max_attachment_size=int(getattr(settings, "max_attachment_size", 50 * 1024 * 1024)),
        )

        # Guard: require IMAP server and credentials
        if not cfg.imap_server or not cfg.email_user or not cfg.email_password:
            logger.warning("Email polling is enabled but IMAP settings are incomplete; skipping email polling startup")
            return

        app.state.email_agent = EmailProcessingAgent(cfg)

        async def _poll_loop():
            while True:
                try:
                    await asyncio.get_event_loop().run_in_executor(None, app.state.email_agent.process_new_emails)
                except Exception as e:
                    logger.error(f"Email polling cycle error: {e}")
                await asyncio.sleep(app.state.email_agent.config.check_interval)

        app.state.email_polling_task = asyncio.create_task(_poll_loop())
        logger.info("Email polling enabled on startup")
    except Exception as e:
        logger.error(f"Failed to start email polling: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown tasks"""
    logger.info("Shutting down AI Facultative Reinsurance System API")
    
    # Cleanup tasks here
    # - Close database connections
    # - Save any pending data
    # - Cleanup temporary files
    # Stop email polling if running
    task = getattr(app.state, "email_polling_task", None)
    if task:
        task.cancel()
        app.state.email_polling_task = None

# Health check endpoint (simple)
@app.get("/health", tags=["Health"])
async def basic_health_check():
    """
    Basic health check endpoint for load balancers
    """
    return {
        "status": "healthy",
        "timestamp": "2024-01-15T12:00:00Z",
        "version": "1.0.0"
    }

# Maintenance mode endpoints
@app.post("/admin/maintenance/enable", tags=["Admin"])
async def enable_maintenance_mode(message: str = "System maintenance in progress"):
    """Enable maintenance mode (admin only)"""
    maintenance_middleware.enable_maintenance(message)
    return {"message": "Maintenance mode enabled", "maintenance_message": message}

@app.post("/admin/maintenance/disable", tags=["Admin"])
async def disable_maintenance_mode():
    """Disable maintenance mode (admin only)"""
    maintenance_middleware.disable_maintenance()
    return {"message": "Maintenance mode disabled"}
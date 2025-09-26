"""
API v1 routes
"""
from fastapi import APIRouter
from .auth import router as auth_router
from .documents import router as documents_router
from .applications import router as applications_router
from .business_limits import router as business_limits_router
from .market_grouping import router as market_grouping_router
from .task_monitoring import router as task_monitoring_router
from .analytics import router as analytics_router
from .admin import router as admin_router
from .health import router as health_router
from .email_processing import router as email_processing_router
from .reports import router as reports_router

api_router = APIRouter()

# Authentication routes
api_router.include_router(
    auth_router,
    prefix="/auth",
    tags=["Authentication"]
)

# Document management routes
api_router.include_router(
    documents_router,
    prefix="/documents",
    tags=["Documents"]
)

# Application management routes
api_router.include_router(
    applications_router,
    prefix="/applications",
    tags=["Applications"]
)

# Business limits routes
api_router.include_router(
    business_limits_router,
    prefix="/business-limits",
    tags=["Business Limits"]
)

# Market grouping routes
api_router.include_router(
    market_grouping_router,
    prefix="/market-grouping",
    tags=["Market Grouping"]
)

# Task monitoring routes
api_router.include_router(
    task_monitoring_router,
    prefix="/tasks",
    tags=["Task Monitoring"]
)

# Analytics and reporting routes
api_router.include_router(
    analytics_router,
    prefix="/analytics",
    tags=["Analytics"]
)

# System administration routes
api_router.include_router(
    admin_router,
    prefix="/admin",
    tags=["Administration"]
)

# Health and monitoring routes
api_router.include_router(
    health_router,
    prefix="/health",
    tags=["Health & Monitoring"]
)

# Email processing routes
api_router.include_router(
    email_processing_router,
    prefix="/email",
    tags=["Email Processing"]
)

# Report generation routes
api_router.include_router(
    reports_router,
    tags=["Reports"]
)
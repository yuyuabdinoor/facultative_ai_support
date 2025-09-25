"""
API v1 routes
"""
from fastapi import APIRouter
from .documents import router as documents_router

api_router = APIRouter()

# Include document routes
api_router.include_router(
    documents_router,
    prefix="/documents",
    tags=["documents"]
)
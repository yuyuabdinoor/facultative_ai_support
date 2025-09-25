"""
FastAPI main application entry point
"""
# Import compatibility patch first to fix Python 3.12+ collections issues
from . import compat_startup

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.v1 import api_router
from .core.config import settings

app = FastAPI(
    title="AI Facultative Reinsurance System",
    description="AI-Powered Facultative Reinsurance Decision Support System",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # NextJS frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix=settings.api_v1_prefix)

@app.get("/")
async def root():
    return {"message": "AI Facultative Reinsurance System API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
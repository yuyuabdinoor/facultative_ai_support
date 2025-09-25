#!/usr/bin/env python3
"""
Simple server runner with compatibility fixes
"""

# Import compatibility patches FIRST
from app.compat_startup import *

# Now import and run uvicorn
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Disable reload to avoid multiprocessing issues
        log_level="info"
    )
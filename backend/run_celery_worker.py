#!/usr/bin/env python3
"""
Celery worker runner with compatibility fixes
"""

# Import compatibility patches FIRST
from app.compat_startup import *

# Now import and run celery worker directly
if __name__ == "__main__":
    try:
        from app.celery import celery_app
        
        # Run celery worker directly
        celery_app.worker_main([
            'worker',
            '--loglevel=info',
            '--concurrency=1'
        ])
    except Exception as e:
        print(f"Failed to start Celery worker: {e}")
        print("Celery worker functionality will be unavailable")
        # Keep the process running so Docker doesn't restart it
        import time
        while True:
            time.sleep(60)
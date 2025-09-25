#!/usr/bin/env python3
"""
Celery beat runner with compatibility fixes
"""

# Import compatibility patches FIRST
from app.compat_startup import *

# Now import and run celery beat directly
if __name__ == "__main__":
    try:
        from app.celery import celery_app
        
        # Run celery beat directly
        celery_app.start([
            'celery',
            'beat',
            '--loglevel=info'
        ])
    except Exception as e:
        print(f"Failed to start Celery beat: {e}")
        print("Celery beat functionality will be unavailable")
        # Keep the process running so Docker doesn't restart it
        import time
        while True:
            time.sleep(60)
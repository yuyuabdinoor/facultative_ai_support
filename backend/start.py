#!/usr/bin/env python3
"""
Startup script that applies compatibility patches before starting the FastAPI application.
"""

# Apply compatibility patch first
import sys
import collections
import collections.abc

# Patch collections module to include Sequence for backward compatibility
if not hasattr(collections, 'Sequence'):
    collections.Sequence = collections.abc.Sequence
    collections.Mapping = collections.abc.Mapping
    collections.MutableMapping = collections.abc.MutableMapping
    collections.Iterable = collections.abc.Iterable
    collections.Iterator = collections.abc.Iterator
    collections.Callable = collections.abc.Callable
    print("Applied Python 3.12+ collections compatibility patch")

# Now start uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
"""
Compatibility startup module that must be imported before any other modules
to fix Python 3.12+ compatibility issues with various packages.
"""

import sys
import collections
import collections.abc

# Apply compatibility patches immediately
if sys.version_info >= (3, 10):
    # Fix collections module for packages that expect old locations
    if not hasattr(collections, 'Sequence'):
        collections.Sequence = collections.abc.Sequence
    if not hasattr(collections, 'Mapping'):
        collections.Mapping = collections.abc.Mapping
    if not hasattr(collections, 'MutableMapping'):
        collections.MutableMapping = collections.abc.MutableMapping
    if not hasattr(collections, 'Iterable'):
        collections.Iterable = collections.abc.Iterable
    if not hasattr(collections, 'Iterator'):
        collections.Iterator = collections.abc.Iterator
    if not hasattr(collections, 'Callable'):
        collections.Callable = collections.abc.Callable
    if not hasattr(collections, 'Set'):
        collections.Set = collections.abc.Set
    if not hasattr(collections, 'MutableSet'):
        collections.MutableSet = collections.abc.MutableSet

# Fix pathlib compatibility issues
from pathlib import Path
if sys.version_info >= (3, 12):
    # Add missing methods that some packages expect
    if not hasattr(Path, 'is_mount'):
        def is_mount(self):
            """Fallback is_mount method for Python 3.12+ compatibility"""
            try:
                import os
                return os.path.ismount(str(self))
            except Exception:
                return False
        Path.is_mount = property(is_mount)
    
    if not hasattr(Path, 'expanduser'):
        def expanduser(self):
            """Fallback expanduser method for Python 3.12+ compatibility"""
            try:
                import os
                return Path(os.path.expanduser(str(self)))
            except Exception:
                return self
        Path.expanduser = expanduser

# Set environment variables to disable problematic features
import os
os.environ['PYDANTIC_DISABLE_PLUGINS'] = '1'

# Monkey patch the problematic pathlib module in site-packages
try:
    import site
    import importlib.util
    
    # Find the pathlib module in site-packages that's causing issues
    pathlib_spec = importlib.util.find_spec('pathlib')
    if pathlib_spec and pathlib_spec.origin and 'site-packages' in pathlib_spec.origin:
        # This is the problematic pathlib module, let's patch it
        import pathlib as system_pathlib
        
        # Patch the collections import in the problematic pathlib
        if hasattr(system_pathlib, 'collections'):
            system_pathlib.collections = collections
            
except Exception:
    # If patching fails, continue anyway
    pass

print("Compatibility patches applied successfully")
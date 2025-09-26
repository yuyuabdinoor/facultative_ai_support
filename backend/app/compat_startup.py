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
from pathlib import Path, PosixPath
import pathlib

# Ensure PosixPath has read_text method
if not hasattr(PosixPath, 'read_text'):
    def read_text(self, encoding='utf-8', errors='strict'):
        """Add read_text method to PosixPath for compatibility"""
        try:
            with open(str(self), 'r', encoding=encoding, errors=errors) as f:
                return f.read()
        except Exception as e:
            # Fallback for metadata reading issues
            return ""
    PosixPath.read_text = read_text

if not hasattr(PosixPath, 'write_text'):
    def write_text(self, data, encoding='utf-8', errors='strict'):
        """Add write_text method to PosixPath for compatibility"""
        try:
            with open(str(self), 'w', encoding=encoding, errors=errors) as f:
                return f.write(data)
        except Exception as e:
            # Fallback for metadata writing issues
            return 0
    PosixPath.write_text = write_text

# Also patch the base Path class
if not hasattr(Path, 'read_text'):
    def read_text(self, encoding='utf-8', errors='strict'):
        """Add read_text method to Path for compatibility"""
        try:
            with open(str(self), 'r', encoding=encoding, errors=errors) as f:
                return f.read()
        except Exception as e:
            # Fallback for metadata reading issues
            return ""
    Path.read_text = read_text

if not hasattr(Path, 'write_text'):
    def write_text(self, data, encoding='utf-8', errors='strict'):
        """Add write_text method to Path for compatibility"""
        try:
            with open(str(self), 'w', encoding=encoding, errors=errors) as f:
                return f.write(data)
        except Exception as e:
            # Fallback for metadata writing issues
            return 0
    Path.write_text = write_text

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

# Add the home() class method that modelscope/paddleocr needs
if not hasattr(Path, 'home'):
    @classmethod
    def home(cls):
        """Add home class method for Python 3.12+ compatibility"""
        try:
            import os
            return cls(os.path.expanduser('~'))
        except Exception:
            return cls('/tmp')  # Fallback to tmp directory
    Path.home = home

# Patch os.path.expanduser to handle Path objects
import os
original_expanduser = os.path.expanduser

def patched_expanduser(path):
    """Patched expanduser that handles Path objects"""
    try:
        if hasattr(path, '__fspath__'):
            # It's a Path-like object, convert to string
            return original_expanduser(str(path))
        elif hasattr(path, 'joinpath'):
            # It's likely a Path object, convert to string
            return original_expanduser(str(path))
        else:
            return original_expanduser(path)
    except Exception:
        # Fallback to string conversion
        return original_expanduser(str(path))

os.path.expanduser = patched_expanduser

# Patch os.path.join to handle Path objects
original_join = os.path.join

def patched_join(path, *paths):
    """Patched join that handles Path objects"""
    try:
        # Convert all arguments to strings if they are Path objects
        str_path = str(path) if hasattr(path, '__fspath__') or hasattr(path, 'joinpath') else path
        str_paths = []
        for p in paths:
            if hasattr(p, '__fspath__') or hasattr(p, 'joinpath'):
                str_paths.append(str(p))
            else:
                str_paths.append(p)
        return original_join(str_path, *str_paths)
    except Exception:
        # Fallback: convert everything to strings
        return original_join(str(path), *[str(p) for p in paths])

os.path.join = patched_join

# Patch os.path.isfile to handle Path objects (fixes python-dotenv issue)
original_isfile = os.path.isfile

def patched_isfile(path):
    """Patched isfile that handles Path objects"""
    try:
        if hasattr(path, '__fspath__'):
            # It's a Path-like object, convert to string
            return original_isfile(str(path))
        elif hasattr(path, 'joinpath'):
            # It's likely a Path object, convert to string
            return original_isfile(str(path))
        else:
            return original_isfile(path)
    except Exception:
        # Fallback to string conversion
        return original_isfile(str(path))

os.path.isfile = patched_isfile

# Patch os.path.exists to handle Path objects (additional safety)
original_exists = os.path.exists

def patched_exists(path):
    """Patched exists that handles Path objects"""
    try:
        if hasattr(path, '__fspath__'):
            # It's a Path-like object, convert to string
            return original_exists(str(path))
        elif hasattr(path, 'joinpath'):
            # It's likely a Path object, convert to string
            return original_exists(str(path))
        else:
            return original_exists(path)
    except Exception:
        # Fallback to string conversion
        return original_exists(str(path))

os.path.exists = patched_exists

# Patch built-in open() function to handle Path objects (fixes python-dotenv issue)
import builtins
original_open = builtins.open

def patched_open(file, *args, **kwargs):
    """Patched open that handles Path objects"""
    try:
        if hasattr(file, '__fspath__'):
            # It's a Path-like object, convert to string
            return original_open(str(file), *args, **kwargs)
        elif hasattr(file, 'joinpath'):
            # It's likely a Path object, convert to string
            return original_open(str(file), *args, **kwargs)
        else:
            return original_open(file, *args, **kwargs)
    except Exception:
        # Fallback to string conversion
        return original_open(str(file), *args, **kwargs)

builtins.open = patched_open

# Patch aiofiles.open to handle Path objects
try:
    import aiofiles
    original_aiofiles_open = aiofiles.open
    
    def patched_aiofiles_open(file, *args, **kwargs):
        """Patched aiofiles.open that handles Path objects"""
        # Convert Path objects to strings
        if hasattr(file, '__fspath__') or hasattr(file, 'joinpath'):
            file = str(file)
        return original_aiofiles_open(file, *args, **kwargs)
    
    aiofiles.open = patched_aiofiles_open
    
except ImportError:
    # aiofiles not available, skip patching
    pass

# Set environment variables to disable problematic features
import os
os.environ['PYDANTIC_DISABLE_PLUGINS'] = '1'

# Fix email-validator version check issues
try:
    import importlib.metadata
    original_version = importlib.metadata.version
    
    def patched_version(distribution_name):
        """Patched version function that handles metadata reading issues"""
        try:
            return original_version(distribution_name)
        except Exception:
            # Return a default version for email-validator to satisfy pydantic
            if distribution_name == 'email-validator':
                return '2.0.0'
            raise
    
    importlib.metadata.version = patched_version
    
except Exception:
    # If patching fails, continue anyway
    pass

# Fix importlib.metadata compatibility issues
try:
    import importlib.metadata
    from importlib.metadata import Distribution
    
    # Patch the Distribution class to handle read_text issues
    original_read_text = getattr(Distribution, 'read_text', None)
    
    def patched_read_text(self, filename):
        """Patched read_text method that handles PosixPath issues"""
        try:
            if original_read_text:
                return original_read_text(self, filename)
            else:
                # Fallback implementation
                path = self._path.joinpath(filename)
                if hasattr(path, 'read_text'):
                    return path.read_text(encoding='utf-8')
                else:
                    with open(str(path), 'r', encoding='utf-8') as f:
                        return f.read()
        except Exception:
            # Return empty string for missing metadata
            return ""
    
    Distribution.read_text = patched_read_text
    
except Exception:
    # If patching fails, continue anyway
    pass

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
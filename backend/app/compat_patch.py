"""
Compatibility patch for Python 3.12+ to fix collections import issues
"""

import sys
import collections
import collections.abc

# Fix for Python 3.12+ where collections.Sequence was moved to collections.abc
if sys.version_info >= (3, 10):
    # Patch collections module to include items that were moved to collections.abc
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
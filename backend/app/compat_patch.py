"""
Compatibility patch for Python 3.12+ collections.Sequence import issue.

This module patches the collections module to provide backward compatibility
for packages that still import Sequence from collections instead of collections.abc.
"""

import sys
import collections
import collections.abc

# Patch collections module to include Sequence for backward compatibility
if not hasattr(collections, 'Sequence'):
    collections.Sequence = collections.abc.Sequence

# Also patch other commonly used abstract base classes that moved
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

# Ensure the patch is applied before other imports
print("Applied Python 3.12+ collections compatibility patch")
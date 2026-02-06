import os
import sys

# Try to import from installed pointops2 first
try:
    from pointops2 import *
except ImportError:
    # Fallback: import from local build directory
    _pointops2_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _build_dir = os.path.join(_pointops2_dir, 'build', 'lib.linux-x86_64-cpython-311')
    if os.path.exists(_build_dir) and _build_dir not in sys.path:
        sys.path.insert(0, _build_dir)
    try:
        from pointops2 import *
    except ImportError:
        raise ImportError(
            "Failed to import pointops2. Please install it by running:\n"
            "  cd third_party/pointops2 && pip install -e ."
        )
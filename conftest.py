"""Root conftest: makes bare `pytest` behave like `python -m pytest` (sys.path)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

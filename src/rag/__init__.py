# src/rag package — re-exports knowledge_base interface so
# `from . import rag; rag.search(...)` in agent.py continues to work,
# and test patches like `patch("src.agent.rag.search")` stay valid.
from .knowledge_base import build_index, search

__all__ = ["build_index", "search"]

# src/workflows package
# Re-exports graph builder for backward compatibility.
from .graph import build_graph

__all__ = ["build_graph"]

"""Evaluation package for comparative model analysis.

Exports:
    ComparativeEvaluator: Iterates over registered models, accumulates confusion
        matrices, delegates to reporter and visualizer.
"""
from src.evaluation.comparative import ComparativeEvaluator  # noqa: F401

__all__ = ["ComparativeEvaluator"]

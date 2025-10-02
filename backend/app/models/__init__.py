"""
Models package for APH-IF Backend API.

Contains Pydantic models for request/response validation and structured API responses.
"""

# Re-export commonly used response models for convenient imports
from .structured_responses import (
    ConfidenceMetadata,
    EngineMetadata,
    FusionResultPayload,
    PerformanceDashboardResponse,
    SemanticResultPayload,
    StructuredHealthResponse,
    StructuredQueryResponse,
    TraversalResultPayload,
)

__all__ = [
    "EngineMetadata",
    "ConfidenceMetadata",
    "SemanticResultPayload",
    "TraversalResultPayload",
    "FusionResultPayload",
    "StructuredQueryResponse",
    "StructuredHealthResponse",
    "PerformanceDashboardResponse",
]
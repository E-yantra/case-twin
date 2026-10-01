"""Compatibility exports for the local-only AI adapter."""
from local_ai import (
    generate_embedding, query_local_model, query_medgemma,
    query_medgemma_comparison, query_medgemma_localization,
)

__all__ = [
    "generate_embedding", "query_local_model", "query_medgemma",
    "query_medgemma_comparison", "query_medgemma_localization",
]

"""Stage 1 label-description retrieval baselines."""

from .corpus import RegistryDocument, build_registry_documents
from .hybrid import build_field_label_index, field_index_scores, fuse_retrieval_scores
from .metrics import summarize_retrieval
from .ranking import stable_rank

__all__ = [
    "RegistryDocument",
    "build_registry_documents",
    "build_field_label_index",
    "field_index_scores",
    "fuse_retrieval_scores",
    "stable_rank",
    "summarize_retrieval",
]

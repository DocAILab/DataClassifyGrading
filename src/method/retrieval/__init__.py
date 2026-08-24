"""Stage 1 label-description retrieval baselines."""

from .corpus import RegistryDocument, build_registry_documents
from .metrics import summarize_retrieval
from .ranking import stable_rank

__all__ = ["RegistryDocument", "build_registry_documents", "stable_rank", "summarize_retrieval"]

"""Deterministic, provenance-bearing label corpus construction."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

from agent.task import LeafRegistry


@dataclass(frozen=True)
class RegistryDocument:
    category_id: str
    text: str
    provenance: str
    checksum: str


def build_registry_documents(registry: LeafRegistry) -> tuple[RegistryDocument, ...]:
    documents: list[RegistryDocument] = []
    for category in registry.categories:
        description = " ".join(category.description.split())
        text = f"标签名称：{category.category_id}"
        provenance = "label_name_fallback"
        if description:
            text += f"\n标签描述：{description}"
            provenance = "registry_description"
        documents.append(
            RegistryDocument(
                category_id=category.category_id,
                text=text,
                provenance=provenance,
                checksum=sha256(text.encode("utf-8")).hexdigest(),
            )
        )
    return tuple(documents)

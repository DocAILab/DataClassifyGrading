"""Frozen Registry views with a strict label-only Top-K information boundary."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


class CategoryCatalog:
    """Expose labels first and descriptions only for validated candidate IDs."""

    def __init__(self, registry_entries: Mapping[str, Mapping[str, Any]]) -> None:
        self._entries = {str(key): dict(value) for key, value in registry_entries.items()}
        if not self._entries:
            raise ValueError("category registry is empty")
        width = max(3, len(str(len(self._entries))))
        self._option_to_category = {
            f"C{index:0{width}d}": category
            for index, category in enumerate(sorted(self._entries), start=1)
        }
        self._category_to_option = {
            category: option_id for option_id, category in self._option_to_category.items()
        }

    def selection_options(self) -> list[dict[str, str]]:
        """Return stable short choices and full paths, without canonical IDs or details."""

        labels = {entry["category"]: entry["label"] for entry in self.labels()}
        return [
            {"option_id": option_id, "label": labels[category]}
            for option_id, category in self._option_to_category.items()
        ]

    def resolve_option_ids(self, option_ids: Sequence[str], k: int) -> tuple[str, ...]:
        """Validate ranked model choices before translating to frozen category IDs."""

        if k < 1 or k > len(self._entries):
            raise ValueError("top_k must be between 1 and registry size")
        if not isinstance(option_ids, (list, tuple)) or len(option_ids) != k:
            raise ValueError(f"top_k result must contain exactly {k} option IDs")
        if any(not isinstance(item, str) or item not in self._option_to_category
               for item in option_ids):
            raise ValueError("top_k result contains an unknown option ID")
        if len(set(option_ids)) != len(option_ids):
            raise ValueError("top_k result contains duplicate option IDs")
        return tuple(self._option_to_category[item] for item in option_ids)

    def option_details(self, ids: Sequence[str]) -> list[dict[str, str]]:
        """Expose descriptions only for selected choices, still without long IDs."""

        return [
            {
                "option_id": self._category_to_option[item["category"]],
                "label": item["label"],
                "description": item["description"],
            }
            for item in self.details(ids)
        ]

    def labels(self) -> list[dict[str, str]]:
        result: list[dict[str, str]] = []
        for category, entry in self._entries.items():
            path = entry.get("path")
            if isinstance(path, (list, tuple)) and path:
                label = " > ".join(str(part) for part in path)
            else:
                label = str(entry.get("label") or category.partition(":")[2] or entry.get("name") or category)
            result.append({"category": category, "label": label})
        return result

    def validate_top_k(self, ids: Sequence[str], k: int) -> tuple[str, ...]:
        if k < 1 or k > len(self._entries):
            raise ValueError("top_k must be between 1 and registry size")
        if not isinstance(ids, (list, tuple)) or len(ids) != k:
            raise ValueError(f"top_k result must contain exactly {k} IDs")
        if any(not isinstance(item, str) or item not in self._entries for item in ids):
            raise ValueError("top_k result contains an unknown category ID")
        if len(set(ids)) != len(ids):
            raise ValueError("top_k result contains duplicate category IDs")
        return tuple(ids)

    def details(self, ids: Sequence[str]) -> list[dict[str, str]]:
        if not isinstance(ids, (list, tuple)) or not ids or len(set(ids)) != len(ids):
            raise ValueError("candidate IDs must be a nonempty distinct list")
        labels = {entry["category"]: entry["label"] for entry in self.labels()}
        result: list[dict[str, str]] = []
        for category in ids:
            if category not in self._entries:
                raise ValueError(f"unknown category ID: {category}")
            entry = self._entries[category]
            result.append({
                "category": category,
                "label": labels[category],
                "description": str(entry.get("description") or entry.get("definition") or ""),
            })
        return result

    def frozen_entry(self, category: str) -> dict[str, Any]:
        if category not in self._entries:
            raise ValueError(f"unknown category ID: {category}")
        return {"category": category, **self._entries[category]}

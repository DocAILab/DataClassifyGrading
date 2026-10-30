"""Label-free, deterministic balanced presentation and exact vote aggregation."""

from __future__ import annotations

import hashlib
import json
import random
from fractions import Fraction


SELECTION_POLICY = "canonical-sha256-shuffle-cyclic3-votes-rrf-hash-v1"


def _digest(seed: int, domain: str, field_name: str, option_id: str | None = None) -> str:
    values = [seed, domain, field_name]
    if option_id is not None:
        values.append(option_id)
    return hashlib.sha256(json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def balanced_options(options: list[dict], domain: str, field_name: str, seed: int) -> list[list[dict]]:
    """Shuffle canonical IDs once, then present floor(i*N/3) cyclic offsets."""
    ids = [item.get("option_id") for item in options]
    if not ids or any(not isinstance(item, str) or not item for item in ids) or len(set(ids)) != len(ids):
        raise ValueError("options require distinct nonempty option IDs")
    canonical = [dict(item) for item in sorted(options, key=lambda item: item["option_id"])]
    random.Random(int(_digest(seed, domain, field_name), 16)).shuffle(canonical)
    rounds = []
    for index in range(3):
        offset = index * len(canonical) // 3
        rounds.append([dict(item) for item in canonical[offset:] + canonical[:offset]])
    return rounds


def ranking_scores(rankings: list[list[str]], allowed: list[str], domain: str,
                   field_name: str, seed: int, top_k: int = 5) -> list[dict]:
    """Return exact score records in the fixed aggregation order for audit traces."""
    if top_k < 1 or top_k > len(allowed) or len(set(allowed)) != len(allowed):
        raise ValueError("top_k must fit a distinct allowed catalog")
    if len(rankings) < 2:
        raise ValueError("balanced-vote requires at least two valid rankings")
    for ranking in rankings:
        if (not isinstance(ranking, list) or len(ranking) != top_k
                or any(not isinstance(item, str) or item not in allowed for item in ranking)
                or len(set(ranking)) != top_k):
            raise ValueError("each ranking requires exactly top_k distinct in-catalog IDs")
    scores = []
    for option_id in sorted(allowed):
        ranks = [ranking.index(option_id) + 1 for ranking in rankings if option_id in ranking]
        if ranks:
            score = sum((Fraction(1, rank) for rank in ranks), Fraction())
            scores.append((option_id, len(ranks), score, _digest(seed, domain, field_name, option_id)))
    scores.sort(key=lambda item: (-item[1], -item[2], item[3]))
    return [{"option_id": option_id, "votes": votes, "reciprocal_rank_score": str(score), "tie_hash": tie_hash}
            for option_id, votes, score, tie_hash in scores]


def aggregate_rankings(rankings: list[list[str]], allowed: list[str], domain: str,
                       field_name: str, seed: int, top_k: int = 5) -> list[str]:
    """Require quorum, validate all rankings and select by votes, exact RRF, hash."""
    return [item["option_id"] for item in ranking_scores(rankings, allowed, domain, field_name, seed, top_k)[:top_k]]

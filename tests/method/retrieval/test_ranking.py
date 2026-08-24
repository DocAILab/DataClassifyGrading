import numpy as np
import pytest

from method.retrieval.ranking import stable_rank


def test_stable_rank_breaks_ties_by_registry_order():
    ranked = stable_rank(np.array([0.8, 0.8, 0.9]), ("A", "B", "C"), top_k=3)
    assert [label for label, _ in ranked] == ["C", "A", "B"]
    assert [score for _, score in ranked] == pytest.approx([0.9, 0.8, 0.8])


@pytest.mark.parametrize("scores", [np.array([0.1, np.nan]), np.array([0.1])])
def test_stable_rank_rejects_invalid_score_vectors(scores):
    with pytest.raises(ValueError, match="finite vector"):
        stable_rank(scores, ("A", "B"), top_k=2)

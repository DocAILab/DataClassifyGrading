import json
from types import SimpleNamespace

import pytest
import torch

from agent.task import LeafRegistry
from method.dpo.label_scoring import (
    completion_mean_log_probs,
    load_completed_score_rows,
    mine_score_rows,
    retrieve_semantic_hard_negatives,
    score_candidate_answers,
)
from method.dpo.script.mine_preferences import adapter_identity


REGISTRY = LeafRegistry.from_mapping(
    {
        "categories": [
            {"category_id": "contact_email", "description": "electronic mail contact address"},
            {"category_id": "login_email", "description": "email used to sign in"},
            {"category_id": "backup_email", "description": "secondary email address"},
            {"category_id": "phone", "description": "telephone contact number"},
            {"category_id": "name", "description": "person full name"},
            {"category_id": "postal", "description": "postal delivery address"},
        ]
    }
)


def test_retrieve_semantic_hard_negatives_returns_deterministic_nearest_wrong_labels():
    first = retrieve_semantic_hard_negatives(
        "email_address", "contact_email", REGISTRY, count=4
    )
    second = retrieve_semantic_hard_negatives(
        "email_address", "contact_email", REGISTRY, count=4
    )

    assert first == second
    assert first[:2] == ["backup_email", "login_email"]
    assert len(first) == len(set(first)) == 4
    assert "contact_email" not in first
    assert set(first) <= set(REGISTRY.ids)


def test_retrieve_semantic_hard_negatives_requires_registry_golden_and_four_wrong():
    with pytest.raises(ValueError, match="golden"):
        retrieve_semantic_hard_negatives("email", "outside", REGISTRY)
    with pytest.raises(ValueError, match="count"):
        retrieve_semantic_hard_negatives("email", "contact_email", REGISTRY, count=6)


def test_completion_mean_log_probs_matches_full_log_softmax_reference():
    logits = torch.tensor(
        [
            [[2.0, 0.0, -1.0], [0.0, 3.0, 1.0], [1.0, -1.0, 2.0]],
            [[0.0, 1.0, 2.0], [2.0, 1.0, 0.0], [-2.0, 0.0, 2.0]],
        ]
    )
    token_ids = torch.tensor([[0, 1, 2], [2, 0, 2]])
    mask = torch.tensor([[1, 1, 1], [0, 1, 1]], dtype=torch.bool)

    actual = completion_mean_log_probs(logits, token_ids, mask)
    reference = torch.log_softmax(logits, dim=-1).gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)
    expected = (reference * mask).sum(-1) / mask.sum(-1)

    assert torch.allclose(actual, expected)


def test_completion_mean_log_probs_rejects_empty_completion():
    with pytest.raises(ValueError, match="completion"):
        completion_mean_log_probs(
            torch.zeros(1, 2, 3),
            torch.zeros(1, 2, dtype=torch.long),
            torch.zeros(1, 2, dtype=torch.bool),
        )


def test_load_completed_score_rows_supports_resume_and_rejects_duplicates(tmp_path):
    path = tmp_path / "scores.jsonl"
    rows = [
        {"source_id": "one", "scores": {"A": -1.0}},
        {"source_id": "two", "scores": {"B": -2.0}},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    assert load_completed_score_rows(path) == {"one": rows[0], "two": rows[1]}

    path.write_text(
        json.dumps(rows[0]) + "\n" + json.dumps(rows[0]) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_completed_score_rows(path)


def test_load_completed_score_rows_returns_empty_for_missing_file(tmp_path):
    assert load_completed_score_rows(tmp_path / "missing.jsonl") == {}


class _TinyTokenizer:
    pad_token_id = 0
    padding_side = "left"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = "|".join(message["content"] for message in messages)
        return text + ("|assistant:" if add_generation_prompt else "|end")

    def _ids(self, text):
        return [ord(char) % 31 + 1 for char in text]

    def __call__(self, texts, padding=False, return_tensors=None, add_special_tokens=False):
        if isinstance(texts, str):
            return {"input_ids": self._ids(texts)}
        rows = [self._ids(text) for text in texts]
        width = max(map(len, rows))
        input_ids = [[0] * (width - len(row)) + row for row in rows]
        attention_mask = [[0] * (width - len(row)) + [1] * len(row) for row in rows]
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
        }


class _NextTokenModel(torch.nn.Module):
    def forward(self, input_ids, attention_mask, logits_to_keep):
        batch, width = input_ids.shape
        start = width - logits_to_keep
        logits = torch.full((batch, logits_to_keep, 32), -5.0)
        for row in range(batch):
            for local, absolute in enumerate(range(start, width)):
                if absolute + 1 < width:
                    logits[row, local, input_ids[row, absolute + 1]] = 5.0
        return SimpleNamespace(logits=logits)


def test_score_candidate_answers_is_batch_size_invariant_and_completion_only():
    prompt = [{"role": "user", "content": "choose"}]
    answers = {"A": '{"answer":"A"}', "BB": '{"answer":"BB"}', "CCC": '{"answer":"CCC"}'}

    first = score_candidate_answers(
        _NextTokenModel(), _TinyTokenizer(), prompt, answers, batch_size=1, device="cpu"
    )
    second = score_candidate_answers(
        _NextTokenModel(), _TinyTokenizer(), prompt, answers, batch_size=3, device="cpu"
    )

    assert first == pytest.approx(second)
    assert set(first) == set(answers)
    assert all(value < 0.0 for value in first.values())


def test_mine_score_rows_resumes_completed_ids_and_records_provenance(tmp_path):
    output = tmp_path / "scores.jsonl"
    completed = {
        "source_id": "done",
        "scores": {label: -1.0 for label in REGISTRY.ids[:5]},
        "model_identity": "sft-sha",
    }
    output.write_text(json.dumps(completed) + "\n", encoding="utf-8")
    records = [
        {
            "id": "done",
            "metadata": {"field_name": "email"},
            "classification": {"level_4": "contact_email"},
        },
        {
            "id": "new",
            "metadata": {"field_name": "email_address"},
            "classification": {"level_4": "contact_email"},
        },
    ]
    calls = []

    def score_fn(prompt, answers):
        calls.append((prompt, answers))
        return {label: -float(index) for index, label in enumerate(answers)}

    report = mine_score_rows(
        records,
        REGISTRY,
        output,
        score_fn=score_fn,
        model_identity="sft-sha",
        seed=42,
    )

    assert len(calls) == 1
    assert report["existing_rows"] == 1
    assert report["new_rows"] == 1
    rows = load_completed_score_rows(output)
    assert set(rows) == {"done", "new"}
    assert rows["new"]["retrieval_policy"] == "field_registry_char_ngram_v1"
    assert rows["new"]["model_identity"] == "sft-sha"
    assert rows["new"]["real_test_split_read"] is False
    assert len(rows["new"]["scores"]) == 5


def test_adapter_identity_is_stable_and_changes_with_weights(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    config = adapter / "adapter_config.json"
    weights = adapter / "adapter_model.safetensors"
    config.write_text('{"r":16}', encoding="utf-8")
    weights.write_bytes(b"first")

    first = adapter_identity("/models/qwen", adapter)
    assert first == adapter_identity("/models/qwen", adapter)
    assert first.startswith("sha256:")

    weights.write_bytes(b"second")
    assert adapter_identity("/models/qwen", adapter) != first

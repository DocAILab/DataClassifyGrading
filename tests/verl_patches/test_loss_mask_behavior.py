"""CPU behavior guards for the patched VeRL multiturn SFT loss masks.

These tests intentionally import the installed ``verl==0.9.0`` package rather
than copying the patched implementation into the repository.  They are skipped
locally when that runtime is unavailable and should be run in the server's CPU
venv with::

    <SERVER_VENV>/bin/python -m pytest \
        tests/verl_patches/test_loss_mask_behavior.py -q

No model, tokenizer files, CUDA device, or GPU is required: the dataset test
uses a deterministic CPU-only chat-template seam and the loss test supplies a
small tensor of synthetic log probabilities.
"""

from __future__ import annotations

import importlib.metadata

import pytest


# Collection must decide whether the optional runtime exists before importing
# torch or VeRL: local checkouts intentionally do not carry either dependency.
try:
    VERL_VERSION = importlib.metadata.version("verl")
except importlib.metadata.PackageNotFoundError:
    VERL_VERSION = None

_SKIP_REASON = (
    "requires installed verl==0.9.0 (run in the CPU-capable server venv: "
    "<SERVER_VENV>/bin/python -m pytest "
    "tests/verl_patches/test_loss_mask_behavior.py -q)"
)

# Mark the whole module before touching torch/VeRL.  This keeps a checkout
# without the optional runtime collectable while still collecting test items
# (and therefore returning pytest's normal all-skipped status).
if VERL_VERSION is None:
    pytestmark = pytest.mark.skip(reason=_SKIP_REASON)
else:
    # A present-but-wrong runtime is never skippable.  Keep this check at module
    # collection so a version/API mismatch fails fast instead of hiding behind
    # a per-test skip marker.
    if VERL_VERSION != "0.9.0":
        raise RuntimeError(f"expected patched verl==0.9.0, got {VERL_VERSION!r}")
    try:
        import torch
        import verl
        import verl.utils.dataset.multiturn_sft_dataset as dataset_module
        from verl.workers.utils.losses import sft_loss
    except Exception as exc:  # pragma: no cover - exercised by broken runtimes
        raise RuntimeError(
            "installed verl==0.9.0 is missing the loss-mask dataset/loss API"
        ) from exc


class _SyntheticTokenizer:
    """Minimal tokenizer seam used to exercise answer-mask localization.

    The patched dataset only needs ``encode`` for locating answer/level value
    spans in this test; chat rendering itself is provided by the monkeypatch in
    the test below.
    """

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return {
            "A": [901],
            "L1": [902],
            '{"answer":"A","level":"L1"}': [601, 901, 602, 902, 603],
        }[text]


class _Values:
    """Small stand-in for a no-padding TensorDict nested tensor."""

    def __init__(self, values: torch.Tensor):
        self._values = values

    def values(self) -> torch.Tensor:
        return self._values


def _install_fake_template(monkeypatch, segments: dict[str, list[int]]) -> None:
    """Install a deterministic prefix-diff chat-template seam.

    Reasoning renders before content, exactly like the Qwen3.5 template; an
    empty content string renders zero tokens (the seam used by the
    empty-content boundary computation).
    """

    def fake_apply_chat_template(processor, messages, **kwargs):
        del processor, kwargs
        ids: list[int] = []
        for message in messages:
            if message.get("reasoning_content"):
                ids.extend(segments[message["reasoning_content"]])
            if message["content"]:
                ids.extend(segments[message["content"]])
        input_ids = torch.tensor([ids], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    monkeypatch.setattr(dataset_module, "apply_chat_template", fake_apply_chat_template)


def _make_dataset(tokenizer) -> "MultiTurnSFTDataset":
    """Dataset instance without touching parquet/model assets."""

    dataset = dataset_module.MultiTurnSFTDataset.__new__(dataset_module.MultiTurnSFTDataset)
    dataset.tokenizer = tokenizer
    dataset.processor = tokenizer
    dataset.apply_chat_template_kwargs = {}
    dataset.generation_prompt = []
    return dataset


def _build_dataset_for_mask_test(dataset_module, monkeypatch, *, collision: bool = False):
    """Construct a dataset instance without touching parquet/model assets."""

    tokenizer = _SyntheticTokenizer()
    segments = {
        "user question": [101, 102],
        "<think>plan one</think>": [201, 202],
        "<tool_call>search</tool_call>": [301, 302],
        "search result": [401, 402],
        "<think>plan two</think>": [501, 502],
        "terminal reasoning": [701, 901, 702, 902, 703],
        '{"answer":"A","level":"L1"}': [601, 901, 602, 902, 603],
    }
    _install_fake_template(monkeypatch, segments)
    dataset = _make_dataset(tokenizer)

    messages = [
        {"role": "user", "content": "user question"},
        {"role": "assistant", "content": "<think>plan one</think>"},
        {"role": "assistant", "content": "<tool_call>search</tool_call>"},
        {"role": "tool", "content": "search result"},
        {"role": "assistant", "content": "<think>plan two</think>"},
        {
            "role": "assistant",
            "content": '{"answer":"A","level":"L1"}',
            **({"reasoning_content": "terminal reasoning"} if collision else {}),
        },
    ]
    return dataset, messages


def test_multiturn_role_loss_mask_matrix_and_answer_spans(monkeypatch) -> None:
    """Lock the six-role mask matrix and answer/level-only answer_mask spans."""

    dataset, messages = _build_dataset_for_mask_test(dataset_module, monkeypatch)
    expected = [
        ("user", 0),
        ("assistant think", 1),
        ("assistant tool_call", 1),
        ("tool response", 0),
        ("assistant think", 1),
        ("assistant final", 1),
    ]

    observed = []
    for index, message in enumerate(messages):
        result = dataset._process_single_message(
            index=index,
            message=message,
            full_message=messages,
            tools=[{"type": "function", "function": {"name": "search"}}] if index == 0 else None,
            enable_thinking=True,
        )
        assert len(result) == 5, "patched dataset must return answer_mask"
        input_ids, loss_mask, answer_mask, attention_mask, _ = result
        assert input_ids.shape == loss_mask.shape == answer_mask.shape == attention_mask.shape
        observed.append((input_ids, loss_mask, answer_mask))

    for (label, expected_value), (input_ids, loss_mask, answer_mask) in zip(expected, observed):
        del input_ids
        assert loss_mask.tolist() == [expected_value] * len(loss_mask), label

    # The terminal JSON has framework/key tokens at 601/602/603 and only the
    # answer/level values at 901/902.  Neither keys, punctuation, nor any prior
    # turn may receive the 8x answer weighting channel.
    assert observed[-1][2].tolist() == [0, 1, 0, 1, 0]
    for _, _, answer_mask in observed[:-1]:
        assert not answer_mask.any().item()


def test_answer_mask_anchors_terminal_content_after_reasoning_collision(monkeypatch) -> None:
    """Identical opaque values in think must not receive answer weighting."""

    dataset, messages = _build_dataset_for_mask_test(dataset_module, monkeypatch, collision=True)
    input_ids, loss_mask, answer_mask, attention_mask, _ = dataset._process_single_message(
        index=len(messages) - 1,
        message=messages[-1],
        full_message=messages,
        enable_thinking=True,
    )

    assert input_ids.tolist() == [701, 901, 702, 902, 703, 601, 901, 602, 902, 603]
    assert loss_mask.tolist() == [1] * 10
    assert attention_mask.tolist() == [1] * 10
    assert answer_mask.tolist() == [0, 0, 0, 0, 0, 0, 1, 0, 1, 0]


def test_answer_mask_identical_json_in_reasoning_stays_1x(monkeypatch) -> None:
    """A verbatim JSON repeated inside think must stay 1x; only the terminal
    content region receives the 8x channel (empty-render boundary)."""

    tokenizer = _SyntheticTokenizer()
    content = '{"answer":"A","level":"L1"}'
    reasoning = 'I considered {"answer":"A","level":"L1"} before the final response.'
    _install_fake_template(
        monkeypatch,
        {
            "user question": [101, 102],
            reasoning: [801, 802, 601, 901, 602, 902, 603, 803, 804],
            content: [601, 901, 602, 902, 603],
        },
    )
    dataset = _make_dataset(tokenizer)
    messages = [
        {"role": "user", "content": "user question"},
        {"role": "assistant", "content": content, "reasoning_content": reasoning},
    ]
    input_ids, loss_mask, answer_mask, attention_mask, _ = dataset._process_single_message(
        index=1,
        message=messages[1],
        full_message=messages,
        enable_thinking=True,
    )

    expected = [801, 802, 601, 901, 602, 902, 603, 803, 804, 601, 901, 602, 902, 603]
    assert input_ids.tolist() == expected
    assert loss_mask.tolist() == [1] * len(expected)
    assert attention_mask.tolist() == [1] * len(expected)
    # Reasoning (0..8) keeps the 1x loss channel; terminal content value spans
    # (10, 12) are the only 8x-eligible positions.
    assert answer_mask.tolist() == [0] * 9 + [0, 1, 0, 1, 0]


def test_answer_mask_ambiguous_identical_delta_fails_closed(monkeypatch) -> None:
    """Think byte-identical to content makes the boundary unprovable: the mask
    fails closed to all zeros instead of guessing."""

    tokenizer = _SyntheticTokenizer()
    content = '{"answer":"A","level":"L1"}'
    _install_fake_template(monkeypatch, {content: [601, 901, 602, 902, 603]})
    dataset = _make_dataset(tokenizer)
    messages = [{"role": "assistant", "content": content, "reasoning_content": content}]
    input_ids, loss_mask, answer_mask, attention_mask, _ = dataset._process_single_message(
        index=0,
        message=messages[0],
        full_message=messages,
        enable_thinking=True,
    )

    assert input_ids.tolist() == [601, 901, 602, 902, 603, 601, 901, 602, 902, 603]
    assert loss_mask.tolist() == [1] * 10
    assert attention_mask.tolist() == [1] * 10
    assert answer_mask.tolist() == [0] * 10


def test_answer_mask_truncated_terminal_value_fails_closed() -> None:
    """Incomplete or absent terminal content must never produce partial marks."""

    tokenizer = _SyntheticTokenizer()
    content = '{"answer":"A","level":"L1"}'
    # Delta truncated mid-content: the complete content encoding is absent.
    assert dataset_module._answer_mask_positions(
        tokenizer,
        torch.tensor([[701, 901, 702, 902, 703, 601, 901, 602]]),
        content,
        empty_input_ids=torch.tensor([[701, 901, 702, 902, 703]]),
    ) == []
    # Reasoning-only delta: the content vanished entirely.
    assert dataset_module._answer_mask_positions(
        tokenizer,
        torch.tensor([[701, 901, 702, 902, 703]]),
        content,
        empty_input_ids=torch.tensor([[701, 901, 702, 902, 703]]),
    ) == []
    # No empty render available and no unique full-content occurrence.
    assert dataset_module._answer_mask_positions(
        tokenizer,
        torch.tensor([[701, 901, 702, 902, 703, 601, 901, 602]]),
        content,
    ) == []
    # Content cut off mid-value cannot even be encoded: fails closed.
    assert dataset_module._answer_mask_positions(
        tokenizer, torch.tensor([[701, 901, 702, 902, 703]]), '{"answer":"A","level":"L1'
    ) == []


def test_sft_loss_uses_scheme_c_answer_mask_weight_once() -> None:
    """Answer spans are 8x while all other loss-masked tokens remain 1x."""

    # The loss implementation rolls masks left to align each input position
    # with the next-token log probability.  With an all-one loss mask and two
    # answer positions, the weighted numerator is 8*2 + 1*3 = 19.
    log_probs = _Values(torch.full((5,), -1.0, dtype=torch.float32))
    data = {
        "pad_mode": "no_padding",
        "dp_size": 1,
        "batch_num_tokens": 5,
        "loss_mask": _Values(torch.ones(5, dtype=torch.float32)),
        "answer_mask": _Values(torch.tensor([0, 1, 0, 1, 0], dtype=torch.float32)),
    }

    loss, metrics = sft_loss(None, {"log_probs": log_probs}, data)

    assert metrics == {}
    torch.testing.assert_close(loss, torch.tensor(19.0 / 5.0))


def test_sft_loss_answer_mask_does_not_double_count_base_weight() -> None:
    """The 8x answer branch substitutes (rather than adds to) the 1x branch."""

    log_probs = _Values(torch.full((3,), -1.0, dtype=torch.float32))
    data = {
        "pad_mode": "no_padding",
        "dp_size": 1,
        "batch_num_tokens": 3,
        "loss_mask": _Values(torch.ones(3, dtype=torch.float32)),
        "answer_mask": _Values(torch.tensor([1, 0, 0], dtype=torch.float32)),
    }

    loss, _ = sft_loss(None, {"log_probs": log_probs}, data)

    # After the implementation's left shift, one answer position is weighted
    # 8x and the remaining two positions 1x: (8 + 1 + 1) / 3, not 10 / 3.
    torch.testing.assert_close(loss, torch.tensor(10.0 / 3.0))

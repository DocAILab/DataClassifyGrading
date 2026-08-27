"""Deterministic tool-trajectory SFT dataset generator and validator.

The RLOO tool-loop consumes full chat trajectories, so the SFT baseline for
it must teach the same behavior: complete ``messages`` spanning system/user,
assistant tool calls, real tool results, and one strict terminal assistant
JSON object. This module generates those trajectories programmatically and
deterministically from canonical records plus the REAL registry/corpus, with
no shortcut answers injected into tool results.

The pipeline is shougang-only: the dataset argument must equal
``FORMAL_RELEASE_NAME`` (the formal singleton release contract; finance does
not participate in the trajectory data line).

Trajectory classes (one deterministic class per canonical record):

- ``direct``: no tools; the assistant answers with the strict terminal JSON
  immediately after the user turn.
- ``single_tool``: one ``search_categories`` call, real result, terminal JSON.
- ``multi_tool``: ``search_categories`` -> ``get_category_details`` ->
  ``get_category_examples``, real results, terminal JSON (three different
  tools, within the formal ``max_assistant_turns=4`` cap).
- ``no_result``: ``browse_categories`` + a scoped search in a level-1 group
  that does NOT contain the ground truth; the model must still emit the
  terminal JSON even though tools never surfaced the answer.

The terminal assistant message also carries a ``reasoning_content`` key
(thinking text) produced by a pluggable think generator. The default is the
local deterministic :class:`MockThinkGenerator` (no credentials, no network);
a real deepseek-v4-flash-backed generator plugs into the same protocol later
(runtime-local credentials only). Generated think text is enforced against
``max_think_tokens`` (default 128, coordinator decision): over-limit text is
truncated to the prefix within budget or the row is discarded, per
``think_over_limit``. Loss scope is pinned by decision: think enters the SFT
loss at weight 1 and the answer/level value spans at weight 8 (the server
scheme-C answer_mask patch is the contract; the patch is not changed).

Leakage discipline (audited by :func:`validate_tool_trajectory_dataset`):

- tool results are byte-exact outputs of ``CategoryToolEnvironment`` calls
  re-executed from the recorded arguments (inference-time content only);
- tool-call arguments come only from the record metadata, ids returned by
  earlier tool results, and scope keys returned by browse;
- the system/user prompt and the reasoning_content never contain the
  canonical ground-truth id and never frame the answer as "this is the
  answer";
- the strict terminal JSON is the ONLY supervised label (the training loss
  masks user/tool messages per the existing messages contract); it must parse
  with ``parse_final_tool_answer`` and match ground_truth + level.

Split boundaries come from canonical schema v2 embedded ``split`` fields.
The exporter contract (label-gap gate, per-split parquet + sha256 report,
atomic publication) mirrors ``script.verl.sft.export``; grading is REQUIRED
because the terminal JSON carries a sensitivity level.

Metadata fields are parameterized through ``TaskConfig.metadata_fields``
(explicit ``--metadata-fields``). NOTE: the live RLOO prompt seam
(``agent.training.rl.sample.build_native_tool_prompt``) currently requires
exactly the four native fields; when the configured fields equal that set,
this module's prompt is byte-identical to the RLOO seam. The four-vs-two
field contract is owner-blocked — see the phase design note.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Protocol, Sequence

from agent.hashing import sha256_file
from agent.release_policy import FORMAL_RELEASE_NAME
from agent.task.contracts import CorpusCategory, GradingConfig, LeafRegistry, TaskConfig
from agent.task.prompt_choices import PromptChoiceRegistry
from agent.task.prompts import Prompt
from agent.training.common import canonical_target
from agent.training.rl.native_tools import (
    CategoryToolEnvironment,
    parse_final_tool_answer,
)
from agent.training.rl.sample import (
    NATIVE_TOOL_TRAJECTORY_FORMAT,
    PROMPT_METADATA_FIELDS,
    render_catalog_index,
    visible_metadata,
)
from agent.training.sft.dataset import _write_parquet

SPLITS = ("train", "val", "test")
TRAJECTORY_CLASSES = ("direct", "single_tool", "multi_tool", "no_result")
_STAGE = "tool_trajectory"
# Expected number of tool calls per class (direct has none; multi_tool calls
# three different tools; no_result calls browse + scoped search).
_EXPECTED_TOOL_CALLS = {
    "direct": 0,
    "single_tool": 1,
    "multi_tool": 3,
    "no_result": 2,
}
_TOOL_TOP_K = 5
_MAX_DETAIL_IDS = 2
_MAX_EXAMPLE_IDS = 1
_EXAMPLE_LIMIT = 2
# Default ceiling for generated think text: 128 tokens (coordinator
# decision 2026-08-27; the SFT line is terminal-think only). RL-stage budgets
# are 64 per tool turn / 128 terminal (design note). Mock think (~30 tokens)
# is unaffected. Loss scope is pinned: think enters the SFT loss at weight 1,
# answer/level value spans at weight 8 (server scheme-C patch is the
# contract; the patch itself is not changed).
_DEFAULT_MAX_THINK_TOKENS = 128
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")


class ThinkGenerator(Protocol):
    """Pluggable think-text generator for the terminal assistant turn.

    Implementations must be deterministic for identical inputs when used in
    reproducible exports, and must never receive or emit ground-truth
    content. The default :class:`MockThinkGenerator` is local and credential-
    free; a deepseek-v4-flash-backed implementation plugs in here with
    runtime-local credentials only.
    """

    name: str

    def generate(
        self,
        *,
        source_id: str,
        metadata: Mapping[str, str],
        trajectory_class: str,
    ) -> str: ...


class MockThinkGenerator:
    """Deterministic local think generator (no credentials, no network).

    Emits short reasoning text anchored to inference-time content only
    (metadata and trajectory class); it never contains canonical ids or the
    ground truth, and its output is stable across runs.
    """

    name = "mock"

    def generate(
        self,
        *,
        source_id: str,
        metadata: Mapping[str, str],
        trajectory_class: str,
    ) -> str:
        field = str(metadata.get("field_name", "") or "the field")
        table = str(metadata.get("table_name", "") or "its table")
        return (
            f"I inspect the catalog for field {field!r} in table {table!r}; "
            f"for the {trajectory_class} trajectory I weigh candidate "
            "definitions and examples from the catalog, then emit the "
            "terminal JSON with one opaque choice id and the approved level."
        )


class FileThinkGenerator:
    """Read pre-generated think text from JSONL shards by sample id.

    Backs the ``--think-source file:<path>`` seam: the collect step writes
    one JSONL context shard per (split, shard) with an empty ``think`` field;
    a label-aware sub-agent (e.g. deepseek-v4-flash) fills ``think`` in its
    own reserved shard file only, and export/assemble reads every shard from
    ``path`` (a single file or a directory of ``*.jsonl`` shards) keyed by
    ``sample_id``. Missing samples fail fast so a partial shard set can never
    silently produce a release.
    """

    name = "file"

    def __init__(self, path: str | Path) -> None:
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(f"think shard source not found: {source}")
        shard_files = (
            sorted(source.glob("*.jsonl")) if source.is_dir() else [source]
        )
        if not shard_files:
            raise ValueError(f"think shard directory contains no *.jsonl: {source}")
        self._think: dict[str, str] = {}
        self._shard_files = [str(path) for path in shard_files]
        for shard in shard_files:
            for line_number, line in enumerate(
                shard.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                except (json.JSONDecodeError, TypeError) as exc:
                    raise ValueError(
                        f"invalid JSON in think shard {shard} line {line_number}: {exc}"
                    ) from exc
                sample_id = entry.get("sample_id") if isinstance(entry, Mapping) else None
                think = entry.get("think") if isinstance(entry, Mapping) else None
                if not isinstance(sample_id, str) or not sample_id.strip():
                    raise ValueError(
                        f"think shard {shard} line {line_number} lacks sample_id"
                    )
                if not isinstance(think, str):
                    raise ValueError(
                        f"think shard {shard} line {line_number} lacks a string think"
                    )
                if sample_id in self._think:
                    raise ValueError(
                        f"duplicate sample_id {sample_id!r} across think shards"
                    )
                self._think[sample_id] = think

    def generate(
        self,
        *,
        source_id: str,
        metadata: Mapping[str, str],
        trajectory_class: str,
    ) -> str:
        if source_id not in self._think:
            raise ValueError(
                f"think shards ({len(self._shard_files)} file(s)) are missing "
                f"sample_id {source_id!r}; assemble requires every record"
            )
        return self._think[source_id]


def estimate_think_tokens(text: str, tokenizer=None) -> int:
    """Estimate the token count of one think text.

    With a real tokenizer the encoded length is used. Without one (local CPU
    fixtures) a conservative character approximation is applied: every CJK
    character counts as one token and every four ASCII characters as one.
    The approximation is documented as conservative; the formal ceiling
    check runs with the qwen3.5 tokenizer on the training host.
    """

    if not isinstance(text, str) or not text.strip():
        return 0
    if tokenizer is not None:
        encoded = tokenizer.encode(text)
        if hasattr(encoded, "numel"):
            return int(encoded.numel())
        return len(encoded)
    cjk = len(_CJK_RE.findall(text))
    other = len(text) - cjk
    return cjk + (other + 3) // 4


def _think_closed(text: str) -> bool:
    """Return whether every bracket opened in think text is balanced.

    The label-aware file source may quote structured fragments; unbalanced
    brackets mean the think text is an unfinished fragment (or a truncated
    JSON lookalike) and must be rejected.
    """

    return text.count("{") == text.count("}") and text.count("[") == text.count("]")


_THINK_KEY_LOOKALIKE = re.compile(r'"(?:answer|level)"\s*:')
_THINK_TERMINAL_JSON = re.compile(r'\{\s*"answer"\s*:')


def _truncate_think_to_budget(text: str, max_tokens: int, tokenizer=None) -> str:
    """Return the longest prefix whose estimated token count fits the budget.

    The prefix is backed off until brackets balance, so a truncated think
    never leaves an unfinished JSON/fragment structure open (closedness is a
    public audit constraint for every think source).
    """

    if max_tokens <= 0:
        return ""
    budget = max_tokens * 4
    kept: list[str] = []
    used = 0
    for character in text:
        cost = 4 if _CJK_RE.match(character) else 1
        if used + cost > budget:
            break
        kept.append(character)
        used += cost
    result = "".join(kept)
    if tokenizer is not None:
        # Re-check with the exact tokenizer and shrink the tail if needed.
        while result and estimate_think_tokens(result, tokenizer) > max_tokens:
            result = result[:-1]
    while result and not _think_closed(result):
        result = result[:-1]
    return result
# System text mirrors agent.training.rl.sample.build_native_tool_prompt so
# that, with the native four metadata fields configured, prompt bytes are
# identical to the live RLOO raw_prompt seam. Guarded by a regression test.
_TOOL_SYSTEM_TEMPLATE = (
    "You classify one database field into one category and assign a sensitivity level.\n"
    "The full category catalog is listed below as \"choice_id|name\" lines; recall "
    "candidates from this catalog directly using field and table semantics. When "
    "candidate names alone are ambiguous, call get_category_details or "
    "get_category_examples on the candidate ids to inspect definitions and samples. "
    "If the catalog seems insufficient you may call search_categories(field_name, "
    "table_name) once as a fallback hint. Make at most three tool calls total.\n"
    "Your terminal response must be exactly one JSON object with keys answer and "
    "level. answer must be an opaque choice_id from the catalog, never a category "
    "name or canonical id. level must be one approved sensitivity code. Do not "
    "output reasoning, Markdown, or extra keys.\n"
    "Approved sensitivity levels:\n"
)


def select_trajectory_class(source_id: str) -> str:
    """Return the deterministic trajectory class for one stable record id."""

    if not isinstance(source_id, str) or not source_id.strip():
        raise ValueError("trajectory class selection requires a stable source id")
    digest = hashlib.sha256(
        f"tool-trajectory:{source_id}".encode("utf-8")
    ).hexdigest()
    return TRAJECTORY_CLASSES[int(digest, 16) % len(TRAJECTORY_CLASSES)]


def build_tool_trajectory_prompt(
    metadata: Mapping[str, str],
    grading: GradingConfig,
    registry: LeafRegistry,
    config: TaskConfig,
) -> Prompt:
    """Build the catalog-in-context tool-loop prompt with configured fields.

    With ``config.metadata_fields`` equal to the native four-field set the
    bytes are identical to ``build_native_tool_prompt`` (the live RLOO seam);
    any other field set renders the same system text with the configured
    fields in the user turn (owner-blocked four-vs-two resolution deferred).
    """

    rubric = [[code, description] for code, description in grading.rubric()]
    system = (
        _TOOL_SYSTEM_TEMPLATE
        + json.dumps(rubric, ensure_ascii=False, separators=(",", ":"))
        + "\nCatalog (choice_id|name):\n"
        + render_catalog_index(registry)
    )
    ordered = {field: str(metadata.get(field, "")) for field in config.metadata_fields}
    user = (
        "Classify this field metadata:\n"
        + json.dumps(ordered, ensure_ascii=False, separators=(",", ":"))
    )
    return Prompt(system=system, user=user)


def render_tool_call(name: str, arguments: Mapping[str, Any]) -> str:
    """Render one qwen3_coder tool call.

    String arguments are emitted raw, list arguments as compact JSON, in the
    caller-supplied order. Format matches the live verl qwen3_coder parser:
    ``<tool_call><function=NAME><parameter=KEY>VALUE</parameter></function></tool_call>``.
    """

    if not isinstance(name, str) or not name.strip():
        raise ValueError("tool call requires a non-empty function name")
    if not isinstance(arguments, Mapping):
        raise ValueError("tool call arguments must be an object")
    parts = []
    for key, value in arguments.items():
        if isinstance(value, str):
            rendered = value
        elif isinstance(value, (list, tuple)):
            rendered = json.dumps(list(value), ensure_ascii=False, separators=(",", ":"))
        elif isinstance(value, (int, float, bool)) or value is None:
            rendered = json.dumps(value)
        else:
            raise ValueError(f"unsupported tool argument type for {key!r}")
        parts.append(f"<parameter={key}>{rendered}</parameter>")
    return f"<tool_call><function={name}" + "".join(parts) + "</function></tool_call>"


def _scope_key(category: Any) -> str:
    """Return the stable level-1 group key for one registry category.

    Mirrors ``agent.training.rl.native_tools._level1_key`` over public
    registry data (explicit path head, else the leading alphabetic code of
    the canonical id head). Kept local so the generator needs no label
    access beyond public registry fields.
    """

    path = getattr(category, "path", ()) or ()
    if path:
        return str(path[0])
    cid = str(category.category_id)
    head = cid.split(":", 1)[-1]
    for ch in head:
        if ch.isalpha():
            return ch.upper()
    return head[:1]


def _terminal_json(gt_choice_id: str, level: str) -> str:
    return json.dumps({"answer": gt_choice_id, "level": level}, ensure_ascii=False)


def _tool_message(result: Mapping[str, Any]) -> dict[str, str]:
    return {
        "role": "tool",
        "content": json.dumps(result, ensure_ascii=False),
    }


def _assistant_message(content: str) -> dict[str, str]:
    return {"role": "assistant", "content": content}


def _result_choice_ids(result: Mapping[str, Any]) -> list[str]:
    """Collect every choice id a tool result surfaces, in stable result order.

    List (not set) order is required: the generator derives later tool-call
    arguments from this order, and cross-run byte determinism must not depend
    on hash randomization.
    """

    ids: list[str] = []
    seen: set[str] = set()
    for bucket in ("candidates", "categories", "leaves"):
        for entry in result.get(bucket, []) or []:
            choice_id = entry.get("choice_id") if isinstance(entry, Mapping) else None
            if isinstance(choice_id, str) and choice_id.strip() and choice_id not in seen:
                seen.add(choice_id)
                ids.append(choice_id)
    return ids


def _build_trajectory_context(
    item: Mapping[str, Any],
    index: int,
    source: Path,
    *,
    registry: LeafRegistry,
    corpus: Mapping[str, CorpusCategory],
    config: TaskConfig,
    grading: GradingConfig,
    env: CategoryToolEnvironment,
    dataset: str = FORMAL_RELEASE_NAME,
) -> dict[str, Any]:
    """Build the deterministic think-free trajectory context for one record.

    Shared by :func:`build_trajectory` (think injection) and the ``collect``
    step (label-aware shard export): every field except the think text is
    derived here exactly once, so collect contexts and assembled rows always
    agree byte-for-byte on prompts, tool calls, and results.
    """

    if dataset != FORMAL_RELEASE_NAME:
        raise ValueError(
            "tool-trajectory SFT dataset must be exactly "
            f"{FORMAL_RELEASE_NAME} (shougang-only line)"
        )
    ground_truth = canonical_target(item, index, source, registry)
    if ground_truth is None:
        raise ValueError(f"item {index} in {source} is not resolved")
    source_id = str(item.get("id", "") or "").strip()
    if not source_id:
        raise ValueError(f"item {index} in {source} has no stable id")
    raw_level = item.get(grading.gt_field, "")
    level = "" if raw_level is None else str(raw_level).strip()
    if not level:
        raise ValueError(
            f"record {source_id!r} has no grading label under {grading.gt_field!r}"
        )
    if level not in grading.levels:
        raise ValueError(
            f"record {source_id!r} has grading label {level!r} outside "
            f"configured levels {list(grading.levels)}"
        )
    metadata = visible_metadata(item.get("metadata", {}), config)
    prompt = build_tool_trajectory_prompt(metadata, grading, registry, config)
    gt_choice_id = env.choices.choice_id_of(ground_truth)
    terminal = _terminal_json(gt_choice_id, level)
    terminal_object = {"answer": gt_choice_id, "level": level}
    trajectory_class = select_trajectory_class(source_id)
    messages: list[dict[str, str]] = [
        {"role": "system", "content": prompt.system},
        {"role": "user", "content": prompt.user},
    ]
    context: dict[str, Any] = {
        "source_id": source_id,
        "dataset": dataset,
        "trajectory_class": trajectory_class,
        "prompt": {"system": prompt.system, "user": prompt.user},
        "metadata": metadata,
        "ground_truth": ground_truth,
        "ground_truth_level": level,
        "terminal_json": terminal_object,
    }

    if trajectory_class == "direct":
        messages.append({"role": "assistant", "content": terminal})
        context["messages"] = messages
        context["tool_calls"] = []
        return context

    field_name = str(metadata.get("field_name", "") or "")
    table_name = str(metadata.get("table_name", "") or "")
    if not field_name:
        raise ValueError(
            f"record {source_id!r} requires a non-empty metadata field_name "
            f"for {trajectory_class} trajectory"
        )
    if trajectory_class == "single_tool":
        plan = [
            ("search_categories", {"field_name": field_name, "table_name": table_name}),
        ]
    elif trajectory_class == "multi_tool":
        plan = [
            ("search_categories", {"field_name": field_name, "table_name": table_name}),
            ("get_category_details", {"choice_ids": []}),
            ("get_category_examples", {"choice_ids": [], "limit": _EXAMPLE_LIMIT}),
        ]
    elif trajectory_class == "no_result":
        plan = [
            ("browse_categories", {}),
            ("search_categories", {"field_name": field_name, "table_name": table_name, "scope": ""}),
        ]
    else:  # pragma: no cover - guarded by select_trajectory_class
        raise ValueError(f"unknown trajectory class {trajectory_class!r}")

    tool_calls: list[dict[str, Any]] = []
    prior_choice_ids: list[str] = []
    browse_groups: list[str] = []
    for name, raw_arguments in plan:
        arguments = dict(raw_arguments)
        if name == "search_categories":
            scope = arguments.get("scope")
            if scope is not None and trajectory_class == "no_result":
                # Deterministic pick: the smallest level-1 scope key that does
                # NOT contain the ground truth, so the tool result cannot
                # surface the answer. Requires >= 2 level-1 groups.
                gt_group = _scope_key(registry.get(ground_truth))
                other_groups = sorted(group for group in browse_groups if group != gt_group)
                if not other_groups:
                    raise ValueError(
                        f"record {source_id!r} no_result trajectory requires a "
                        "registry with at least two level-1 groups; single-group "
                        "registries cannot guarantee a ground-truth-free tool result"
                    )
                arguments["scope"] = other_groups[0]
                result = env.search_categories(
                    str(arguments["field_name"]),
                    str(arguments.get("table_name", "")),
                    scope=arguments["scope"],
                )
            else:
                result = env.search_categories(
                    str(arguments["field_name"]),
                    str(arguments.get("table_name", "")),
                )
        elif name == "get_category_details":
            if not prior_choice_ids:
                raise ValueError(
                    f"record {source_id!r} details call requires search results"
                )
            arguments["choice_ids"] = prior_choice_ids[:_MAX_DETAIL_IDS]
            result = env.get_category_details(arguments["choice_ids"])
        elif name == "get_category_examples":
            if not prior_choice_ids:
                raise ValueError(
                    f"record {source_id!r} examples call requires search results"
                )
            arguments["choice_ids"] = prior_choice_ids[:_MAX_EXAMPLE_IDS]
            result = env.get_category_examples(
                arguments["choice_ids"], limit=arguments["limit"]
            )
        elif name == "browse_categories":
            result = env.browse_categories()
            browse_groups = [
                str(group["scope_key"]) for group in result.get("groups", [])
            ]
        else:  # pragma: no cover - guarded by the class plans above
            raise ValueError(f"unexpected tool call {name!r}")
        tool_calls.append({"name": name, "arguments": arguments})
        for choice_id in _result_choice_ids(result):
            if choice_id not in prior_choice_ids:
                prior_choice_ids.append(choice_id)
        messages.append(_assistant_message(render_tool_call(name, arguments)))
        messages.append(_tool_message(result))

    messages.append({"role": "assistant", "content": terminal})
    context["messages"] = messages
    context["tool_calls"] = tool_calls
    return context


def build_trajectory(
    item: Mapping[str, Any],
    index: int,
    source: Path,
    *,
    registry: LeafRegistry,
    corpus: Mapping[str, CorpusCategory],
    config: TaskConfig,
    grading: GradingConfig,
    env: CategoryToolEnvironment,
    dataset: str = FORMAL_RELEASE_NAME,
    think_generator: ThinkGenerator | None = None,
    max_think_tokens: int = _DEFAULT_MAX_THINK_TOKENS,
    think_over_limit: str = "truncate",
    think_tokenizer=None,
) -> dict[str, Any] | None:
    """Build one deterministic tool-trajectory row for a resolved record.

    The trajectory context (prompt/tool calls/results/terminal JSON) comes
    from :func:`_build_trajectory_context`; think text is injected into the
    terminal assistant message with the over-limit policy applied. Returns
    None when ``think_over_limit == "discard"`` and the think text exceeds
    ``max_think_tokens`` (the caller counts the skip).
    """

    if dataset != FORMAL_RELEASE_NAME:
        raise ValueError(
            "tool-trajectory SFT dataset must be exactly "
            f"{FORMAL_RELEASE_NAME} (shougang-only line)"
        )
    if think_generator is None:
        think_generator = MockThinkGenerator()
    if think_over_limit not in {"truncate", "discard"}:
        raise ValueError("think_over_limit must be truncate or discard")
    if max_think_tokens <= 0:
        raise ValueError("max_think_tokens must be positive")
    context = _build_trajectory_context(
        item,
        index,
        source,
        registry=registry,
        corpus=corpus,
        config=config,
        grading=grading,
        env=env,
        dataset=dataset,
    )
    think_text = think_generator.generate(
        source_id=context["source_id"],
        metadata=context["metadata"],
        trajectory_class=context["trajectory_class"],
    )
    think_truncated = False
    if estimate_think_tokens(think_text, think_tokenizer) > max_think_tokens:
        if think_over_limit == "discard":
            return None
        think_text = _truncate_think_to_budget(
            think_text, max_think_tokens, think_tokenizer
        )
        if not think_text.strip():
            raise ValueError(
                f"record {context['source_id']!r} think truncation produced an "
                "empty reasoning_content; no closed prefix fits the budget"
            )
        think_truncated = True
    context["messages"][-1]["reasoning_content"] = think_text
    return {
        "messages": context["messages"],
        "stage": _STAGE,
        "trajectory_class": context["trajectory_class"],
        "trajectory_format": NATIVE_TOOL_TRAJECTORY_FORMAT,
        "source_id": context["source_id"],
        "dataset": context["dataset"],
        "metadata": context["metadata"],
        "ground_truth": context["ground_truth"],
        "ground_truth_level": context["ground_truth_level"],
        "tool_calls": context["tool_calls"],
        "think_source": think_generator.name,
        "think_truncated": think_truncated,
    }



def load_json_records(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"input canonical file not found: {source}")
    with source.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list):
        raise ValueError(f"canonical dataset must be a JSON array: {source}")
    if not all(isinstance(item, dict) for item in value):
        raise ValueError(f"canonical dataset items must be objects: {source}")
    return value


def _registry(value: LeafRegistry | str | Path) -> LeafRegistry:
    return value if isinstance(value, LeafRegistry) else LeafRegistry.from_path(value)


def _config(value: TaskConfig | str | Path) -> TaskConfig:
    return value if isinstance(value, TaskConfig) else TaskConfig.from_path(value)


def _corpus_map(corpus: Mapping[str, CorpusCategory]) -> Mapping[str, CorpusCategory]:
    if not isinstance(corpus, Mapping) or not corpus:
        raise ValueError("tool trajectories require a canonical corpus mapping")
    return corpus


def collect_tool_trajectory_contexts(
    canonical_file: str | Path,
    collect_dir: str | Path,
    registry: LeafRegistry | str | Path,
    corpus: Mapping[str, CorpusCategory],
    task_config: TaskConfig | str | Path,
    grading: GradingConfig,
    *,
    dataset: str = FORMAL_RELEASE_NAME,
    shard_size: int = 64,
) -> dict[str, Any]:
    """Export think-free trajectory contexts as JSONL shards for a label-aware
    sub-agent (e.g. deepseek-v4-flash) to fill ``think`` in.

    Every shard line carries the deterministic trajectory context (prompt,
    tool calls, tool results, ground truth, terminal JSON) with an empty
    ``think`` field; the sub-agent writes only its own reserved shard file.
    ``export_tool_trajectory_dataset`` with ``think_source=file:<collect_dir>``
    is the assemble path: it re-derives the same contexts, reads the filled
    think by ``sample_id``, and validates the merged release.
    """

    if dataset != FORMAL_RELEASE_NAME:
        raise ValueError(
            "tool-trajectory SFT dataset must be exactly "
            f"{FORMAL_RELEASE_NAME} (shougang-only line)"
        )
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    leaf_registry = _registry(registry)
    config = _config(task_config)
    canonical_path = Path(canonical_file)
    output_root = Path(collect_dir)
    if not canonical_path.is_file():
        raise FileNotFoundError(f"canonical dataset not found: {canonical_path}")
    canonical_records = load_json_records(canonical_path)
    corpus_map = _corpus_map(corpus)
    env = CategoryToolEnvironment(leaf_registry, corpus_map)

    shard_rows: dict[str, list[dict[str, Any]]] = {name: [] for name in SPLITS}
    for index, item in enumerate(canonical_records):
        status = str(item.get("resolution_status", "") or "").strip()
        split = str(item.get("split", "") or "").strip()
        if status != "resolved" or split not in SPLITS:
            continue
        raw_level = item.get(grading.gt_field)
        if raw_level is None or (isinstance(raw_level, str) and not raw_level.strip()):
            continue
        context = _build_trajectory_context(
            item,
            index,
            canonical_path,
            registry=leaf_registry,
            corpus=corpus_map,
            config=config,
            grading=grading,
            env=env,
            dataset=dataset,
        )
        shard_rows[split].append(
            {
                "sample_id": context["source_id"],
                "split": split,
                "trajectory_class": context["trajectory_class"],
                "prompt": context["prompt"],
                "tool_calls": context["tool_calls"],
                "tool_results": [
                    message["content"]
                    for message in context["messages"]
                    if message["role"] == "tool"
                ],
                "ground_truth": context["ground_truth"],
                "ground_truth_level": context["ground_truth_level"],
                "terminal_json": context["terminal_json"],
                "think": "",
            }
        )

    output_root.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "format": "verl_tool_trajectory_think_collect_v1",
        "dataset": dataset,
        "label_aware": True,
        "instruction": (
            "fill the think field with reasoning consistent with the given "
            "terminal_json/ground_truth; never emit a terminal JSON object or "
            "\"answer\"/\"level\" key-value lookalikes inside think"
        ),
        "shard_size": shard_size,
        "shards": [],
        "splits": {},
    }
    total = 0
    for split in SPLITS:
        rows = shard_rows[split]
        report["splits"][split] = len(rows)
        total += len(rows)
        for shard_index in range(0, len(rows), shard_size):
            shard_rows_for_index = rows[shard_index : shard_index + shard_size]
            shard_path = output_root / f"think-{split}-{shard_index // shard_size:03d}.jsonl"
            with shard_path.open("w", encoding="utf-8", newline="\n") as handle:
                for entry in shard_rows_for_index:
                    handle.write(
                        json.dumps(entry, ensure_ascii=False) + "\n"
                    )
            report["shards"].append(
                {
                    "path": str(shard_path),
                    "rows": len(shard_rows_for_index),
                    "sha256": sha256_file(shard_path),
                }
            )
    report["total_samples"] = total
    (output_root / "collect_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def export_tool_trajectory_dataset(
    canonical_file: str | Path,
    output_dir: str | Path,
    registry: LeafRegistry | str | Path,
    corpus: Mapping[str, CorpusCategory],
    task_config: TaskConfig | str | Path,
    grading: GradingConfig,
    *,
    dataset: str = FORMAL_RELEASE_NAME,
    think_generator: ThinkGenerator | None = None,
    max_think_tokens: int = _DEFAULT_MAX_THINK_TOKENS,
    think_over_limit: str = "truncate",
    think_tokenizer=None,
    allow_label_gaps: Sequence[str] = (),
    allow_any_label_gap: bool = False,
) -> dict[str, Any]:
    """Export canonical records to tool-trajectory SFT parquet (per split).

    Shougang-only: ``dataset`` must equal ``FORMAL_RELEASE_NAME`` (finance is
    not part of the trajectory data line). Grading is REQUIRED: the terminal
    assistant JSON carries a sensitivity level and the catalog-in-context
    prompt embeds the rubric. Split boundaries come from canonical schema v2
    embedded ``split`` fields only. The label-gap gate and report shape mirror
    the verified SFT exporter. Think text comes from ``think_generator``
    (default: local deterministic mock; no credentials) and is enforced
    against ``max_think_tokens`` with the configured over-limit policy.
    """

    if dataset != FORMAL_RELEASE_NAME:
        raise ValueError(
            "tool-trajectory SFT dataset must be exactly "
            f"{FORMAL_RELEASE_NAME} (shougang-only line)"
        )
    if think_generator is None:
        think_generator = MockThinkGenerator()
    if think_over_limit not in {"truncate", "discard"}:
        raise ValueError("think_over_limit must be truncate or discard")
    if max_think_tokens <= 0:
        raise ValueError("max_think_tokens must be positive")
    leaf_registry = _registry(registry)
    config = _config(task_config)
    canonical_path = Path(canonical_file)
    output_root = Path(output_dir)
    if not canonical_path.is_file():
        raise FileNotFoundError(f"canonical dataset not found: {canonical_path}")
    canonical_records = load_json_records(canonical_path)
    corpus_map = _corpus_map(corpus)
    env = CategoryToolEnvironment(leaf_registry, corpus_map)

    canonical_resolved = 0
    idless_non_resolved = 0
    for index, item in enumerate(canonical_records):
        item_id = str(item.get("id", "") or "").strip()
        status = str(item.get("resolution_status", "") or "").strip()
        if status == "resolved":
            canonical_resolved += 1
            canonical_target(item, index, canonical_path, leaf_registry)
            if not item_id:
                raise ValueError(f"resolved canonical record without id: {canonical_path}")
            assigned_split = str(item.get("split", "") or "").strip()
            if not assigned_split:
                raise ValueError(
                    f"canonical record {item_id!r} is resolved without a split "
                    "assignment; run script.canonical.split first"
                )
            if assigned_split not in SPLITS:
                raise ValueError(
                    f"canonical record {item_id!r} carries unknown split "
                    f"{assigned_split!r}"
                )
        elif not item_id:
            idless_non_resolved += 1

    report: dict[str, Any] = {
        "format": "verl_tool_trajectory_messages_parquet",
        "trajectory_format": NATIVE_TOOL_TRAJECTORY_FORMAT,
        "trajectory_classes": list(TRAJECTORY_CLASSES),
        "dataset": dataset,
        "label_source": (
            "canonical target.category_id (resolution_status == resolved); "
            "terminal assistant JSON is the only supervised label"
        ),
        "leakage_policy": (
            "tool results are byte-exact CategoryToolEnvironment outputs "
            "(inference-time registry/corpus content); tool-call arguments "
            "derive from record metadata, prior tool results, and browse "
            "scope keys only"
        ),
        "think": {
            "generator": think_generator.name,
            "max_tokens": max_think_tokens,
            "over_limit_policy": think_over_limit,
            "tokenizer": "exact" if think_tokenizer is not None else "conservative_approx",
        },
        "split_source": "embedded_v2",
        "metadata_fields": list(config.metadata_fields),
        "task_name": config.task_name,
        "registry_size": len(leaf_registry.categories),
        "corpus_size": len(corpus_map),
        "canonical_resolved": canonical_resolved,
        "idless_non_resolved_records": idless_non_resolved,
        "grading": {
            "enabled": True,
            "levels": list(grading.levels),
            "gt_field": grading.gt_field,
        },
        "splits": {},
        "trajectory_class_counts": {},
        "think_truncated": 0,
        "think_discarded": 0,
    }

    split_ground_truths: dict[str, set[str]] = {name: set() for name in SPLITS}
    split_levels: dict[str, set[str]] = {name: set() for name in SPLITS}
    all_rows: dict[str, list[dict[str, Any]]] = {}
    think_truncated_total = 0
    think_discarded_total = 0
    for split in SPLITS:
        view = [
            item
            for item in canonical_records
            if str(item.get("split", "") or "").strip() == split
        ]
        rows: list[dict[str, Any]] = []
        skipped_no_grading_label = 0
        skipped_think_discarded = 0
        truncated_think = 0
        for index, item in enumerate(view):
            item_id = str(item.get("id", "") or "").strip()
            if not item_id:
                raise ValueError(f"split item {index} in {canonical_path} has no id")
            status = str(item.get("resolution_status", "") or "").strip()
            if status != "resolved":
                raise ValueError(
                    f"canonical record with non-resolved status {status!r} carries "
                    f"a {split!r} assignment; re-run script.canonical.split"
                )
            raw_level = item.get(grading.gt_field)
            if raw_level is None or (
                isinstance(raw_level, str) and not raw_level.strip()
            ):
                skipped_no_grading_label += 1
                continue
            ground_truth = canonical_target(item, index, canonical_path, leaf_registry)
            if ground_truth is None:
                raise ValueError(f"internal error: unresolved record {item_id!r}")
            split_ground_truths[split].add(ground_truth)
            split_levels[split].add(str(raw_level).strip())
            row = build_trajectory(
                item,
                index,
                canonical_path,
                registry=leaf_registry,
                corpus=corpus_map,
                config=config,
                grading=grading,
                env=env,
                dataset=dataset,
                think_generator=think_generator,
                max_think_tokens=max_think_tokens,
                think_over_limit=think_over_limit,
                think_tokenizer=think_tokenizer,
            )
            if row is None:
                skipped_think_discarded += 1
                continue
            if row["think_truncated"]:
                truncated_think += 1
            rows.append(row)
        if not rows:
            raise ValueError(f"no resolved records available in {canonical_path}")
        all_rows[split] = rows
        think_truncated_total += truncated_think
        think_discarded_total += skipped_think_discarded

        class_counts = {name: 0 for name in TRAJECTORY_CLASSES}
        for row in rows:
            class_counts[row["trajectory_class"]] += 1
        report["splits"][split] = {
            "split_records": len(view),
            "exported_records": len(rows),
            "skipped_no_grading_label": skipped_no_grading_label,
            "skipped_think_discarded": skipped_think_discarded,
            "think_truncated": truncated_think,
            "trajectory_class_counts": class_counts,
        }
        for name, count in class_counts.items():
            report["trajectory_class_counts"][name] = (
                report["trajectory_class_counts"].get(name, 0) + count
            )
    report["think_truncated"] = think_truncated_total
    report["think_discarded"] = think_discarded_total

    blocking_gaps: list[dict[str, str]] = []
    waived_gaps: list[dict[str, str]] = []
    blocking_levels: list[dict[str, str]] = []
    waived_levels: list[dict[str, str]] = []
    allowed_gaps = {str(label) for label in allow_label_gaps}
    for later_split in ("val", "test"):
        for label in sorted(split_ground_truths[later_split] - split_ground_truths["train"]):
            entry = {"label": label, "split": later_split}
            if allow_any_label_gap or label in allowed_gaps:
                waived_gaps.append(entry)
            else:
                blocking_gaps.append(entry)
        for code in sorted(split_levels[later_split] - split_levels["train"]):
            entry = {"label": code, "split": later_split}
            if allow_any_label_gap or code in allowed_gaps:
                waived_levels.append(entry)
            else:
                blocking_levels.append(entry)
    gate_status = "failed" if (blocking_gaps or blocking_levels) else (
        "waived" if (waived_gaps or waived_levels) else "passed"
    )
    report["label_gap_gate"] = {
        "status": gate_status,
        "blocking": blocking_gaps,
        "waived": waived_gaps,
        "blocking_levels": blocking_levels,
        "waived_levels": waived_levels,
    }
    if blocking_gaps or blocking_levels:
        raise ValueError(
            "label-gap gate failed; labels absent from train but present in "
            f"val/test: {blocking_gaps + blocking_levels}. Waive via "
            "allow_label_gaps/allow_any_label_gap after review."
        )

    output_root.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        destination = output_root / f"{split}.parquet"
        _write_parquet(all_rows[split], destination)
        report["splits"][split].update(
            {
                "output_file": str(destination),
                "parquet_sha256": sha256_file(destination),
            }
        )
    (output_root / "export_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def _validate_row(
    row: Mapping[str, Any],
    index: int,
    registry: LeafRegistry,
    config: TaskConfig,
    corpus: Mapping[str, CorpusCategory],
    grading: GradingConfig,
    env: CategoryToolEnvironment,
    dataset: str = FORMAL_RELEASE_NAME,
    max_think_tokens: int = _DEFAULT_MAX_THINK_TOKENS,
) -> list[str]:
    errors: list[str] = []
    if row.get("stage") != _STAGE:
        errors.append("stage must be tool_trajectory")
    if row.get("trajectory_format") != NATIVE_TOOL_TRAJECTORY_FORMAT:
        errors.append(f"trajectory_format must be {NATIVE_TOOL_TRAJECTORY_FORMAT}")
    if row.get("dataset") != dataset or dataset != FORMAL_RELEASE_NAME:
        errors.append(
            f"dataset must be exactly {FORMAL_RELEASE_NAME} (shougang-only line)"
        )
    trajectory_class = row.get("trajectory_class")
    if trajectory_class not in TRAJECTORY_CLASSES:
        errors.append(f"trajectory_class must be one of {list(TRAJECTORY_CLASSES)}")
        return errors
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        errors.append("messages must be a non-empty list")
        return errors
    if any(not isinstance(message, Mapping) for message in messages):
        errors.append("every message must be an object")
        return errors
    for message_index, message in enumerate(messages):
        if not set(message) <= {"role", "content", "reasoning_content"}:
            errors.append(
                f"message {message_index} has unexpected keys {sorted(set(message))}"
            )
    roles = [message.get("role") for message in messages]
    expected_tool_calls = _EXPECTED_TOOL_CALLS[trajectory_class]
    if roles[:2] != ["system", "user"]:
        errors.append("messages must start with system, user")
    tail = roles[2:]
    expected_tail = []
    for _ in range(expected_tool_calls):
        expected_tail.extend(["assistant", "tool"])
    expected_tail.append("assistant")
    if tail != expected_tail:
        errors.append(
            "roles must be system,user followed by "
            f"(assistant,tool)*{expected_tool_calls} then assistant; "
            f"got {roles!r}"
        )
    contents = [
        message.get("content") if isinstance(message, Mapping) else None
        for message in messages
    ]
    if not all(isinstance(content, str) for content in contents):
        errors.append("every message content must be a string")
    elif any(
        token in "".join(contents)
        for token in ("<|im_start|>", "<|im_end|>")
    ):
        errors.append("messages must not contain serialized chat-template tokens")

    source_id = row.get("source_id")
    if not isinstance(source_id, str) or not source_id.strip():
        errors.append("source_id must be a non-empty string")
    ground_truth = row.get("ground_truth")
    if not isinstance(ground_truth, str) or ground_truth not in registry.ids:
        errors.append("ground_truth must be an ID from the leaf registry")
    level = row.get("ground_truth_level")
    if not isinstance(level, str) or level.strip() not in grading.levels:
        errors.append(f"ground_truth_level must be one of {list(grading.levels)}")
    metadata = row.get("metadata")
    if not isinstance(metadata, Mapping) or set(metadata) != set(config.metadata_fields):
        errors.append("metadata must exactly match task config metadata_fields")

    if errors:
        return errors

    terminal = contents[-1]
    try:
        parsed = parse_final_tool_answer(terminal, registry=registry, grading=grading)
    except (TypeError, ValueError) as exc:
        errors.append(f"terminal assistant JSON is invalid: {exc}")
        return errors
    if parsed.category_id != ground_truth or parsed.level != level.strip():
        errors.append("terminal JSON must equal ground_truth and ground_truth_level")

    if not all(
        isinstance(content, str) and content
        for content in (contents[0], contents[1])
    ):
        errors.append("system/user contents must be non-empty")
    elif ground_truth in contents[0] + contents[1]:
        errors.append("system/user prompt must not expose canonical ground truth")

    # Think (reasoning_content) audit, source-aware (coordinator decision
    # 2026-08-27): public constraints for every source are non-empty, within
    # the token ceiling, closed (balanced brackets), no terminal JSON object
    # and no `"answer"`/`"level"` key-value lookalikes (answer_mask
    # doppelgangers). Mock think stays ground-truth-free; file think (the
    # label-aware sub-agent seam) may contain bare ground-truth ids as
    # legitimate reasoning.
    think_source = row.get("think_source")
    if not isinstance(think_source, str) or not think_source.strip():
        errors.append("think_source must be a non-empty generator name")
    terminal_message = messages[-1]
    reasoning = terminal_message.get("reasoning_content")
    if not isinstance(reasoning, str) or not reasoning.strip():
        errors.append("terminal assistant message must carry non-empty reasoning_content")
    else:
        if estimate_think_tokens(reasoning) > max_think_tokens:
            errors.append(
                f"reasoning_content exceeds max_think_tokens={max_think_tokens}"
            )
        if not _think_closed(reasoning):
            errors.append("reasoning_content is not closed (unbalanced brackets)")
        if _THINK_KEY_LOOKALIKE.search(reasoning):
            errors.append(
                "reasoning_content must not contain answer/level key-value "
                "lookalikes (answer_mask doppelganger)"
            )
        if _THINK_TERMINAL_JSON.search(reasoning):
            errors.append(
                "reasoning_content must not contain the terminal JSON object"
            )
        if think_source == "mock" and ground_truth in reasoning:
            errors.append("mock reasoning_content must not expose canonical ground truth")
    for message in messages[:-1]:
        # Parquet list<struct> round-trips pad missing keys with None; only a
        # non-None reasoning_content on a non-terminal message is a violation.
        if message.get("reasoning_content") is not None:
            errors.append("only the terminal assistant message may carry reasoning_content")

    tool_calls = row.get("tool_calls")
    if not isinstance(tool_calls, list) or len(tool_calls) != expected_tool_calls:
        errors.append(
            f"tool_calls must record exactly {expected_tool_calls} calls"
        )
        return errors

    # Re-execute every recorded call and compare byte-for-byte with the tool
    # message contents (leakage audit: results are real environment output).
    tool_messages = [m for m in messages if m.get("role") == "tool"]
    prior_choice_ids: set[str] = set()
    browse_groups: set[str] = set()
    seen_names: set[str] = set()
    for call_index, call in enumerate(tool_calls):
        if not isinstance(call, Mapping) or not isinstance(call.get("name"), str):
            errors.append(f"tool_calls[{call_index}] must carry a name")
            continue
        name = call["name"]
        if name in seen_names and name == "search_categories":
            errors.append("search_categories must not be called twice")
        seen_names.add(name)
        arguments = call.get("arguments")
        if not isinstance(arguments, Mapping):
            errors.append(f"tool_calls[{call_index}] must carry arguments")
            continue
        # Parquet list<struct> round-trips pad missing struct keys with None;
        # treat None-valued keys as absent before re-execution.
        arguments = {
            key: value for key, value in arguments.items() if value is not None
        }
        try:
            if name == "search_categories":
                field_name = str(arguments.get("field_name", ""))
                table_name = str(arguments.get("table_name", ""))
                if field_name != metadata.get("field_name", "") or (
                    table_name != metadata.get("table_name", "")
                ):
                    errors.append(
                        "search arguments must equal the record metadata field/table"
                    )
                scope = arguments.get("scope")
                if scope is None:
                    result = env.search_categories(field_name, table_name)
                else:
                    if scope not in browse_groups:
                        errors.append("search scope must come from a prior browse result")
                    result = env.search_categories(field_name, table_name, scope=scope)
            elif name == "get_category_details":
                choice_ids = list(arguments.get("choice_ids", []))
                if not choice_ids or any(
                    choice_id not in prior_choice_ids for choice_id in choice_ids
                ):
                    errors.append("details choice_ids must come from prior tool results")
                result = env.get_category_details(choice_ids)
            elif name == "get_category_examples":
                choice_ids = list(arguments.get("choice_ids", []))
                if not choice_ids or any(
                    choice_id not in prior_choice_ids for choice_id in choice_ids
                ):
                    errors.append("examples choice_ids must come from prior tool results")
                result = env.get_category_examples(choice_ids, limit=arguments["limit"])
            elif name == "browse_categories":
                result = env.browse_categories()
                browse_groups = {
                    str(group["scope_key"]) for group in result.get("groups", [])
                }
            else:
                errors.append(f"unexpected tool name {name!r}")
                continue
        except (TypeError, ValueError, RuntimeError) as exc:
            errors.append(f"tool_calls[{call_index}] re-execution failed: {exc}")
            continue
        prior_choice_ids |= set(_result_choice_ids(result))
        expected_content = json.dumps(result, ensure_ascii=False)
        if call_index >= len(tool_messages):
            errors.append("tool_calls exceed the number of tool messages")
            continue
        actual_content = tool_messages[call_index].get("content")
        if actual_content != expected_content:
            errors.append(
                f"tool message {call_index} is not the byte-exact environment result"
            )

    if trajectory_class == "no_result":
        surfaced = _result_choice_ids_of_messages(tool_messages)
        if ground_truth in registry.ids:
            choices = PromptChoiceRegistry.from_registry(registry)
            if choices.choice_id_of(ground_truth) in surfaced:
                errors.append(
                    "no_result trajectory must not surface the ground-truth choice id"
                )
    return errors


def _result_choice_ids_of_messages(tool_messages: Sequence[Mapping[str, Any]]) -> set[str]:
    ids: set[str] = set()
    for message in tool_messages:
        content = message.get("content")
        if not isinstance(content, str):
            continue
        try:
            value = json.loads(content)
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping):
            ids |= set(_result_choice_ids(value))
    return ids


def validate_tool_trajectory_dataset(
    dataset_dir: str | Path,
    registry: LeafRegistry | str | Path,
    corpus: Mapping[str, CorpusCategory],
    task_config: TaskConfig | str | Path,
    grading: GradingConfig,
    *,
    dataset: str = FORMAL_RELEASE_NAME,
    max_think_tokens: int = _DEFAULT_MAX_THINK_TOKENS,
) -> dict[str, Any]:
    """Return a structured validation report for a tool-trajectory release."""

    if dataset != FORMAL_RELEASE_NAME:
        raise ValueError(
            "tool-trajectory SFT dataset must be exactly "
            f"{FORMAL_RELEASE_NAME} (shougang-only line)"
        )
    leaf_registry = _registry(registry)
    config = _config(task_config)
    corpus_map = _corpus_map(corpus)
    env = CategoryToolEnvironment(leaf_registry, corpus_map)
    root = Path(dataset_dir)
    report: dict[str, Any] = {
        "format": "verl_tool_trajectory_messages_parquet",
        "valid": True,
        "trajectory_format": NATIVE_TOOL_TRAJECTORY_FORMAT,
        "trajectory_classes": list(TRAJECTORY_CLASSES),
        "dataset": dataset,
        "max_think_tokens": max_think_tokens,
        "metadata_fields": list(config.metadata_fields),
        "task_name": config.task_name,
        "registry_size": len(leaf_registry.categories),
        "corpus_size": len(corpus_map),
        "grading": {
            "enabled": True,
            "levels": list(grading.levels),
            "gt_field": grading.gt_field,
        },
        "splits": {},
        "cross_split_errors": [],
    }
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("tool-trajectory validation requires pyarrow") from exc
    source_ids_by_split: dict[str, set[str]] = {}
    for split in SPLITS:
        path = root / f"{split}.parquet"
        details = {"rows": 0, "errors": []}
        rows: list[Mapping[str, Any]] = []
        if not path.is_file():
            details["errors"].append(f"missing parquet file: {path}")
        else:
            try:
                rows = pq.read_table(path).to_pylist()
                details["rows"] = len(rows)
                if not rows:
                    details["errors"].append("parquet split must contain at least one row")
                for index, row in enumerate(rows):
                    for error in _validate_row(
                        row,
                        index,
                        leaf_registry,
                        config,
                        corpus_map,
                        grading,
                        env,
                        dataset=dataset,
                        max_think_tokens=max_think_tokens,
                    ):
                        details["errors"].append(f"row {index}: {error}")
                seen: set[str] = set()
                for row in rows:
                    source_id = row.get("source_id")
                    if isinstance(source_id, str) and source_id.strip():
                        if source_id in seen:
                            details["errors"].append(
                                f"duplicate source_id in split: {source_id}"
                            )
                        seen.add(source_id)
            except Exception as exc:
                details["errors"].append(f"cannot read parquet: {exc}")
        source_ids_by_split[split] = {
            source_id
            for row in rows
            if isinstance((source_id := row.get("source_id")), str)
            and source_id.strip()
        }
        report["splits"][split] = details
        if details["errors"]:
            report["valid"] = False

    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            overlap = source_ids_by_split[left] & source_ids_by_split[right]
            if overlap:
                examples = ", ".join(sorted(overlap)[:5])
                report["cross_split_errors"].append(
                    f"source_id overlap between {left} and {right}: {examples}"
                )
    if report["cross_split_errors"]:
        report["valid"] = False
    return report


__all__ = [
    "SPLITS",
    "TRAJECTORY_CLASSES",
    "select_trajectory_class",
    "build_tool_trajectory_prompt",
    "render_tool_call",
    "build_trajectory",
    "export_tool_trajectory_dataset",
    "validate_tool_trajectory_dataset",
]

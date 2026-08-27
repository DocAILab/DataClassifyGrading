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

- ``direct``: no tools; the assistant thinks briefly, then emits the strict
  terminal JSON immediately after the user turn.
- ``single_tool``: one assistant think + ``search_categories`` call, real
  result, terminal think + JSON.
- ``multi_tool``: three assistant think/tool rounds
  (``search_categories`` -> ``get_category_details`` ->
  ``get_category_examples``), real results, then terminal think + JSON.
- ``no_result``: two assistant think/tool rounds (``browse_categories`` + a
  scoped search in a level-1 group that does NOT contain the ground truth);
  the model recovers and still emits terminal JSON even though tools never
  surfaced the answer.

Every assistant message also carries a ``reasoning_content`` key (thinking
text) produced by a pluggable think generator. Tool-call turns use a short
think budget (at most 64 tokens); the terminal assistant keeps the 128-token
budget. The default is the local deterministic :class:`MockThinkGenerator`
(no credentials, no network); a real deepseek-v4-flash-backed generator plugs
into the same protocol later (runtime-local credentials only). Generated think
text is enforced against the per-turn budget: over-limit text is truncated to
the prefix within budget or the row is discarded, per ``think_over_limit``.
Loss scope is pinned by decision: think enters the SFT loss at weight 1 and the
answer/level value spans at weight 8 (the server scheme-C answer_mask patch is
the contract; the patch is not changed).

Leakage discipline (audited by :func:`validate_tool_trajectory_dataset`):

- tool results are byte-exact outputs of ``CategoryToolEnvironment`` calls
  re-executed from the recorded arguments (inference-time content only);
- tool-call arguments come only from the record metadata, ids returned by
  earlier tool results, and scope keys returned by browse;
- the system/user prompt and mock reasoning_content never contain the
  canonical ground-truth id; label-aware file reasoning may contain a bare id
  but never frames a terminal JSON object or answer/level key-value lookalike;
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
import inspect
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
# Default ceilings for generated think text (coordinator decision
# 2026-08-27): terminal think keeps a 128-token cap and every tool-call
# assistant turn gets the short 64-token cap. ``max_think_tokens`` remains
# the public terminal cap; tool turns use ``min(64, max_think_tokens)`` so a
# caller choosing a smaller global budget still gets a valid row.
_DEFAULT_MAX_THINK_TOKENS = 128
_TOOL_MAX_THINK_TOKENS = 64
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")


class ThinkGenerator(Protocol):
    """Pluggable think-text generator for each assistant turn.

    ``turn_index`` is zero-based across assistant messages (tool calls first,
    terminal answer last), and ``is_terminal`` identifies the final turn.
    Implementations must be deterministic for identical inputs when used in
    reproducible exports. They must not receive or emit ground-truth content
    unless the source is explicitly label-aware (the file source is audited
    separately). The default :class:`MockThinkGenerator` is local and
    credential-free; a deepseek-v4-flash-backed implementation plugs in here
    with runtime-local credentials only.

    Third-party generators implementing the original three-argument protocol
    remain supported: the exporter only supplies turn arguments when the
    ``generate`` signature accepts them.
    """

    name: str

    def generate(
        self,
        *,
        source_id: str,
        metadata: Mapping[str, str],
        trajectory_class: str,
        turn_index: int = 0,
        is_terminal: bool = True,
    ) -> str: ...


class MockThinkGenerator:
    """Deterministic local think generator (no credentials, no network).

    Emits one or two short reasoning sentences anchored to inference-time
    content only (metadata, trajectory class, and turn position). It never
    contains canonical ids or the ground truth, and its output is stable
    across runs. Tool turns deliberately use distinct plans so a mock export
    teaches the assistant/tool alternation rather than repeating terminal
    prose on every turn.
    """

    name = "mock"

    _TOOL_THINK = {
        "single_tool": (
            "I will search the catalog using the field and table metadata before selecting a candidate.",
        ),
        "multi_tool": (
            "I will search the catalog for candidates that match the visible field semantics.",
            "I will inspect the returned definitions to narrow the candidate set.",
            "I will check examples for the remaining candidate before answering.",
        ),
        "no_result": (
            "I will browse category scopes before trying a targeted search.",
            "The scoped lookup may miss the target, so I will recover from the available catalog context.",
        ),
    }

    def generate(
        self,
        *,
        source_id: str,
        metadata: Mapping[str, str],
        trajectory_class: str,
        turn_index: int = 0,
        is_terminal: bool = True,
    ) -> str:
        del source_id  # deterministic output is intentionally source-id agnostic
        if not is_terminal:
            options = self._TOOL_THINK.get(trajectory_class)
            if options:
                return options[min(max(turn_index, 0), len(options) - 1)]
            return "I will inspect the catalog before selecting a candidate."
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

    The recommended shard shape is a terminal ``think`` string plus an ordered
    ``tool_think`` list (one entry for each assistant tool-call turn). Compact
    list/mapping forms and their historical aliases remain accepted, but
    conflicting or mixed sources are rejected instead of being resolved by
    precedence. A legacy one-string shard is reusable for every assistant turn
    only when *all* new-schema markers are absent.
    """

    name = "file"
    _ASSISTANT_ALIASES = ("assistant_think", "assistant_thinks", "think_turns")
    _TOOL_ALIASES = ("tool_think", "tool_thinks")
    _INLINE_ASSISTANT_ALIASES = ("assistant", "turns", "assistant_think", "assistant_thinks")
    _INLINE_TOOL_ALIASES = ("tool", "tools", "tool_think", "tool_thinks")
    _INLINE_TERMINAL_ALIASES = ("terminal", "final")

    def __init__(self, path: str | Path) -> None:
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(f"think shard source not found: {source}")
        shard_files = (
            sorted(source.glob("*.jsonl")) if source.is_dir() else [source]
        )
        if not shard_files:
            raise ValueError(f"think shard directory contains no *.jsonl: {source}")
        self._think: dict[str, dict[str, Any]] = {}
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
                if not isinstance(entry, Mapping):
                    raise ValueError(
                        f"think shard {shard} line {line_number} must contain an object"
                    )
                sample_id = entry.get("sample_id")
                if not isinstance(sample_id, str) or not sample_id.strip():
                    raise ValueError(
                        f"think shard {shard} line {line_number} lacks sample_id"
                    )
                if sample_id in self._think:
                    raise ValueError(
                        f"duplicate sample_id {sample_id!r} across think shards"
                    )
                self._think[sample_id] = self._normalize_entry(
                    entry, shard=shard, line_number=line_number
                )

    @staticmethod
    def _alias_value(
        mapping: Mapping[str, Any],
        aliases: Sequence[str],
        *,
        field: str,
        shard: Path,
        line_number: int,
    ) -> tuple[bool, Any]:
        """Return one alias value, rejecting duplicate spellings.

        Alias precedence is unsafe for label-aware data: two producers can
        otherwise provide different terminal/tool slots and the later lookup
        silently wins. Presence is tracked separately from the value so an
        explicit ``null``/empty field can never be mistaken for omission.
        """

        present = [name for name in aliases if name in mapping]
        if len(present) > 1:
            raise ValueError(
                f"think shard {shard} line {line_number} has conflicting "
                f"{field} aliases: {', '.join(present)}"
            )
        if not present:
            return False, None
        return True, mapping[present[0]]

    @classmethod
    def _string_list(
        cls,
        value: Any,
        *,
        field: str,
        shard: Path,
        line_number: int,
        present: bool = True,
    ) -> list[str] | None:
        if not present:
            return None
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ValueError(
                f"think shard {shard} line {line_number} field {field!r} "
                "must be a list of strings"
            )
        for index, item in enumerate(value):
            if not item.strip():
                raise ValueError(
                    f"think shard {shard} line {line_number} {field} slot "
                    f"{index} must be non-empty"
                )
        return list(value)

    @staticmethod
    def _nonempty_string(
        value: Any, *, field: str, shard: Path, line_number: int
    ) -> str:
        if not isinstance(value, str):
            raise ValueError(
                f"think shard {shard} line {line_number} {field} must be a string"
            )
        if not value.strip():
            raise ValueError(
                f"think shard {shard} line {line_number} {field} must be non-empty"
            )
        return value

    @classmethod
    def _normalize_entry(
        cls, entry: Mapping[str, Any], *, shard: Path, line_number: int
    ) -> dict[str, Any]:
        think_present = "think" in entry
        think = entry.get("think")
        terminal: str | None = None
        assistant: list[str] | None = None
        inline_tool: list[str] | None = None
        inline_assistant_present = False
        inline_tool_present = False
        inline_terminal_present = False

        if isinstance(think, str):
            terminal = cls._nonempty_string(
                think, field="terminal think", shard=shard, line_number=line_number
            )
        elif isinstance(think, list):
            assistant = cls._string_list(
                think,
                field="think",
                shard=shard,
                line_number=line_number,
            )
            if assistant:
                terminal = assistant[-1]
        elif isinstance(think, Mapping):
            allowed = set(cls._INLINE_ASSISTANT_ALIASES) | set(
                cls._INLINE_TOOL_ALIASES
            ) | set(cls._INLINE_TERMINAL_ALIASES)
            unknown = sorted(set(think) - allowed)
            if unknown:
                raise ValueError(
                    f"think shard {shard} line {line_number} has unsupported "
                    f"think mapping fields: {unknown}"
                )
            inline_terminal_present, terminal_value = cls._alias_value(
                think,
                cls._INLINE_TERMINAL_ALIASES,
                field="terminal think",
                shard=shard,
                line_number=line_number,
            )
            if inline_terminal_present:
                terminal = cls._nonempty_string(
                    terminal_value,
                    field="terminal think",
                    shard=shard,
                    line_number=line_number,
                )
            inline_assistant_present, assistant_value = cls._alias_value(
                think,
                cls._INLINE_ASSISTANT_ALIASES,
                field="assistant think",
                shard=shard,
                line_number=line_number,
            )
            assistant = cls._string_list(
                assistant_value,
                field="think.assistant",
                shard=shard,
                line_number=line_number,
                present=inline_assistant_present,
            )
            inline_tool_present, tool_value = cls._alias_value(
                think,
                cls._INLINE_TOOL_ALIASES,
                field="tool_think",
                shard=shard,
                line_number=line_number,
            )
            inline_tool = cls._string_list(
                tool_value,
                field="think.tool",
                shard=shard,
                line_number=line_number,
                present=inline_tool_present,
            )
            if inline_assistant_present and inline_tool_present:
                raise ValueError(
                    f"think shard {shard} line {line_number} mixes assistant "
                    "and tool think sources"
                )
            if assistant:
                derived_terminal = assistant[-1]
                if terminal is None:
                    terminal = derived_terminal
                elif terminal != derived_terminal:
                    raise ValueError(
                        f"think shard {shard} line {line_number} has conflicting "
                        "terminal think sources"
                    )
        elif think_present and think is not None:
            raise ValueError(
                f"think shard {shard} line {line_number} field 'think' "
                "must be a string, list, or object"
            )

        assistant_alias_present, explicit_assistant = cls._alias_value(
            entry,
            cls._ASSISTANT_ALIASES,
            field="assistant think",
            shard=shard,
            line_number=line_number,
        )
        parsed_assistant = cls._string_list(
            explicit_assistant,
            field="assistant_think",
            shard=shard,
            line_number=line_number,
            present=assistant_alias_present,
        )
        tool_alias_present, explicit_tool = cls._alias_value(
            entry,
            cls._TOOL_ALIASES,
            field="tool_think",
            shard=shard,
            line_number=line_number,
        )
        parsed_tool = cls._string_list(
            explicit_tool,
            field="tool_think",
            shard=shard,
            line_number=line_number,
            present=tool_alias_present,
        )

        # The only mixed source intentionally supported is the recommended
        # terminal string + top-level ordered tool_think list. All other
        # compact/explicit combinations are ambiguous, even when one value is
        # empty; never let an empty explicit list be backfilled by inline data.
        if inline_assistant_present or inline_tool_present:
            if assistant_alias_present or tool_alias_present:
                raise ValueError(
                    f"think shard {shard} line {line_number} mixes inline and "
                    "top-level think sources"
                )
        if isinstance(think, (list, Mapping)) and (
            assistant_alias_present or tool_alias_present
        ):
            raise ValueError(
                f"think shard {shard} line {line_number} mixes compact and "
                "top-level think sources"
            )
        if isinstance(think, str) and assistant_alias_present:
            raise ValueError(
                f"think shard {shard} line {line_number} mixes terminal think "
                "with assistant think sources"
            )

        if parsed_assistant is not None:
            assistant = parsed_assistant
            if assistant:
                derived_terminal = assistant[-1]
                if terminal is None:
                    terminal = derived_terminal
                elif terminal != derived_terminal:
                    raise ValueError(
                        f"think shard {shard} line {line_number} has conflicting "
                        "terminal think sources"
                    )
        if parsed_tool is not None:
            inline_tool = parsed_tool

        # ``assistant_turns`` is a schema marker, not optional metadata. Even
        # a null/incorrect value must not silently opt into legacy fallback.
        assistant_turns_present = "assistant_turns" in entry
        assistant_turns = entry.get("assistant_turns")
        if assistant_turns_present:
            if (
                isinstance(assistant_turns, bool)
                or not isinstance(assistant_turns, int)
                or assistant_turns < 1
            ):
                raise ValueError(
                    f"think shard {shard} line {line_number} assistant_turns "
                    "must be a positive integer"
                )

        # A terminal value is mandatory for both legacy and new shards. This
        # catches empty strings at load time rather than allowing a row with an
        # empty reasoning_content to reach parquet publication/validation.
        if terminal is None:
            raise ValueError(
                f"think shard {shard} line {line_number} lacks terminal think"
            )
        terminal = cls._nonempty_string(
            terminal, field="terminal think", shard=shard, line_number=line_number
        )

        tool_explicit = tool_alias_present or inline_tool_present
        assistant_explicit = assistant_alias_present or inline_assistant_present or isinstance(
            think, list
        )
        new_schema = (
            assistant_turns_present
            or tool_explicit
            or assistant_explicit
            or isinstance(think, Mapping)
        )

        if assistant_turns_present and assistant is not None:
            if len(assistant) != assistant_turns:
                raise ValueError(
                    f"think shard {shard} line {line_number} assistant_turns "
                    "does not match assistant think slots"
                )
        if assistant_turns_present and inline_tool is not None:
            if len(inline_tool) + 1 != assistant_turns:
                raise ValueError(
                    f"think shard {shard} line {line_number} assistant_turns "
                    "does not match tool_think slots"
                )
        # The marker alone already declares the tool round count (``assistant_
        # turns - 1``); an entry that declares tool rounds without any
        # tool_think/assistant slots is internally inconsistent for every
        # trajectory class and is rejected here at load time instead of
        # surfacing per-sample during generate().
        if (
            assistant_turns_present
            and assistant_turns > 1
            and inline_tool is None
            and assistant is None
        ):
            raise ValueError(
                f"think shard {shard} line {line_number} assistant_turns "
                f"{assistant_turns} declares {assistant_turns - 1} tool "
                "rounds but no tool_think/assistant think slots are present"
            )

        return {
            "terminal": terminal,
            "tool": inline_tool,
            "assistant": assistant,
            "tool_explicit": tool_explicit,
            "assistant_explicit": assistant_explicit,
            "new_schema": new_schema,
            "assistant_turns": assistant_turns,
        }

    def generate(
        self,
        *,
        source_id: str,
        metadata: Mapping[str, str],
        trajectory_class: str,
        turn_index: int = 0,
        is_terminal: bool = True,
    ) -> str:
        del metadata
        if source_id not in self._think:
            raise ValueError(
                f"think shards ({len(self._shard_files)} file(s)) are missing "
                f"sample_id {source_id!r}; assemble requires every record"
            )
        expected_tool_calls = _EXPECTED_TOOL_CALLS.get(trajectory_class)
        if expected_tool_calls is None:
            raise ValueError(f"unknown trajectory class {trajectory_class!r}")
        if not isinstance(turn_index, int) or isinstance(turn_index, bool) or turn_index < 0:
            raise ValueError("turn_index must be a non-negative integer")
        entry = self._think[source_id]
        expected_assistant_turns = expected_tool_calls + 1
        assistant_turns = entry.get("assistant_turns")
        if assistant_turns is not None and assistant_turns != expected_assistant_turns:
            raise ValueError(
                f"think shard sample_id {source_id!r} assistant_turns must be "
                f"{expected_assistant_turns} for {trajectory_class}"
            )

        tool = entry.get("tool")
        if entry.get("tool_explicit"):
            if not isinstance(tool, list) or len(tool) != expected_tool_calls:
                raise ValueError(
                    f"think shard sample_id {source_id!r} must provide exactly "
                    f"{expected_tool_calls} tool_think slots for {trajectory_class}"
                )
        assistant = entry.get("assistant")
        if entry.get("assistant_explicit"):
            if not isinstance(assistant, list) or len(assistant) != expected_assistant_turns:
                raise ValueError(
                    f"think shard sample_id {source_id!r} must provide exactly "
                    f"{expected_assistant_turns} assistant think slots for {trajectory_class}"
                )

        # New schemas cannot silently fall back to their terminal string for a
        # missing tool slot. A complete assistant list is an independent
        # compact form; otherwise a tool trajectory needs explicit tool slots.
        if (
            entry.get("new_schema")
            and expected_tool_calls > 0
            and not entry.get("tool_explicit")
            and not entry.get("assistant_explicit")
        ):
            raise ValueError(
                f"think shard sample_id {source_id!r} new schema requires "
                f"{expected_tool_calls} tool_think slots for {trajectory_class}"
            )

        if is_terminal:
            # The terminal value is always independently validated during
            # normalization and never selected from a tool slot.
            return entry["terminal"]

        if turn_index >= expected_tool_calls:
            raise ValueError(
                f"think shard sample_id {source_id!r} tool turn index {turn_index} "
                f"is outside {expected_tool_calls} expected slots"
            )
        if entry.get("tool_explicit"):
            value = tool[turn_index]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"think shard sample_id {source_id!r} tool_think slot "
                    f"{turn_index} must be non-empty"
                )
            return value
        if entry.get("assistant_explicit"):
            value = assistant[turn_index]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"think shard sample_id {source_id!r} assistant think slot "
                    f"{turn_index} must be non-empty"
                )
            return value

        # The only path that reuses one string is a true legacy shard: every
        # new marker (assistant_turns, tool/assistant aliases, compact list or
        # mapping) is absent.
        if not entry.get("new_schema"):
            return entry["terminal"]
        raise ValueError(
            f"think shard sample_id {source_id!r} is missing a tool think slot"
        )


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
_THINK_SENTENCE_END = re.compile(r"[.!?。！？]+")


def _think_sentence_count(text: str) -> int:
    """Return a conservative sentence count for the short-think contract."""

    if not isinstance(text, str) or not text.strip():
        return 0
    return max(1, len(_THINK_SENTENCE_END.findall(text)))


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


def _generate_think(
    think_generator: ThinkGenerator,
    *,
    source_id: str,
    metadata: Mapping[str, str],
    trajectory_class: str,
    turn_index: int,
    is_terminal: bool,
) -> str:
    """Call a think generator while preserving the pre-turn API seam.

    The original generator protocol accepted only ``source_id``, ``metadata``
    and ``trajectory_class``.  New generators can use ``turn_index`` and
    ``is_terminal`` to produce per-assistant-turn thoughts; signature probing
    keeps existing local/test generators source-compatible and avoids catching
    unrelated ``TypeError`` exceptions raised inside a generator.
    """

    generate = think_generator.generate
    base_kwargs = {
        "source_id": source_id,
        "metadata": metadata,
        "trajectory_class": trajectory_class,
    }
    try:
        parameters = inspect.signature(generate).parameters
    except (TypeError, ValueError):  # pragma: no cover - unusual C extensions
        parameters = {}
    accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    kwargs = dict(base_kwargs)
    if accepts_kwargs or "turn_index" in parameters:
        kwargs["turn_index"] = turn_index
    elif "assistant_turn" in parameters:
        kwargs["assistant_turn"] = turn_index
    elif "turn" in parameters:
        kwargs["turn"] = turn_index
    if accepts_kwargs or "is_terminal" in parameters:
        kwargs["is_terminal"] = is_terminal
    elif "terminal" in parameters:
        kwargs["terminal"] = is_terminal
    return generate(**kwargs)


def _think_budget(is_terminal: bool, max_think_tokens: int) -> int:
    """Return the cap for one assistant turn.

    ``max_think_tokens`` is historically the terminal cap; keeping tool turns
    bounded by the fixed 64-token policy while applying a smaller requested
    cap makes the CLI's existing option intuitive for smoke tests.
    """

    return max_think_tokens if is_terminal else min(_TOOL_MAX_THINK_TOKENS, max_think_tokens)


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
    from :func:`_build_trajectory_context`; one think text is injected into
    every assistant message. Tool-call turns use a 64-token cap and the
    terminal assistant uses ``max_think_tokens`` (128 by default). Returns
    None when ``think_over_limit == "discard"`` and any turn exceeds its cap
    (the caller counts the skip).
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
    think_truncated_turns = 0
    assistant_turn = 0
    messages = context["messages"]
    for message in messages:
        if message.get("role") != "assistant":
            continue
        is_terminal = message is messages[-1]
        turn_cap = _think_budget(is_terminal, max_think_tokens)
        think_text = _generate_think(
            think_generator,
            source_id=context["source_id"],
            metadata=context["metadata"],
            trajectory_class=context["trajectory_class"],
            turn_index=assistant_turn,
            is_terminal=is_terminal,
        )
        if estimate_think_tokens(think_text, think_tokenizer) > turn_cap:
            if think_over_limit == "discard":
                return None
            think_text = _truncate_think_to_budget(
                think_text, turn_cap, think_tokenizer
            )
            if not think_text.strip():
                raise ValueError(
                    f"record {context['source_id']!r} think truncation produced an "
                    "empty reasoning_content; no closed prefix fits the budget"
                )
            think_truncated_turns += 1
        message["reasoning_content"] = think_text
        assistant_turn += 1
    think_truncated = think_truncated_turns > 0
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
        "think_truncated_turns": think_truncated_turns,
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
    """Export think-free trajectory contexts as JSONL shards.

    Every shard line carries the deterministic trajectory context (prompt,
    tool calls, tool results, ground truth, terminal JSON), an empty terminal
    ``think`` field, and one empty ``tool_think`` slot per tool-call assistant
    turn. A label-aware sub-agent (e.g. deepseek-v4-flash) fills both fields in
    its reserved shard. ``export_tool_trajectory_dataset`` with
    ``think_source=file:<collect_dir>`` re-derives the same contexts, reads
    the filled thoughts by ``sample_id``, and validates the merged release.
    Legacy producers may fill only ``think``; ``FileThinkGenerator`` keeps
    that format compatible while still injecting a thought into each turn.
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
                "assistant_turns": len(context["tool_calls"]) + 1,
                # ``think`` remains the terminal slot for backwards
                # compatibility; tool-call assistant turns have explicit
                # ordered slots so a label-aware generator can fill every
                # assistant turn independently.
                "think": "",
                "tool_think": ["" for _ in context["tool_calls"]],
            }
        )

    output_root.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "format": "verl_tool_trajectory_think_collect_v2",
        "dataset": dataset,
        "label_aware": True,
        "instruction": (
            "fill think (terminal) and tool_think (one entry per tool-call "
            "assistant turn) with one or two short reasoning sentences "
            "consistent with the given terminal_json/ground_truth; each "
            "tool_think entry is capped at 64 tokens and terminal think at "
            "128; never emit a terminal JSON object or \"answer\"/\"level\" "
            "key-value lookalikes inside think"
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
            "tool_max_tokens": min(_TOOL_MAX_THINK_TOKENS, max_think_tokens),
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
        "think_truncated_turns": 0,
        "think_discarded": 0,
    }

    split_ground_truths: dict[str, set[str]] = {name: set() for name in SPLITS}
    split_levels: dict[str, set[str]] = {name: set() for name in SPLITS}
    all_rows: dict[str, list[dict[str, Any]]] = {}
    think_truncated_total = 0
    think_truncated_turns_total = 0
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
        truncated_think_turns = 0
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
            truncated_think_turns += int(row.get("think_truncated_turns", 0) or 0)
            rows.append(row)
        if not rows:
            raise ValueError(f"no resolved records available in {canonical_path}")
        all_rows[split] = rows
        think_truncated_total += truncated_think
        think_truncated_turns_total += truncated_think_turns
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
            "think_truncated_turns": truncated_think_turns,
            "trajectory_class_counts": class_counts,
        }
        for name, count in class_counts.items():
            report["trajectory_class_counts"][name] = (
                report["trajectory_class_counts"].get(name, 0) + count
            )
    report["think_truncated"] = think_truncated_total
    report["think_truncated_turns"] = think_truncated_turns_total
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
    # 2026-08-27): every assistant turn must carry non-empty, one/two-sentence
    # reasoning within its per-turn cap, with balanced brackets, no terminal
    # JSON object and no `"answer"`/`"level"` key-value lookalikes
    # (answer_mask doppelgangers). Mock think stays ground-truth-free; file
    # think (the label-aware sub-agent seam) may contain bare ground-truth ids
    # as legitimate reasoning.
    think_source = row.get("think_source")
    if not isinstance(think_source, str) or not think_source.strip():
        errors.append("think_source must be a non-empty generator name")
    for message_index, message in enumerate(messages):
        reasoning = message.get("reasoning_content")
        if message.get("role") != "assistant":
            # Parquet list<struct> round-trips absent keys with None.  Any
            # actual thought on user/system/tool content is a schema leak.
            if reasoning is not None:
                errors.append(
                    f"message {message_index} non-assistant must not carry reasoning_content"
                )
            continue
        is_terminal = message is messages[-1]
        cap = _think_budget(is_terminal, max_think_tokens)
        turn_name = "terminal" if is_terminal else "tool"
        if not isinstance(reasoning, str) or not reasoning.strip():
            errors.append(
                f"{turn_name} assistant message must carry non-empty reasoning_content"
            )
            continue
        if estimate_think_tokens(reasoning) > cap:
            if is_terminal:
                errors.append(
                    f"reasoning_content exceeds max_think_tokens={max_think_tokens}"
                )
            else:
                errors.append(
                    f"tool reasoning_content exceeds {cap}-token cap"
                )
        if _think_sentence_count(reasoning) > 2:
            errors.append(
                f"{turn_name} reasoning_content must contain at most two sentences"
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

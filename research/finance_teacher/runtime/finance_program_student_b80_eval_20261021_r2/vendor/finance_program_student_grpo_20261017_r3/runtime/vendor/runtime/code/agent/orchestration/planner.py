"""Prompt construction and model-independent Program planning adapters."""

from __future__ import annotations

import json
from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

from .catalog import SkillSpec, default_catalog
from .contracts import MetadataRecord, Program
from .validator import DEFAULT_MAX_STEPS, ValidationIssue, validate_program


PLANNER_SYSTEM_PROMPT = """你是数据库字段分类分级任务的 DSL Program Planner。
你的职责是根据当前字段元数据、已有状态和可用 Skill Catalog，生成一个可执行的 Program。
只允许使用 Catalog 中出现的 Skill；只能引用当前真实存在的元数据或前一步已经产生的状态。
引用语法必须严格遵守：`$record` 表示完整记录，`$domain` 表示域名，`$metadata.<name>` 表示元数据字段，`$state.<name>` 表示前一步产生的状态。
禁止使用 `$metadata`、`$record.metadata.<name>`、`$record.domain` 或其他未列出的路径；这些写法即使看起来等价，也不是合法 DSL 引用。
不要补造缺失的表名、描述或样例值。Program 必须以 FINALIZE_CLASSIFICATION 结束。
对于前一步 Skill 产生的结果，后续参数必须使用对应的 `$state.<name>` 引用，不能把空数组、空对象或模型臆造的结果直接填入必需参数。
Program 最多 10 步；如果已有信息足够，可以采用下面这个合法的最小骨架：
{"program":[
  {"tool":"INTERPRET_FIELD_NAME","arguments":{"field_name":"$metadata.field_name"}},
  {"tool":"SEARCH_STANDARD","arguments":{"semantic_query":"$state.field_semantics"}},
  {"tool":"INFER_CATEGORY","arguments":{"candidate_standard_categories":"$state.standard_candidates","all_available_evidence":"$state.field_semantics"}},
  {"tool":"INFER_GRADING_LEVEL","arguments":{"aligned_category":"$state.category","aligned_standard_entry":"$state.aligned_standard_entry"}},
  {"tool":"FINALIZE_CLASSIFICATION","arguments":{"final_evidence_state":"$state.final_evidence_state"}}
]}
只输出一个完整 JSON 对象，不要输出 Markdown、解释或额外文字，不能省略括号。"""


TWO_STAGE_PLANNER_SYSTEM_PROMPT = """你是数据库字段分类分级任务的 DSL Program Planner。
每个 Program 步骤只包含一次 Tool Call；只能调用本轮 Skill Catalog 中的工具。
仅使用可见的 domain 与 field_name，缺少其他元数据时通过可用的元数据 Skill 查询，不能编造。
引用语法只允许 `$record`、`$domain`、`$metadata.<name>`、`$state.<name>`。
先调用 SELECT_CATEGORY_TOPK 从完整类别路径目录选候选 ID，再调用 INFER_CATEGORY_FROM_TOPK 从这些候选中选一个类别。
第二步的 candidate_standard_categories 必须引用 `$state.standard_candidates`，不能自己填写候选。
只有 Top-1 类别确定后才调用 INFER_GRADING_LEVEL，最后必须调用 FINALIZE_CLASSIFICATION。
合法的最小骨架：
{"program":[
  {"tool":"INTERPRET_FIELD_NAME","arguments":{"field_name":"$metadata.field_name"}},
  {"tool":"SELECT_CATEGORY_TOPK","arguments":{"semantic_query":"$state.field_semantics"}},
  {"tool":"INFER_CATEGORY_FROM_TOPK","arguments":{"candidate_standard_categories":"$state.standard_candidates","all_available_evidence":"$state.field_semantics"}},
  {"tool":"INFER_GRADING_LEVEL","arguments":{"aligned_category":"$state.category","aligned_standard_entry":"$state.aligned_standard_entry"}},
  {"tool":"FINALIZE_CLASSIFICATION","arguments":{"final_evidence_state":"$state.grading_level"}}
]}
只输出一个完整 JSON 对象，不输出解释或 Markdown。"""


class PlannerError(ValueError):
    """Raised when generation, JSON parsing, or Program validation fails."""

    def __init__(
        self,
        message: str,
        *,
        raw_response: str | None = None,
        issues: Sequence[ValidationIssue] = (),
    ) -> None:
        super().__init__(message)
        self.raw_response = raw_response
        self.issues = tuple(issues)


@dataclass(frozen=True)
class PlannerResult:
    """A validated Program together with the exact prompt and model response."""

    program: Program
    messages: tuple[dict[str, str], ...]
    raw_response: str


def _catalog_payload(catalog: Mapping[str, SkillSpec]) -> list[dict[str, Any]]:
    return [
        {
            "name": spec.name,
            "role": spec.role,
            "required_inputs": list(spec.required_inputs),
            "optional_inputs": list(spec.optional_inputs),
            "produces": list(spec.produces),
            "max_calls": spec.max_calls,
            "terminal": spec.terminal,
        }
        for spec in catalog.values()
    ]


def build_planner_messages(
    record: MetadataRecord,
    catalog: Mapping[str, SkillSpec] | None = None,
    *,
    state: Mapping[str, Any] | None = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    two_stage: bool = False,
) -> list[dict[str, str]]:
    """Build deterministic chat messages for a model that generates a Program."""

    active_catalog = catalog or default_catalog()
    available_metadata = [
        name
        for name in ("domain", *record.metadata.keys())
        if record.available(name)
    ]
    payload = {
        "task": "为当前数据库字段选择必要的元数据、语义、标准对齐和证据推理 Skill，并生成完整 Program。",
        "record": (
            {"domain": record.domain, "metadata": dict(record.metadata)}
            if two_stage else record.to_dict()
        ),
        "available_metadata": available_metadata,
        "state": dict(state or {}),
        "skill_catalog": _catalog_payload(active_catalog),
        "constraints": {
            "max_steps": max_steps,
            "final_skill": "FINALIZE_CLASSIFICATION",
            "output_schema": {"program": [{"tool": "SKILL_NAME", "arguments": {}}]},
        },
    }
    return [
        {"role": "system", "content": (
            TWO_STAGE_PLANNER_SYSTEM_PROMPT if two_stage else PLANNER_SYSTEM_PROMPT
        )},
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False, indent=2),
        },
    ]


def _decode_single_json_object(text: str) -> Mapping[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    start = cleaned.find("{")
    if start < 0:
        raise PlannerError("model response does not contain a JSON object", raw_response=text)

    decoder = json.JSONDecoder()
    try:
        value, end = decoder.raw_decode(cleaned[start:])
    except JSONDecodeError as exc:
        raise PlannerError(f"invalid JSON response: {exc.msg}", raw_response=text) from exc

    trailing = cleaned[start + end :].strip()
    if trailing:
        if trailing.startswith("{"):
            message = "model response contains multiple JSON objects"
        else:
            message = "model response contains trailing non-JSON content"
        raise PlannerError(message, raw_response=text)
    if not isinstance(value, Mapping):
        raise PlannerError("model response JSON must be an object", raw_response=text)
    return value


def parse_program_response(text: str) -> Program:
    """Parse one model response into a Program without silently repairing it."""

    if not isinstance(text, str) or not text.strip():
        raise PlannerError("model response must be a non-empty string", raw_response=text if isinstance(text, str) else None)
    payload = _decode_single_json_object(text)
    try:
        return Program.from_mapping(payload)
    except (TypeError, ValueError) as exc:
        raise PlannerError(f"invalid Program schema: {exc}", raw_response=text) from exc


class PromptPlanner:
    """Adapt any text generator to the validated DSL Program contract."""

    def __init__(
        self,
        generate: Callable[[list[dict[str, str]]], str],
        *,
        catalog: Mapping[str, SkillSpec] | None = None,
        max_steps: int = DEFAULT_MAX_STEPS,
        max_retries: int = 1,
        two_stage: bool = False,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        self.generate = generate
        self.catalog = dict(catalog or default_catalog())
        self.max_steps = max_steps
        self.max_retries = max_retries
        self.two_stage = two_stage

    @staticmethod
    def _validation_feedback(issues: Sequence[ValidationIssue]) -> str:
        rendered = "\n".join(f"- {issue}" for issue in issues)
        return (
            "上一版 Program 未通过 DSL 校验，请修正后重新输出完整 JSON 对象。\n"
            "校验错误：\n"
            f"{rendered}\n"
            "只允许使用 `$record`、`$domain`、`$metadata.<name>` 和 `$state.<name>`；"
            "禁止使用 `$metadata` 或 `$record.metadata.<name>`。"
            "不要解释，不要输出 Markdown，只输出修正后的 JSON。"
        )

    def plan(
        self,
        record: MetadataRecord,
        *,
        state: Mapping[str, Any] | None = None,
    ) -> PlannerResult:
        messages = build_planner_messages(
            record,
            self.catalog,
            state=state,
            max_steps=self.max_steps,
            two_stage=self.two_stage,
        )
        attempt_messages = list(messages)
        for attempt in range(self.max_retries + 1):
            try:
                raw_response = self.generate(attempt_messages)
            except Exception as exc:  # pragma: no cover - depends on external generator
                raise PlannerError(f"planner generator failed: {exc}") from exc
            if not isinstance(raw_response, str):
                raise PlannerError("planner generator must return a string", raw_response=None)

            program = parse_program_response(raw_response)
            issues = validate_program(
                record,
                program,
                self.catalog,
                initial_state=state,
                max_steps=self.max_steps,
            )
            if not issues:
                return PlannerResult(
                    program=program,
                    messages=tuple(attempt_messages),
                    raw_response=raw_response,
                )

            if attempt == self.max_retries:
                rendered = "; ".join(str(issue) for issue in issues)
                raise PlannerError(
                    f"generated Program failed validation: {rendered}",
                    raw_response=raw_response,
                    issues=issues,
                )

            attempt_messages.extend(
                [
                    {"role": "assistant", "content": raw_response},
                    {"role": "user", "content": self._validation_feedback(issues)},
                ]
            )

        raise AssertionError("unreachable")


class TransformersTextGenerator:
    """Optional local Transformers backend; imports model dependencies lazily."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        max_new_tokens: int = 512,
        local_files_only: bool = True,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        trust_remote_code: bool = False,
        enable_thinking: bool = False,
    ) -> None:
        self.model_path = str(Path(model_path).expanduser())
        self.max_new_tokens = max_new_tokens
        self.local_files_only = local_files_only
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.trust_remote_code = trust_remote_code
        self.enable_thinking = enable_thinking
        self._tokenizer = None
        self._model = None
        self._torch = None
        self.last_generation_metadata: dict[str, Any] = {}

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import (
                AutoConfig,
                AutoModelForCausalLM,
                AutoModelForImageTextToText,
                AutoTokenizer,
            )
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise PlannerError(
                "TransformersTextGenerator requires torch and transformers; "
                "install them in the remote environment first"
            ) from exc
        self._torch = torch

        dtype: Any = self.torch_dtype
        if dtype != "auto":
            dtype = getattr(torch, dtype)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=self.local_files_only,
            trust_remote_code=self.trust_remote_code,
        )
        model_config = AutoConfig.from_pretrained(
            self.model_path,
            local_files_only=self.local_files_only,
            trust_remote_code=self.trust_remote_code,
        )
        auto_model = (
            AutoModelForImageTextToText
            if select_transformers_auto_model(model_config.to_dict()) == "image-text-to-text"
            else AutoModelForCausalLM
        )
        self._model = auto_model.from_pretrained(
            self.model_path,
            local_files_only=self.local_files_only,
            device_map=self.device_map,
            dtype=dtype,
            trust_remote_code=self.trust_remote_code,
        )
        self._model.eval()

    def __call__(self, messages: list[dict[str, str]]) -> str:
        self._load()
        assert self._tokenizer is not None
        assert self._model is not None
        tokenizer = self._tokenizer
        model = self._model
        if not hasattr(tokenizer, "apply_chat_template"):
            raise PlannerError("tokenizer does not provide apply_chat_template")
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
        except TypeError:
            # Generic chat templates may not expose Qwen's thinking switch.
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        inputs = tokenizer(prompt, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {name: value.to(device) for name, value in inputs.items()}
        prompt_length = inputs["input_ids"].shape[-1]
        assert self._torch is not None
        started = perf_counter()
        with self._torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        new_tokens = generated[0][prompt_length:]
        completion_tokens = int(new_tokens.shape[-1])
        self.last_generation_metadata = {
            "token_usage": {
                "prompt_tokens": int(prompt_length),
                "completion_tokens": completion_tokens,
                "total_tokens": int(prompt_length) + completion_tokens,
            },
            "elapsed_seconds": perf_counter() - started,
        }
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def select_transformers_auto_model(config: Mapping[str, Any]) -> str:
    """Choose the compatible Transformers auto class from lightweight config data."""

    architectures = config.get("architectures")
    names = architectures if isinstance(architectures, list) else []
    model_type = str(config.get("model_type") or "")
    if model_type == "qwen3_5" or any(
        str(name).endswith("ForConditionalGeneration") for name in names
    ):
        return "image-text-to-text"
    return "causal-lm"


__all__ = [
    "PLANNER_SYSTEM_PROMPT",
    "PlannerError",
    "PlannerResult",
    "PromptPlanner",
    "TransformersTextGenerator",
    "build_planner_messages",
    "parse_program_response",
    "select_transformers_auto_model",
]

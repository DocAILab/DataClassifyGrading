"""First-version Skill catalog used by the local planner/validator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class SkillSpec:
    name: str
    role: str
    required_inputs: tuple[str, ...]
    optional_inputs: tuple[str, ...] = ()
    produces: tuple[str, ...] = ()
    max_calls: int = 1
    terminal: bool = False


def default_catalog() -> dict[str, SkillSpec]:
    """Return a fresh ordered catalog for the first local experiment."""

    specs = (
        SkillSpec("CHECK_METADATA_AVAILABILITY", "metadata_acquisition", ("current_record",), produces=("metadata_availability",)),
        SkillSpec("GET_TABLE_FIELDS", "metadata_acquisition", ("table_name",), produces=("table_fields",)),
        SkillSpec("SELECT_CONTEXT_FIELDS", "metadata_acquisition", ("target_field", "table_fields"), optional_inputs=("current_semantic_state",), produces=("selected_context_fields",)),
        SkillSpec("INTERPRET_DOMAIN", "semantic_interpretation", ("domain",), produces=("domain_semantics",)),
        SkillSpec("EXPAND_FIELD_ABBREVIATION", "semantic_interpretation", ("field_name",), optional_inputs=("domain",), produces=("field_abbreviation_semantics",)),
        SkillSpec("INTERPRET_FIELD_NAME", "semantic_interpretation", ("field_name",), optional_inputs=("domain", "field_abbreviation_semantics"), produces=("field_semantics",)),
        SkillSpec("DISAMBIGUATE_FIELD_SEMANTICS", "semantic_interpretation", ("field_semantics",), optional_inputs=("domain", "standard_candidates"), produces=("disambiguated_field_semantics",)),
        SkillSpec("INTERPRET_FIELD_DESCRIPTION", "semantic_interpretation", ("field_description",), optional_inputs=("field_name", "domain"), produces=("field_description_semantics",)),
        SkillSpec("INTERPRET_TABLE_NAME", "semantic_interpretation", ("table_name",), optional_inputs=("domain", "table_description"), produces=("table_semantics",)),
        SkillSpec("INTERPRET_TABLE_DESCRIPTION", "semantic_interpretation", ("table_description",), optional_inputs=("table_name", "domain"), produces=("table_description_semantics",)),
        SkillSpec("INTERPRET_CONTEXT_FIELDS", "semantic_interpretation", ("context_fields",), optional_inputs=("domain", "table_name"), produces=("context_field_semantics",)),
        SkillSpec("SYNTHESIZE_METADATA_SEMANTICS", "semantic_interpretation", ("field_semantics",), optional_inputs=("domain_semantics", "table_semantics", "context_field_semantics", "field_description_semantics"), produces=("metadata_semantics",)),
        SkillSpec("SEARCH_STANDARD", "standard_alignment", ("semantic_query",), produces=("standard_candidates",)),
        SkillSpec("ALIGN_FIELD_TO_STANDARD", "standard_alignment", ("field_semantics", "standard_candidates"), produces=("field_standard_alignment",)),
        SkillSpec("COMPARE_STANDARD_CANDIDATES", "standard_alignment", ("standard_candidates", "semantic_evidence"), produces=("standard_comparison", "aligned_standard_entry")),
        SkillSpec("CHECK_EVIDENCE_SUFFICIENCY", "evidence_reasoning", ("current_evidence",), produces=("evidence_sufficiency",)),
        SkillSpec("INFER_CATEGORY", "evidence_reasoning", ("candidate_standard_categories", "all_available_evidence"), produces=("category", "aligned_standard_entry")),
        SkillSpec("INFER_GRADING_LEVEL", "evidence_reasoning", ("aligned_category", "aligned_standard_entry"), produces=("grading_level",)),
        SkillSpec("FINALIZE_CLASSIFICATION", "evidence_reasoning", ("final_evidence_state",), produces=("final_result",), terminal=True),
    )
    return {spec.name: spec for spec in specs}


DYNAMIC_LOOP_SKILLS = (
    "CHECK_METADATA_AVAILABILITY",
    "INTERPRET_DOMAIN",
    "EXPAND_FIELD_ABBREVIATION",
    "INTERPRET_FIELD_NAME",
    "DISAMBIGUATE_FIELD_SEMANTICS",
    "SEARCH_STANDARD",
    "COMPARE_STANDARD_CANDIDATES",
    "CHECK_EVIDENCE_SUFFICIENCY",
    "INFER_CATEGORY",
    "INFER_GRADING_LEVEL",
    "FINALIZE_CLASSIFICATION",
)


def dynamic_loop_catalog(
    *, include_field_metadata: bool = False,
    include_table_fields: bool = False,
    include_description_skills: bool = False,
    two_stage: bool = False,
) -> dict[str, SkillSpec]:
    """Return the approved ordered Skill subset for the dynamic pilot."""

    catalog = default_catalog()
    names = tuple(
        "SELECT_CATEGORY_TOPK" if name == "SEARCH_STANDARD" else
        "INFER_CATEGORY_FROM_TOPK" if name == "INFER_CATEGORY" else name
        for name in DYNAMIC_LOOP_SKILLS
    ) if two_stage else DYNAMIC_LOOP_SKILLS
    if two_stage:
        catalog.update({
            "SELECT_CATEGORY_TOPK": SkillSpec(
                "SELECT_CATEGORY_TOPK", "standard_alignment", ("semantic_query",),
                produces=("standard_candidates",),
            ),
            "INFER_CATEGORY_FROM_TOPK": SkillSpec(
                "INFER_CATEGORY_FROM_TOPK", "evidence_reasoning",
                ("candidate_standard_categories", "all_available_evidence"),
                produces=("category", "aligned_standard_entry"),
            ),
        })
    selected = {name: catalog[name] for name in names}
    if include_field_metadata:
        selected["GET_FIELD_METADATA"] = SkillSpec(
            "GET_FIELD_METADATA", "metadata_acquisition", ("current_record",),
            produces=("field_metadata",),
        )
    if include_table_fields:
        selected["GET_TABLE_FIELDS"] = SkillSpec(
            "GET_TABLE_FIELDS", "metadata_acquisition", ("current_record",),
            produces=("table_fields",),
        )
    if include_description_skills:
        for name in ("INTERPRET_FIELD_DESCRIPTION", "INTERPRET_TABLE_DESCRIPTION"):
            selected[name] = catalog[name]
    return selected


__all__ = ["DYNAMIC_LOOP_SKILLS", "SkillSpec", "default_catalog", "dynamic_loop_catalog"]

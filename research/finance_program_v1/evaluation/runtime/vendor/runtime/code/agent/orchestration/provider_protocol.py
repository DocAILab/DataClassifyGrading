"""One provider response schema, distinct from canonical state publications."""

import json

RETRIEVAL = "SELECT_CATEGORY_TOPK"
DECISION = "INFER_CATEGORY_FROM_TOPK"


def output_schema(stage: str, top_k: int = 5) -> dict:
    summary = {
        "type": "string",
        "minLength": 1,
        "description": "brief evidence summary",
    }
    if stage == RETRIEVAL:
        properties = {
            "option_ids": {
                "type": "array",
                "minItems": top_k,
                "maxItems": top_k,
                "uniqueItems": True,
                "items": {"type": "string"},
                "description": "ranked distinct short IDs copied from supplied category_options",
            },
            "summary": summary,
        }
    elif stage == DECISION:
        properties = {
            "option_id": {
                "type": ["string", "null"],
                "description": "one short ID from actual candidate_details, or null if evidence does not support any candidate",
            },
            "summary": summary,
        }
    else:
        raise ValueError("unknown provider protocol stage")
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def expected_keys(stage: str) -> set[str]:
    return set(output_schema(stage)["properties"])


def output_rule(stage: str, top_k: int) -> str:
    schema = output_schema(stage, top_k)
    return (
        "Return exactly one JSON object matching output_schema: "
        + json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
        + ". Copy only supplied short option IDs; do not emit canonical paths or objects, "
        "a data wrapper, Markdown, duplicate fields, or an incomplete JSON object. "
        "Keep summary brief. Ranking candidates is not a claim all candidates match. "
        "For classification, use option_id=null and state the actual evidence gap in summary "
        "when none is supported; do not invent support. Null is a semantic insufficient-evidence "
        "outcome, not a malformed answer to retry unchanged."
    )


def contract_body(purpose: str) -> str:
    return (
        "# Role\n" + purpose + "\n\n# Provider and publication boundary\n"
        "Use only actual resolved inputs and the supplied standard catalog. The executor supplies "
        "the single authoritative provider output_schema. Return only that provider JSON; "
        "do not return a data wrapper or canonical category/standard objects. The native header's "
        "produces list describes validated code publications, not the provider response shape. "
        "The executor maps accepted short IDs into the frozen catalog; a rank is not proof of "
        "semantic support. Unsupported classification must use the schema's explicit null outcome. "
        "Never invent metadata, candidate IDs, labels, grades or sufficient evidence.\n"
    )

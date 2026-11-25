"""Versioned teacher changes for complete table reads, field search and classification."""
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from unittest.mock import patch

import scope_runtime


CHANGES = ("context", "retrieval", "decision")


def identity(changes=CHANGES):
    changes = sorted(changes)
    if not changes or len(set(changes)) != len(changes) or set(changes) - set(CHANGES):
        raise ValueError("Choose distinct supported evidence changes")
    return {"version": "followup-evidence-v1", "changes": changes,
            "sha256": sha256(Path(__file__).read_bytes()).hexdigest()}


def install_complete_context(controller):
    lookup = controller.bridge.executor.metadata_lookup
    original = lookup.table_context

    def table_context(record):
        result = original(record)
        if not result["truncated"]:
            return result
        fields = result["fields"]
        seen = {record.field_name, *(row["field_name"] for row in fields)}
        for row in lookup._same_table(record):
            if row["field_name"] in seen:
                continue
            seen.add(row["field_name"])
            fields.append({k: deepcopy(row[k]) for k in ("field_name", "field_description", "field_type", "value")
                           if k in row})
        if len(fields) != result["available_count"]:
            raise ValueError("Complete context differs from the scope of the original table read")
        result.update(returned_count=len(fields), omitted_count=0, truncated=False)
        return result

    lookup.table_context = table_context
    return controller


def project(messages, changes, registry):
    stage = messages[0]["content"].splitlines()[0].removeprefix("Execute only the selected Skill ").removesuffix(".")
    if "retrieval" in changes and stage.startswith("SEARCH_STANDARD_"):
        result = deepcopy(messages)
        payload = json.loads(result[-1]["content"])
        domain = payload["record_input"]["domain"]
        definitions = {" > ".join(v["path"]): v["description"] for k, v in registry.items()
                       if k.startswith(domain + ":")}
        payload["category_definition_index"] = [
            {**entry, "description": definitions[entry["label"]]} for entry in payload["category_options"]]
        result[-1]["content"] = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        result[0]["content"] += (
            "\nUse the supplied public definitions to rank plausible alternatives. Consider both what the "
            "column denotes in isolation and what a row records. When they suggest different scopes, "
            "compare both against the acquired same-table facts before pruning to K candidates. "
            "Definitions describe the options and add no evidence about this record.")
        return result
    if "decision" in changes and stage in ("INFER_CATEGORY", "INFER_CATEGORY_FROM_TOPK"):
        result = deepcopy(messages)
        result[0]["content"] += (
            "\nBefore deciding, contrast the two best supported candidates. Identify the target field's "
            "role in one row. Examine evidence for what the entity is and evidence for the activity or "
            "relationship the row records, including facts that favor the alternative. Cite the facts "
            "that distinguish the target's role. Entity identifiers alone do not settle the activity, "
            "and activities in other columns alone do not settle the target's role. If the distinction "
            "remains unsupported, state the missing link using the existing insufficient-evidence output.")
        return result
    return messages


@contextmanager
def use(runtime, record, registry):
    if record is None:  # Replay the protocol recorded before these changes.
        yield
        return
    if record != identity(record["changes"]):
        raise ValueError("Unexpected follow-up evidence protocol")
    changes = record["changes"]
    with ExitStack() as stack:
        if "context" in changes:
            bind = scope_runtime.bind_final_provider

            def bind_context(rollout, install):
                return bind(rollout, lambda c: install_complete_context(install(c)))

            stack.enter_context(patch.object(scope_runtime, "bind_final_provider", bind_context))
        if "retrieval" in changes or "decision" in changes:
            class Actor(runtime.CommonActor):
                def project(self, messages):
                    view = super().project(messages)
                    shared = project(view["messages"], changes, registry)
                    return {**view, "messages": shared, "input_sha256": runtime.previous.message_sha(shared)}

            stack.enter_context(patch.object(runtime, "CommonActor", Actor))
        yield

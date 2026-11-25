"""Regression tests for complete table context and stage-local prompt changes."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from smoke import science
import evidence_followup as update
import table_recovery


class FollowupEvidenceTest(unittest.TestCase):
    def test_complete_context_uses_same_scope_and_preserves_the_first_read(self):
        provider = science.runtime().coverage_pool.scope.SourceScopedMetadata
        rows, scopes = [], []
        for table, group, count in [("target", "a", 67), ("other", "b", 2)]:
            for i in range(count):
                name = f"field-{i:03d}"
                meta = {"database_name": "", "table_name": table, "field_name": name,
                        "field_description": f"{table}-{i}"}
                row = {"id": f"{group}-{i}", "domain": "finance", "metadata": meta}
                rows.append(row)
                scopes.append({**row, "source_table_name": table, "source_table_id": group, "source_database_name": ""})
        lookup = provider(rows, scopes)
        rec = SimpleNamespace(source_id="a-0", domain="finance", field_name="field-000")
        initial = lookup.table_fields(rec)
        self.assertEqual(initial["returned_count"], 30)
        self.assertEqual(lookup.table_context(rec)["fields"], initial["fields"])
        update.install_complete_context(SimpleNamespace(bridge=SimpleNamespace(executor=SimpleNamespace(metadata_lookup=lookup))))
        full = lookup.table_context(rec)
        self.assertEqual(full["fields"][:30], initial["fields"])
        self.assertEqual([r["field_description"] for r in full["fields"]], [f"target-{i}" for i in range(1, 67)])
        self.assertEqual((full["returned_count"], full["available_count"], full["omitted_count"], full["truncated"]),
                         (66, 66, 0, False))
        self.assertEqual(lookup.table_fields(rec), initial)
        self.assertEqual(lookup.table_context(rec), full)
        small = SimpleNamespace(source_id="b-0", domain="finance", field_name="field-000")
        self.assertEqual(lookup.table_context(small)["fields"], lookup.table_fields(small)["fields"])

    def test_prompt_changes_are_scoped_and_preserve_the_source(self):
        entry = {"option_id": "A", "label": "a > b"}
        payload = {"category_options": [entry], "record_input": {"domain": "finance"}, "evidence_boundary": {"facts": {}}}
        catalog = {"finance:a/b": {"path": ["a", "b"], "description": "public definition", "grading_level": "private"}}
        messages = [{"content": "Execute only the selected Skill SEARCH_STANDARD_BY_FIELD."},
                    {"content": json.dumps(payload)}]
        self.assertIs(update.project(messages, ["decision"], catalog), messages)
        enriched = json.loads(update.project(messages, ["retrieval"], catalog)[-1]["content"])
        self.assertEqual(enriched.pop("category_definition_index"), [{**entry, "description": "public definition"}])
        self.assertEqual(enriched, payload)
        messages[0]["content"] = "Execute only the selected Skill INFER_CATEGORY."
        self.assertIs(update.project(messages, ["retrieval"], catalog), messages)
        compared = update.project(messages, ["decision"], catalog)
        self.assertEqual(compared[-1], messages[-1])
        self.assertNotEqual(compared[0], messages[0])

    def test_planner_sees_unread_count_only_after_a_gap(self):
        messages = [{"content": "You are the Global Program Planner for a database-field classification task."}]
        facts = {"observations.table_fields": {"value": {"truncated": True, "omitted_count": 4,
                                                         "returned_count": 30, "available_count": 34}}}
        state = {"evidence_gap_ledger": [], "evidence_boundary": {"facts": facts}}
        def projected():
            return table_recovery.project(messages, [{"content": json.dumps(state)}])
        self.assertIs(projected(), messages)
        state["evidence_gap_ledger"] = [{"missing_evidence": "unclear"}]
        self.assertIn("4 scoped columns are still unread", projected()[0]["content"])
        facts["observations.table_context"] = {"value": {"truncated": False}}
        self.assertIn("already been returned", projected()[0]["content"])

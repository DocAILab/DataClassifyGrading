import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import row_evidence
import field_roles


class RowEvidenceTest(unittest.TestCase):
    def test_only_distinct_citable_descriptions_are_indexed(self):
        def fact(value, eligible=True, role="raw_fact"):
            return {"role": role, "decisive_eligible": eligible, "value": value}

        facts = {
            "metadata.field_description": fact("id"),
            "metadata.table_description": fact("catalog"),
            "observations.table_fields.fields.0.field_description": fact("status"),
            "observations.table_context.fields.1.field_description": fact("status"),
            "observations.table_context.fields.2.field_description": fact("id"),
            "observations.table_context.fields.3.field_description": fact("date"),
            "observations.table_context.fields.4.field_description": fact("hidden", False),
            "observations.table_fields.fields.5.field_description": fact("inference", role="summary"),
            "observations.other_table.fields.0.field_description": fact("out of scope"),
        }
        view = row_evidence.evidence_index(facts)
        self.assertEqual(view["target"]["quote"], "id")
        self.assertEqual(view["table"]["quote"], "catalog")
        self.assertEqual([x["quote"] for x in view["other_distinct_descriptions"]], ["status", "date"])
        for entry in view["other_distinct_descriptions"]:
            self.assertEqual(facts[entry["fact_id"]]["value"], entry["quote"])
        del facts["metadata.field_description"]
        self.assertIsNone(row_evidence.evidence_index(facts))

    def test_no_projection_without_siblings_or_outside_classification(self):
        messages = [{"content": "Execute only the selected Skill INFER_CATEGORY."},
                    {"content": json.dumps({"evidence_boundary": {"facts": {}}})}]
        self.assertIs(row_evidence.project(messages), messages)
        messages[0]["content"] = "Execute only the selected Skill SEARCH_STANDARD_BY_FIELD."
        self.assertIs(row_evidence.project(messages), messages)

    def test_qualified_attributes_keep_both_roles_and_copy_only_when_needed(self):
        index = {"target": {"fact_id": "target", "quote": "投保人联系地址"},
                 "other_distinct_descriptions": [
                     {"fact_id": "other", "quote": "代理人联系地址"},
                     {"fact_id": "date", "quote": "创建日期"}]}
        messages = [{"content": "Execute only the selected Skill INFER_CATEGORY."},
                    {"content": json.dumps({"row_evidence_index": index})}]
        revised = json.loads(field_roles.project(messages)[-1]["content"])
        self.assertEqual(revised.pop("contrasting_field_wording"),
                         [{"target_fact_id": "target", "other_fact_id": "other", "shared_suffix": "人联系地址",
                           "target_prefix": "投保", "other_prefix": "代理"}])
        self.assertEqual(revised, json.loads(messages[-1]["content"]))
        index["other_distinct_descriptions"].pop(0)
        messages[-1]["content"] = json.dumps({"row_evidence_index": index})
        self.assertIs(field_roles.project(messages), messages)


if __name__ == "__main__":
    unittest.main()

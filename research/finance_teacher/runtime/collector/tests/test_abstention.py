import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from abstention import decision, verified_abstention
from role_evidence import project
from review_abstention import suggestion
from smoke import common, save_decision


class DecisionTest(unittest.TestCase):
    def setUp(self):
        self.registry = {"cat-a": {"grading_level": "L1"}, "cat-b": {"grading_level": "L3"}}
        active = [{"label": "B", "category": "cat-b"}, {"label": "A", "category": "cat-a"}]
        catalog = [{"option_id": "01", "label": "A"}, {"option_id": "02", "label": "B"}]
        self.step = {"status": "success", "call": {"tool": "SELECT_CATEGORY_TOPK"}, "lineage": {"candidates": 1},
                     "data": {"standard_candidates": active}, "summary": "observed relationship",
                     "metadata": {"raw_response": json.dumps({"option_ids": ["02", "01"]}),
                                  "messages": [{"content": json.dumps({"category_options": catalog})}]}}
        self.trace = {"stop_reason": "unresolved_no_new_business_evidence", "terminated": False,
                      "gap_ledger": [{"missing_evidence": "unknown owner"}], "final_lineage": {"candidates": 1},
                      "state": {"_provenance": {"standard_candidates": "SELECT_CATEGORY_TOPK"},
                                "standard_candidates": active}, "programs": [{"steps": [self.step]}]}

    def test_model_order_and_registry_grade(self):
        result = decision(self.trace, self.registry, True)
        self.assertEqual(result["status"], "abstained")
        self.assertIsNone(result["prediction"])
        self.assertEqual(result["suggestion"]["category"], "cat-b")
        self.assertEqual(result["suggestion"]["grading_level"], "L3")
        self.assertEqual(result["missing_evidence"], "unknown owner")

    def test_no_stale_or_reordered_candidate(self):
        self.step["lineage"]["candidates"] = 0
        self.assertEqual(decision(self.trace, self.registry, True)["suggestion"]["method"], "catalog_tie")
        self.step["lineage"]["candidates"] = 1
        self.step["metadata"]["raw_response"] = json.dumps({"option_ids": ["01", "02"]})
        self.assertEqual(decision(self.trace, self.registry, True)["suggestion"]["method"], "catalog_tie")

    def test_abstention_without_rank_uses_observed_text_then_explicit_ties(self):
        self.trace["state"] = {"field_description": "债券利率"}
        self.registry["cat-b"]["description"] = "债券发行票面利率"
        result = decision(self.trace, self.registry, True)
        self.assertEqual(result["suggestion"]["category"], "cat-b")
        self.assertEqual(result["suggestion"]["method"], "catalog_text")
        self.assertIsNone(result["prediction"])
        self.trace["state"] = {}
        fallback = decision(self.trace, dict(reversed(list(self.registry.items()))), True)["suggestion"]
        self.assertEqual(fallback["method"], "catalog_tie")
        self.assertEqual(fallback["tied_candidates"], ["cat-a", "cat-b"])
        self.assertTrue(fallback["review_required"])

    def test_closed_abstention_can_survive_an_earlier_malformed_reply(self):
        self.trace["source_id"] = "same-source"
        row = {"native_replay_exact": True, "legal": False, "source_id": "same-source",
               "stop_reason": self.trace["stop_reason"]}
        summary = {"passed": True, "actual_OS_exit": 0, "completed": 1, "native_replays": 1,
                   "wire_complete_all": True, "same_information_exact": True, "unknown_usage": 0,
                   "raw_JSON_objects": 3, "actual_model_calls": 4}
        self.assertTrue(verified_abstention(self.trace, {"rows": [row], "summary": summary}))
        for change in ({"unknown_usage": 1}, {"wire_complete_all": False}, {"actual_OS_exit": 1}):
            self.assertFalse(verified_abstention(self.trace, {"rows": [row], "summary": {**summary, **change}}))

    def test_review_interruption_preserves_a_readable_answer_on_disk(self):
        local = decision(self.trace, self.registry, True)
        with TemporaryDirectory() as folder:
            target = Path(folder)
            common.put(target / "completion.json", {"budget": {"claims": 1, "request_cap": 2, "spent": 0,
                       "cost_cap": 1, "deadline": time.time() + 300}})
            with (patch("smoke.read_decision", return_value=local),
                  patch("smoke.review_abstention.run", side_effect=OSError("dropped connection"))):
                with self.assertRaises(OSError):
                    save_decision(target, True)
            self.assertEqual(common.read(target / "decision.json"), local)

    def test_review_resolves_only_real_citations_and_registered_options(self):
        facts = {"F01": {"source_id": "metadata.field_description", "value": "客户账户"}}
        answer = {"option_id": "02", "reason": "row subject", "uncertainty": "unknown owner", "fact_ids": ["F01"]}
        parse = lambda x: suggestion(json.dumps(x), {"02": "cat-b"}, self.registry, facts)
        self.assertEqual(parse(answer)["citations"], {"metadata.field_description": "客户账户"})
        self.assertEqual(parse(answer)["grading_level"], "L3")
        self.assertIsNone(parse({**answer, "option_id": "01"}))
        self.assertIsNone(parse({**answer, "fact_ids": ["F02"]}))

    def test_technical_failure_and_normal_classification(self):
        self.assertIsNone(decision(self.trace, self.registry, False)["suggestion"])
        self.trace["stop_reason"] = "campaign_budget_exhausted"
        self.assertEqual(decision(self.trace, self.registry, True)["status"], "incomplete")
        prediction = {"category": "cat-b", "grading_level": "L3"}
        self.trace.update(stop_reason="validated_terminal", terminated=True, final_result=prediction)
        result = decision(self.trace, self.registry, True)
        self.assertEqual(result["prediction"], prediction)
        self.assertIsNone(result["suggestion"])

    def test_projection_uses_only_observed_same_qualifier(self):
        def fact(value):
            return {"role": "raw_fact", "decisive_eligible": True, "value": value}
        prefix = "observations.table_fields.fields."
        facts = {"metadata.field_description": fact("收款方-账户"),
                 prefix + "0.field_description": fact("收款方：户名"),
                 prefix + "1.field_description": fact("付款方-名称"),
                 prefix + "2.field_description": {**fact("收款方-行业"), "role": "hypothesis"}}
        m = [{"content": "Execute only the selected Skill INFER_CATEGORY."},
             {"content": json.dumps({"evidence_boundary": {"facts": facts}})}]
        view = json.loads(project(m)[-1]["content"])
        self.assertEqual(view["related_field_descriptions"]["same_table"], {prefix + "0.field_description": "收款方：户名"})
        self.assertEqual(view["evidence_boundary"]["facts"], facts)
        facts.pop(prefix + "0.field_description")
        m[-1]["content"] = json.dumps({"evidence_boundary": {"facts": facts}})
        self.assertIs(project(m), m)


if __name__ == "__main__":
    unittest.main()

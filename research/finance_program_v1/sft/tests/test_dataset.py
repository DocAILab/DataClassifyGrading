import copy
import hashlib
import json
import unittest

try:
    from export_dataset import extract_example, validate_selection
except ImportError:
    extract_example = validate_selection = None


def trace():
    p = {"program": [{"tool": "GET_FIELD_METADATA", "arguments": {}},
                     {"tool": "FINALIZE_CLASSIFICATION", "arguments": {}}]}
    payload = {"skill_library": [{"name": n} for n in ["GET_FIELD_METADATA","FINALIZE_CLASSIFICATION"]+["S%d" % i for i in range(64)]],
               "current": {"domain": "finance", "field_name": "X"}}
    return {"input": {"domain": "finance", "field_name": "X"}, "terminated": True,
            "planner_attempts": [{"status": "success", "messages": [
                {"role": "system", "content": "You are the Global Program Planner."},
                {"role": "user", "content": json.dumps(payload)}],
                "raw_response": json.dumps(p)}],
            "programs": [{"plan": p, "terminated": True, "status": "terminated"}]}


class DatasetTests(unittest.TestCase):
    def test_real_response_and_original_messages_only(self):
        self.assertIsNotNone(extract_example, "missing extraction feature")
        t = trace(); r = extract_example(t, "id1", "a" * 64)
        self.assertEqual(r["messages"], t["planner_attempts"][0]["messages"])
        self.assertEqual(r["response"], t["planner_attempts"][0]["raw_response"])
        self.assertEqual(r["kind"], "initial")
        self.assertNotIn("final_result", r)

    def test_reject_unterminated(self):
        self.assertIsNotNone(extract_example, "missing extraction feature")
        t=trace();t["terminated"]=False
        with self.assertRaises(ValueError):extract_example(t,"id1","a"*64)

    def test_reject_response_not_actual_plan(self):
        self.assertIsNotNone(extract_example, "missing extraction feature")
        t=trace();t["programs"][0]["plan"]["program"][0]["tool"]="OTHER"
        with self.assertRaises(ValueError):extract_example(t,"id1","a"*64)

    def test_reject_future_or_assistant_message(self):
        self.assertIsNotNone(extract_example, "missing extraction feature")
        t=trace();t["planner_attempts"][0]["messages"].append({"role":"tool","content":"future"})
        with self.assertRaises(ValueError):extract_example(t,"id1","a"*64)

    def test_reject_gold_in_message_payload(self):
        self.assertIsNotNone(extract_example, "missing extraction feature")
        t=trace();p=json.loads(t["planner_attempts"][0]["messages"][1]["content"]);p["gold"]="answer"
        t["planner_attempts"][0]["messages"][1]["content"]=json.dumps(p)
        with self.assertRaises(ValueError):extract_example(t,"id1","a"*64)

    def test_reject_incomplete_f66_library(self):
        self.assertIsNotNone(extract_example, "missing extraction feature")
        t=trace();p=json.loads(t["planner_attempts"][0]["messages"][1]["content"]);p["skill_library"].pop()
        t["planner_attempts"][0]["messages"][1]["content"]=json.dumps(p)
        with self.assertRaises(ValueError):extract_example(t,"id1","a"*64)

    def test_duplicate_selection_rejected(self):
        self.assertIsNotNone(validate_selection,"missing selection feature")
        with self.assertRaises(ValueError):validate_selection(["a","a"],["b"],expected=2)

    def test_reserve_overlap_rejected(self):
        self.assertIsNotNone(validate_selection,"missing selection feature")
        with self.assertRaises(ValueError):validate_selection(["a","b"],["b"],expected=2)

    def test_exact_count_and_no_extra_sources(self):
        self.assertIsNotNone(validate_selection,"missing selection feature")
        validate_selection(["a","b"],["c"],expected=2)
        with self.assertRaises(ValueError):validate_selection(["a"],["c"],expected=2)


if __name__ == '__main__':unittest.main()

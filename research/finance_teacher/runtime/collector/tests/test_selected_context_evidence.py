import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from selected_context_evidence import project


class SelectedContextTest(unittest.TestCase):
    def test_only_previously_disclosed_eligible_descriptions(self):
        def fact(value, scope="target_or_same_table_context", eligible=True):
            return dict(value=value, role="raw_fact", source_scope=scope, decisive_eligible=eligible)
        fid="observations.context_fields.1.field_description"
        facts={fid:fact("settlement date"),
               "observations.context_field_descriptions.fields.0.field_description":fact("settlement date"),
               "observations.context_fields.4.field_description":fact("hidden",eligible=False),
               "observations.context_fields.3.field_description":fact("other",scope="other_table"),
               "observations.context_fields.2.field_name":fact("name")}
        inp=[dict(role="system", content="Execute only the selected Skill INFER_CATEGORY."),
             dict(role="user", content=json.dumps({"evidence_boundary":{"facts":facts}}))]
        out=project(inp)
        self.assertEqual(json.loads(out[-1]["content"])["selected_field_evidence"],[{"fact_id":fid,"quote":"settlement date"}])
        self.assertNotIn("selected_field_evidence",inp[-1]["content"])

    def test_name_only_lookup_adds_no_details(self):
        inp=[dict(role="system",content="Execute only the selected Skill INFER_CATEGORY_FROM_TOPK."),
             dict(role="user",content='{"evidence_boundary":{"facts":{}}}')]
        self.assertIs(inp, project(inp))

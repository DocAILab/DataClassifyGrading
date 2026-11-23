"""CPU regression checks for the one public M9 path, using synthetic metadata."""
from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
import unittest

from episode import cohort, install_native, run_episode
from grpo import references, update
from model import Backend, TEMPERATURE
from native_rank_input import make_rank_messages, rank_reply
from native_assets import ROOT, prepare, read
from routing import adapter_route


class Scripted:
    def __init__(self, sufficient=True):
        self.registry = read(ROOT / "registry-12.json")
        self.sufficient = sufficient
        self.stages = []
        self.prompts = []

    def call(self, stage, messages, *, sample):
        self.stages.append(stage)
        self.prompts.append(messages)
        user = json.loads(messages[-1]["content"])
        if stage == "Planner":
            steps = ["GET_FIELD_METADATA", "INTERPRET_FIELD_NAME", "SEARCH_STANDARD_BY_FIELD",
                     "COMPARE_STANDARD_CANDIDATES", "INFER_CATEGORY", "INFER_GRADING_LEVEL",
                     "FINALIZE_CLASSIFICATION"]
            plan = [{"tool": t, "arguments": {"domain": "$domain"} if t == "INTERPRET_FIELD_NAME" else {}}
                    for t in steps]
            return json.dumps({"program": plan}), ({"toy_action": True} if sample else None)
        if stage == "INTERPRET_FIELD_NAME":
            semantic = {"schema": "grounded-source-v1", "status": "observed_description",
                        "field_name": "toy_field", "description": "synthetic bond coupon",
                        "source_consistency": "not_adjudicated"}
            answer = {"summary": "observed_description", "data": {"field_semantics": semantic}}
        elif stage.startswith("SEARCH_STANDARD_"):
            rank_prompt = make_rank_messages(messages, self.registry, read(ROOT / "prompt-templates.json"))
            assert json.loads(rank_prompt[1]["content"])["evidence_boundary"] == user["evidence_boundary"]
            return rank_reply([x["option_id"] for x in self.registry[:5]]), None
        elif stage == "COMPARE_STANDARD_CANDIDATES":
            answer = {"decision": "match", "option_id": "C001", "reason": "synthetic bond"}
        elif stage in ("INFER_CATEGORY", "INFER_CATEGORY_FROM_TOPK"):
            answer = {"option_id": "C001" if self.sufficient else "UNRESOLVED", "summary": "toy",
                      "support": {"sufficient": self.sufficient, "citations": [
                          {"fact_id": "metadata.field_description", "quote": "bond coupon"}
                      ] if self.sufficient else [], "missing_evidence": "" if self.sufficient else "subject unknown"}}
        else:
            raise AssertionError(stage)
        return json.dumps(answer), None


class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        root = Path(cls.tmp.name)
        cls.metadata = {"field_name": "toy_field", "field_description": "synthetic bond coupon"}
        cls.file = root / "public.json"
        cls.file.write_text(json.dumps({"split": "TRAIN", "rows": [
            {"source_id": "toy", "metadata": cls.metadata}]}), encoding="utf-8")
        cls.public = cohort(cls.file, "TRAIN")
        install_native(prepare(root / "dependencies", cls.public["scopes"]))

    def test_seven_step_program_shares_the_train_and_eval_route(self):
        case = self.public["inputs"]["cases"][0]
        evaluator, actor = Scripted(), Scripted()
        left, none = run_episode(self.public, case, evaluator)
        right, actions = run_episode(self.public, case, actor, sample=True)
        self.assertTrue(left["valid"], left)
        self.assertEqual(left, right)
        self.assertEqual(evaluator.prompts, actor.prompts)
        self.assertEqual(actor.stages, ["Planner", "INTERPRET_FIELD_NAME", "SEARCH_STANDARD_BY_FIELD",
                                       "COMPARE_STANDARD_CANDIDATES", "INFER_CATEGORY"])
        self.assertEqual(left["calls"], 7)
        self.assertEqual(none, [])
        self.assertEqual(actions, [{"toy_action": True}])

    def test_insufficient_evidence_abstains(self):
        result, _ = run_episode(self.public, self.public["inputs"]["cases"][0], Scripted(False))
        self.assertFalse(result["valid"])
        self.assertIsNone(result["prediction"])
        self.assertEqual(result["stop_reason"], "abstained_terminal")

    def test_split_and_labels_are_checked_before_rollout(self):
        with self.assertRaises(ValueError):
            cohort(self.file, "VAL")
        p = Path(self.tmp.name) / "labels-in-input.json"
        p.write_text(json.dumps({"split": "TRAIN", "rows": [
            {"source_id": "toy", "metadata": {**self.metadata, "category": "answer"}}]}))
        with self.assertRaises(ValueError):
            cohort(p, "TRAIN")
        p.write_text(json.dumps({"split": "VAL", "references": {}}))
        with self.assertRaises(ValueError):
            references(p, self.public, "TRAIN")


class PolicyTests(unittest.TestCase):
    def test_stage_routing(self):
        self.assertEqual(adapter_route("Planner"), "planner")
        for stage in ["INFER_CATEGORY", "INFER_CATEGORY_FROM_TOPK"]:
            self.assertEqual(adapter_route(stage), "skill_fixed")
        for stage in ["SEARCH_STANDARD_BY_FIELD", "COMPARE_STANDARD_CANDIDATES", "INTERPRET_FIELD_NAME"]:
            self.assertIsNone(adapter_route(stage))

    def test_update_moves_only_planner(self):
        import torch
        from types import SimpleNamespace
        from grpo import policy

        class ToyLayer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = torch.nn.Embedding(8, 8)
                self.layer = torch.nn.Module()
                self.layer.lora_A = torch.nn.ModuleDict({k: torch.nn.Linear(8, 1, bias=False)
                                                         for k in ["planner", "reference", "skill_fixed"]})
                for module in self.layer.lora_A.values():
                    torch.nn.init.zeros_(module.weight)
                self.active = "planner"

            def set_adapter(self, name):
                self.active = name
                self.layer.lora_A[name].requires_grad_(True)

            @contextmanager
            def disable_adapter(self):
                yield
                self.layer.lora_A[self.active].requires_grad_(True)

            def get_input_embeddings(self):
                return self.emb

            def forward(self, input_ids, logits_to_keep, **kwargs):
                logits = self.emb(input_ids) + self.layer.lora_A[self.active].weight
                return SimpleNamespace(logits=logits[:, -logits_to_keep:])

        torch.manual_seed(17)
        b = Backend(ToyLayer(), None)
        opt = torch.optim.AdamW(b.planner_parameters(), lr=.01, weight_decay=0.)
        before = {n: p.detach().clone() for n, p in b.model.named_parameters()}
        with b.route(None):
            self.assertFalse(any(p.requires_grad for p in b.model.parameters()))
        self.assertFalse(any(p.requires_grad for p in b.model.parameters()))
        group = []
        for reward, token in [(1, 2), (0, 3)]:
            with b.route("planner"), torch.no_grad():
                prob = policy.action_log_probs(b.model, [1], [token], TEMPERATURE).tolist()
            group.append((reward, [{"prompt": [1], "tokens": [token], "behavior": prob}]))
        self.assertFalse(update(b, opt, [[group[0], group[0]]]))
        self.assertTrue(update(b, opt, [group]))
        for n, p in b.model.named_parameters():
            self.assertEqual(torch.equal(before[n], p), ".planner." not in n)
            self.assertFalse(p.requires_grad)


if __name__ == "__main__":
    unittest.main()

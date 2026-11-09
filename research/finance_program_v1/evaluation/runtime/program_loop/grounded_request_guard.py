"""Source-local repeat protection for the three exact-copy grounding Skills.

Their provider request is assembled solely from resolved inputs and acquired
source context. An unrelated lineage increment does not make identical input
new. A returned invalid response also cannot improve through an identical
greedy request. Changed real context and explicit correction messages produce
different identities and remain allowed. Nothing is cached as a business fact.
"""

from collections import Counter
from copy import deepcopy
import hashlib
import json

GROUNDING_SKILLS = frozenset(
    {"INTERPRET_FIELD_NAME", "INTERPRET_FIELD_DESCRIPTION", "EXPAND_FIELD_ABBREVIATION"}
)


class GroundedRequestGuard:
    def __init__(self, generate, max_identical_calls):
        self.generate = generate
        self.max_identical_calls = max_identical_calls
        self.returned = Counter()
        self.last_generation_metadata = {}

    def begin_source(self):
        self.returned.clear()
        self.last_generation_metadata = {}

    def __getattr__(self, name):
        return getattr(self.generate, name)

    def __call__(self, messages):
        try:
            payload = json.loads(messages[1]["content"])
        except (IndexError, KeyError, TypeError, ValueError):
            payload = {}
        protected = (
            isinstance(payload, dict)
            and payload.get("skill") in GROUNDING_SKILLS
            and payload.get("grounding_protocol") == "extractive-grounded-source-v1"
        )
        if not protected:
            return self._invoke(messages)
        encoded = json.dumps(
            messages,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        identity = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        if self.returned[identity] >= self.max_identical_calls:
            self.last_generation_metadata = {
                "execution_kind": "blocked_before_model",
                "model_call_count": 0,
                "actual_API_started": False,
                "actual_API_returned": False,
                "ended_with_eos": None,
                "token_usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            }
            raise ValueError(
                "identical grounded model request already returned; blocked before another model API; request_sha256="
                + identity
            )
        result = self._invoke(messages)
        # A transport/backend exception has no returned model response and is
        # handled by the original failure policy and hard budgets.
        self.returned[identity] += 1
        return result

    def _invoke(self, messages):
        self.last_generation_metadata = {}
        try:
            return self.generate(messages)
        finally:
            metadata = getattr(self.generate, "last_generation_metadata", None)
            self.last_generation_metadata = (
                deepcopy(metadata) if isinstance(metadata, dict) else {}
            )

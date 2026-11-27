"""Activate the shared table tool contract for new collections and their replay."""
from contextlib import contextmanager
import hashlib
from pathlib import Path
from unittest.mock import patch

import dsl_table_tools
import scope_runtime
import source_evidence_protocol


def identity():
    return {**dsl_table_tools.identity(), "protocol_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


@contextmanager
def use(runtime, protocol):
    if protocol is None:
        yield
        return
    if protocol != identity():
        raise ValueError("Table evidence protocol differs from the recorded version")
    previous_bind = scope_runtime.bind_final_provider
    previous_project = source_evidence_protocol.source_evidence.project

    def bind(rollout, install):
        return previous_bind(rollout, lambda controller: dsl_table_tools.install(install(controller)))

    def project(messages, native):
        if not messages[0]["content"].startswith("You are the Global Program Planner"):
            return previous_project(messages, native)
        return messages

    class Actor(runtime.CommonActor):
        def project(self, messages):
            view = super().project(messages)
            if messages[0]["content"].startswith("You are the Global Program Planner"):
                from copy import deepcopy
                shared = deepcopy(view["messages"])
                shared[0]["content"] += ("\nChoose acquisitions only when relevant. GET_TABLE_FIELDS returns neighbor "
                    "names only and requires an observed table name. To inspect selected columns, select and "
                    "expand them. GET_COMPLETE_TABLE_CONTEXT can retrieve the available details when broader "
                    "context is needed. Target metadata alone may suffice. Never infer missing facts.")
                return {**view, "messages": shared, "input_sha256": runtime.previous.message_sha(shared)}
            return view

    with (patch.object(scope_runtime, "bind_final_provider", bind),
          patch.object(source_evidence_protocol, "project", lambda messages, native: messages),
          patch.object(source_evidence_protocol.source_evidence, "project", project),
          patch.object(runtime, "CommonActor", Actor)):
        yield

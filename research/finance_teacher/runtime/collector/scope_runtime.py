"""Bind the final R14 metadata provider when the native controller is constructed."""
from contextlib import contextmanager
import hashlib
from pathlib import Path
from types import FunctionType, SimpleNamespace

from r14_common import load

SOURCE = Path(__file__).resolve().parent.parent / "finance_program_evidence_consumption_20261024_r10/scope_binding.py"
PROVIDER = SOURCE.parent.parent / "finance_program_recovery_source_scope_20261024_r8/source_scope.py"


def identity():
    return {"version": "final-source-scope-v1", "binder_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def bind_final_provider(rollout, install):
    # R14 replaces the lookup during controller construction. The preceding
    # binder still holds the earlier lookup and correctly rejects that replacement.
    previous = rollout.__globals__["old"]
    control = SimpleNamespace(**vars(previous.c))
    factory = control.controller

    def create(*args, **kwargs):
        return install(factory(*args, **kwargs))

    control.controller = create
    updated = SimpleNamespace(**{**vars(previous), "c": control})
    return FunctionType(rollout.__code__, {**rollout.__globals__, "old": updated}, rollout.__name__,
                        rollout.__defaults__, rollout.__closure__)


@contextmanager
def use_scope(native, record):
    if record is None:
        yield
        return
    if record != identity():
        raise ValueError("Unexpected final-source binder identity")
    original = native.build
    binder = load("teacher_final_scope_binding", SOURCE)
    # R14 imports the R8 copy; the binder was published against its identical R9 copy.
    if PROVIDER.read_bytes() != binder.PROVIDER.read_bytes():
        raise ValueError("R14 provider differs from the validated source-scope provider")
    binder.PROVIDER = PROVIDER

    def build(*args, **kwargs):
        return bind_final_provider(original(*args, **kwargs), binder.install)

    native.build = build
    try:
        yield
    finally:
        native.build = original

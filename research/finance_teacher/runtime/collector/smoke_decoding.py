"""Bind an optional smoke-test thinking budget to collection and independent replay."""
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch


def configure(module, budget):
    if budget is None:
        return module
    if type(budget) is not int or not 1 <= budget <= 2048:
        raise ValueError("A reduced thinking budget must be between 1 and the frozen budget of 2048")
    original = module.build_body

    def build(messages, config):
        if config.get("thinking_budget") != budget:
            raise ValueError("Admission and saved decoding config differ")
        body = original(messages, config)
        return {**body, "thinking_budget": budget}

    module.build_body = build
    return module


@contextmanager
def verifier(science, budget):
    if budget is None:
        yield
        return
    verifier_module = science.verify200
    loader = science.runtime().previous.original
    original_load = loader.load
    original_replay = verifier_module.load_replay

    def load(name, path):
        module = original_load(name, path)
        return configure(module, budget) if Path(path) == verifier_module.R9 / "campaign.py" else module

    with (patch.object(loader, "load", load),
          patch.object(verifier_module, "load_replay", lambda: configure(original_replay(), budget))):
        yield

"""Targeted check's original native parser initialization."""
from pathlib import Path
import sys

E2 = Path(__file__).resolve().parent.parent / 'finance_program_student_b80_eval_20261021_r2'

def native_skill():
    sys.path.insert(0, str(E2))
    from paired_eval import load_engine, load_inputs
    engine = load_engine(E2)
    inputs, _ = load_inputs(E2)
    controller = engine.old.c.controller(lambda messages: '{}', inputs['study'])
    from agent.orchestration.model_skill_executor import parse_skill_observation
    return parse_skill_observation, controller.bridge.catalog['SYNTHESIZE_METADATA_SEMANTICS']

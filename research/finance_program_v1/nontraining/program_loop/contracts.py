"""Run limits and JSON trace; the accepted Program is immutable."""
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path

from . import _vendor
from agent.orchestration.contracts import Program, ToolCall


def json_snapshot(value):
    """Keep valid JSON data exact; represent invalid model I/O explicitly."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    if isinstance(value, dict):
        return {str(key): json_snapshot(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_snapshot(item) for item in value]
    return {"invalid_json_type": type(value).__name__, "repr": repr(value)}


@dataclass(frozen=True)
class ProgramConfig:
    max_programs: int = 4
    max_program_steps: int = 10
    max_skill_steps: int = 20
    max_planner_attempts: int = 8
    max_identical_calls: int = 1

    def __post_init__(self):
        for key, value in asdict(self).items():
            if type(value) is not int or value < 1:
                raise ValueError(f"{key} must be a positive integer")


@dataclass
class ProgramTrace:
    input: dict
    execution_mode: str = "caller supplied model I/O + real executor"
    planner_attempts: list = field(default_factory=list)
    programs: list = field(default_factory=list)
    state: dict = field(default_factory=dict)
    budget: dict = field(default_factory=lambda: {"programs": 0, "planner_attempts": 0, "skill_steps": 0})
    terminated: bool = False
    final_result: dict | None = None
    stop_reason: str = "running"

    def to_dict(self):
        return json_snapshot(asdict(self))

    def write_json(self, path):
        serialized = json.dumps(self.to_dict(), ensure_ascii=False, indent=2, allow_nan=False)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            stream.write(serialized + "\n")

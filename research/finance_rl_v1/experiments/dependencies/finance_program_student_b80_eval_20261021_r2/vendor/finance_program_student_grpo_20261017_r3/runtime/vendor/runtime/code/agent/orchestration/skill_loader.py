"""Strict loader for one Markdown contract per orchestration Skill."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .catalog import SkillSpec


class SkillContractError(ValueError):
    """Raised when Skill files are missing, duplicated, or inconsistent."""


@dataclass(frozen=True)
class SkillContract:
    name: str
    dsl: str
    required_inputs: tuple[str, ...]
    optional_inputs: tuple[str, ...]
    produces: tuple[str, ...]
    terminal: bool
    body: str
    path: Path


def _parse_list(value: str, *, key: str, path: Path) -> tuple[str, ...]:
    value = value.strip()
    if not value.startswith("[") or not value.endswith("]"):
        raise SkillContractError(f"{path}: {key} must use [a, b] syntax")
    inner = value[1:-1].strip()
    if not inner:
        return ()
    items = tuple(item.strip() for item in inner.split(","))
    if any(not item for item in items):
        raise SkillContractError(f"{path}: {key} contains an empty value")
    return items


def _parse_contract(path: Path) -> SkillContract:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if len(lines) < 3 or lines[0].strip() != "---":
        raise SkillContractError(f"{path}: missing opening front matter marker")
    try:
        end = next(index for index, line in enumerate(lines[1:], start=1) if line.strip() == "---")
    except StopIteration as exc:
        raise SkillContractError(f"{path}: missing closing front matter marker") from exc

    header: dict[str, str] = {}
    for line in lines[1:end]:
        if ":" not in line:
            raise SkillContractError(f"{path}: invalid front matter line {line!r}")
        key, value = line.split(":", 1)
        key = key.strip()
        if key in header:
            raise SkillContractError(f"{path}: duplicate front matter key {key!r}")
        header[key] = value.strip()

    required = {"name", "dsl", "required_inputs", "optional_inputs", "produces", "terminal"}
    if set(header) != required:
        missing = sorted(required - set(header))
        extra = sorted(set(header) - required)
        raise SkillContractError(f"{path}: front matter keys mismatch; missing={missing}, extra={extra}")
    terminal_text = header["terminal"].lower()
    if terminal_text not in {"true", "false"}:
        raise SkillContractError(f"{path}: terminal must be true or false")
    body = "\n".join(lines[end + 1 :]).strip()
    if not body:
        raise SkillContractError(f"{path}: Skill contract body is empty")
    return SkillContract(
        name=header["name"],
        dsl=header["dsl"],
        required_inputs=_parse_list(header["required_inputs"], key="required_inputs", path=path),
        optional_inputs=_parse_list(header["optional_inputs"], key="optional_inputs", path=path),
        produces=_parse_list(header["produces"], key="produces", path=path),
        terminal=terminal_text == "true",
        body=body,
        path=path,
    )


class SkillLoader:
    """Discover and validate Markdown Skill contracts against a catalog."""

    def __init__(
        self,
        root: str | Path,
        catalog: Mapping[str, SkillSpec],
        *,
        extra_roots: tuple[str | Path, ...] = (),
        strict_catalog: bool = True,
    ) -> None:
        self.root = Path(root)
        self.extra_roots = tuple(Path(path) for path in extra_roots)
        self.strict_catalog = strict_catalog
        self.catalog = dict(catalog)
        self._contracts: dict[str, SkillContract] | None = None

    def validate_all(self) -> dict[str, SkillContract]:
        discovered: dict[str, list[SkillContract]] = {}
        for root in (self.root, *self.extra_roots):
            if root.is_dir():
                for path in sorted(root.glob("*.md")):
                    contract = _parse_contract(path)
                    discovered.setdefault(contract.name, []).append(contract)

        validated: dict[str, SkillContract] = {}
        for name, spec in self.catalog.items():
            matches = discovered.get(name, [])
            if not matches:
                raise SkillContractError(f"missing Skill contract for {name}")
            if len(matches) > 1:
                paths = ", ".join(str(item.path) for item in matches)
                raise SkillContractError(f"duplicate Skill contracts for {name}: {paths}")
            contract = matches[0]
            expected_filename = f"{spec.role}_{spec.name}.md"
            if contract.path.name != expected_filename:
                raise SkillContractError(
                    f"{contract.path}: filename must be {expected_filename!r}"
                )
            expected = {
                "dsl": spec.role,
                "required_inputs": spec.required_inputs,
                "optional_inputs": spec.optional_inputs,
                "produces": spec.produces,
                "terminal": spec.terminal,
            }
            actual = {
                "dsl": contract.dsl,
                "required_inputs": contract.required_inputs,
                "optional_inputs": contract.optional_inputs,
                "produces": contract.produces,
                "terminal": contract.terminal,
            }
            for key, value in expected.items():
                if actual[key] != value:
                    raise SkillContractError(
                        f"{contract.path}: {key}={actual[key]!r} does not match catalog {value!r}"
                    )
            validated[name] = contract

        unexpected = sorted(set(discovered) - set(self.catalog))
        if unexpected and self.strict_catalog:
            raise SkillContractError(f"unexpected Skill contracts: {unexpected}")
        self._contracts = validated
        return dict(validated)

    def load(self, name: str) -> SkillContract:
        if name not in self.catalog:
            raise SkillContractError(f"unknown Skill {name!r}")
        if self._contracts is None:
            self.validate_all()
        assert self._contracts is not None
        return self._contracts[name]


__all__ = ["SkillContract", "SkillContractError", "SkillLoader"]

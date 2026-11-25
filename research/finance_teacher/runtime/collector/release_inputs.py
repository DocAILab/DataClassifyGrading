"""Load the immutable metadata projection for the 25 published TRAIN fields."""
from contextlib import contextmanager
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "inputs/cohort.json.gz"
MANIFEST = ROOT / "runtime-sha256.json"


def manifest_sha():
    return hashlib.sha256(MANIFEST.read_bytes()).hexdigest()


def verify_files():
    entries = json.loads(MANIFEST.read_bytes())
    for name, expected in entries.items():
        path = (ROOT / name).resolve(strict=True)
        if not path.is_relative_to(ROOT) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Published runtime or input differs: {name}")
    return manifest_sha()


def bundle():
    data = json.loads(gzip.decompress(INPUT.read_bytes()))
    data["lookup_rows"] = [{k: r[k] for k in ("id", "domain", "metadata")} for r in data["lookup_scopes"]]
    return data


def read_config(path):
    cfg = json.loads(Path(path).read_bytes())
    endpoint = os.environ.get("DASHSCOPE_BASE_URL", "").rstrip("/")
    if endpoint and not re.fullmatch(r"https://[A-Za-z0-9-]+\.cn-beijing\.maas\.aliyuncs\.com/compatible-mode/v1", endpoint):
        raise ValueError("DASHSCOPE_BASE_URL must be the Beijing Bailian workspace compatible-mode/v1 endpoint")
    cfg["endpoint"] = endpoint or None
    return cfg


@contextmanager
def use_catalog(admission, source_protocol, catalog):
    if admission.get("train_table_catalog") != catalog.identity() or admission.get("source_evidence_protocol") != source_protocol.identity():
        raise ValueError("This cohort uses the recorded TRAIN table catalog and source evidence protocol")
    # Expansion and split exclusion were performed before export. The frozen
    # projection already contains their result; repeat neither operation here.
    with patch.object(source_protocol.source_evidence, "extend", lambda contract, present: contract):
        yield

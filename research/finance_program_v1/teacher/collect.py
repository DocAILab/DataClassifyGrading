"""Explicit local bridge to the new grounded runtime, old evidence stays immutable."""
# 连接独立冻结的采集与运行时目录，先校验来源身份，再向模型暴露允许的元数据字段。
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
PROGRAM = ROOT.parent / 'finance_program_grounded_runtime_20261015_r1'
BASE = ROOT.parent / 'finance_program_teacher_f66_20261015_r6'
CAMPAIGN = ROOT.parent / 'finance_program_teacher_campaign_20261015_r1'
METADATA = frozenset(('database_name','database_description','table_name','table_description','field_name','field_description','field_type','value'))


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def label_free(row):
    return {'id': row['id'], 'domain': row['domain'], 'metadata': {k:v for k,v in row['metadata'].items() if k in METADATA}}


def verify():
    path = BASE / 'reviews/frozen-manifest.json'
    assert sha(path) == 'c6570be3abe9e2624bbbf943b27d94c46fb507f916388031e0f4a70d311b4dcb'
    for item in read(path)['files']:assert sha(BASE / item['path']) == item['sha256']
    assert sha(BASE / 'reviews/fixed100-evidence-01.tar.gz') == '6d1183db77f5658ba68d76aba60a8a84279dce2d93ab3ea0152c90b54c66af44'
    return read(path)


def verify_program():
    spec = importlib.util.spec_from_file_location('grounded_runtime_identity', PROGRAM / 'runtime_identity.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    identity = module.verify()
    assert identity['core_signature_sha256'] == 'c34105f2767513d7d50a1c5fa15f8123e790865d25da681efa049182110132f8'
    sys.path.insert(0, str(PROGRAM))
    return identity

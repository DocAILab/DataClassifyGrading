import importlib
import json
from pathlib import Path

import pytest


def test_downloader_targets_official_9b_repo_and_default_path():
    download = importlib.import_module('download_model')

    assert download.REPO == 'Qwen/Qwen3.5-9B'
    assert download.MODEL_DIR == Path(__file__).resolve().parents[1] / 'models' / 'Qwen3.5-9B'


def test_old_08b_identity_is_rejected_before_snapshot_or_manifest(tmp_path):
    download = importlib.import_module('download_model')
    model = tmp_path / 'model'
    model.mkdir()
    identity = {
        'repo_id': 'Qwen/Qwen3.5-0.8B',
        'revision': 'a' * 40,
        'files': [
            {'path': 'config.json', 'bytes': 2, 'expected_sha256': None},
            {'path': 'tokenizer.json', 'bytes': 2, 'expected_sha256': None},
            {'path': 'model.safetensors', 'bytes': 2, 'expected_sha256': None},
        ],
    }
    (model / 'download-identity.json').write_text(json.dumps(identity), encoding='utf-8')

    def forbidden_snapshot(**kwargs):
        pytest.fail('snapshot invoked for an identity from the old model repo')

    with pytest.raises(ValueError, match='invalid official model identity'):
        download.download_model(model, api=object(), snapshot=forbidden_snapshot)

    assert not (model / 'model-manifest.json').exists()

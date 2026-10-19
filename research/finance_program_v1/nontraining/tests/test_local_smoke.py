"""CPU boundary tests: these provide no evidence of actual CUDA generation."""
import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def module(name):
    assert (Path(__file__).resolve().parents[1] / (name + '.py')).is_file(), 'missing implementation: ' + name
    return importlib.import_module(name)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def test_exclusive_output_reservation(tmp_path):
    smoke = module('local_smoke')
    smoke.reserve_run(tmp_path / 'run')
    with pytest.raises(FileExistsError):
        smoke.reserve_run(tmp_path / 'run')


def test_frozen_unlabelled_three_rows_and_six_definitions():
    registry, rows = module('local_smoke').smoke_inputs()
    assert len(rows) == 3 and len(registry) == 6
    assert [r['id'] for r in rows] == ['local-balance', 'local-phone', 'local-invoice']
    assert [r['metadata']['field_name'] for r in rows] == ['account_balance', 'customer_phone', 'invoice_amount']
    assert [r['metadata']['field_description'] for r in rows] == ['账户当前余额', '客户联系手机号', '发票金额']
    assert all(set(r) == {'id', 'domain', 'metadata'} and r['domain'] == '金融' for r in rows)
    assert all(set(r['metadata']) == {'field_name', 'field_description'} for r in rows)
    assert set(registry) == {'account_balance', 'contact_phone', 'invoice_amount', 'customer_name', 'order_id', 'transaction_date'}
    assert all(e['grading_level'] == 'L2' and e['description'] for e in registry.values())


def test_offline_loading_and_fixed_inference_settings():
    smoke = module('local_smoke')
    assert smoke.OFFLINE == {'local_files_only': True, 'trust_remote_code': False}
    opts = smoke.generation_options(SimpleNamespace(eos_token_id=[8, 9], pad_token_id=0), SimpleNamespace(eos_token_id=7, pad_token_id=0))
    assert opts == {'do_sample': False, 'max_new_tokens': 512, 'max_time': 90.0, 'eos_token_id': [8, 9], 'pad_token_id': 0}
    with pytest.raises(ValueError, match='EOS'):
        smoke.generation_options(SimpleNamespace(eos_token_id=None, pad_token_id=None), SimpleNamespace(eos_token_id=None, pad_token_id=None))


def test_context_rejected_without_truncation_and_thinking_disabled():
    smoke = module('local_smoke')
    class Tokenizer:
        def apply_chat_template(self, messages, **kw):
            assert kw == {'tokenize': False, 'add_generation_prompt': True, 'enable_thinking': False}
            return 'whole prompt'
        def __call__(self, prompt, **kw):
            assert prompt == 'whole prompt' and kw == {'return_tensors': 'pt', 'truncation': False}
            return {'input_ids': SimpleNamespace(shape=(1, 32257))}
    with pytest.raises(ValueError, match='context'):
        smoke.encode_messages(Tokenizer(), [{'role': 'user', 'content': 'raw'}])


class BoundaryBackend:
    """Generation-only replacement; guard/executor/controller stay real."""
    def __init__(self, model_path, call_dir):
        self.failures = []
        self.calls = []
    def __call__(self, messages):
        self.calls.append(messages)
        return '```json\n{"program": []}\n```'


@pytest.fixture
def runner_budget(tmp_path):
    output = tmp_path / 'budget-run'
    assert module('local_smoke').main(
        ['--model', str(tmp_path), '--output', str(output)], backend_factory=BoundaryBackend) == 0
    return read(output / 'inputs.json')['budget']


def test_real_runner_uses_fixed_budget10(runner_budget):
    assert runner_budget == dict(max_programs=3, max_program_steps=10, max_skill_steps=12,
                                max_planner_attempts=4, max_identical_calls=1)


@pytest.mark.parametrize('length', [9, 10, 11])
def test_real_parser_boundary_from_runner_budget(runner_budget, length):
    from program_loop import build_program_controller, ProgramConfig
    from program_loop.validation import parse_program
    registry, rows = module('local_smoke').smoke_inputs()
    controller = build_program_controller(lambda messages: '', registry, rows, ProgramConfig(**runner_budget))
    raw = json.dumps({'program': [{'tool': 'INTERPRET_DOMAIN', 'arguments': {}}] * length})
    if length <= 10:
        assert len(parse_program(raw, controller.bridge.catalog, {'domain': '金融'},
                                 runner_budget['max_program_steps']).steps) == length
    else:
        with pytest.raises(ValueError, match='Program length exceeds max_program_steps'):
            parse_program(raw, controller.bridge.catalog, {'domain': '金融'}, runner_budget['max_program_steps'])


def test_raw_rejections_independent_budgets_and_normal_nontermination(tmp_path):
    smoke = module('local_smoke')
    output = tmp_path / 'run'
    assert smoke.main(['--model', str(tmp_path), '--output', str(output)], backend_factory=BoundaryBackend) == 0
    summary = read(output / 'summary.json')
    assert summary['status'] == 'completed' and summary['terminated_count'] == 0
    assert summary['accepted_plans'] == 0 and summary['planner_attempts'] == 12
    for name in ['local-balance', 'local-phone', 'local-invoice']:
        trace = read(output / 'samples' / (name + '.json'))
        assert trace['budget'] == {'programs': 0, 'planner_attempts': 4, 'skill_steps': 0}
        assert all(a['raw_response'] == '```json\n{"program": []}\n```' for a in trace['planner_attempts'])
    assert (output / 'environment.json').is_file()


def test_existing_root_refused_before_backend_init(tmp_path):
    smoke = module('local_smoke')
    output = tmp_path / 'run'
    output.mkdir()
    def forbidden(*a):
        pytest.fail('backend initialized for existing output')
    assert smoke.main(['--model', str(tmp_path), '--output', str(output)], backend_factory=forbidden) == 2
    assert list(output.iterdir()) == []


def test_initialization_failure_is_durable(tmp_path):
    smoke = module('local_smoke')
    def broken(*a):
        raise RuntimeError('checkpoint unavailable')
    output = tmp_path / 'run'
    assert smoke.main(['--model', str(tmp_path), '--output', str(output)], backend_factory=broken) == 2
    summary = read(output / 'summary.json')
    assert summary['status'] == 'initialization_failure' and 'checkpoint unavailable' in summary['error']
    assert (output / 'environment.json').is_file()


def test_backend_faults_not_hidden_by_core_retry(tmp_path):
    smoke = module('local_smoke')
    class Broken(BoundaryBackend):
        def __call__(self, messages):
            raise RuntimeError('generation failed')
    output = tmp_path / 'run'
    assert smoke.main(['--model', str(tmp_path), '--output', str(output)], backend_factory=Broken) == 2
    summary = read(output / 'summary.json')
    assert summary['status'] == 'backend_failure' and len(summary['backend_errors']) == 12
    assert len(list((output / 'samples').glob('*.json'))) == 3


def test_accepted_incomplete_program_is_ordinary_result(tmp_path):
    smoke = module('local_smoke')
    class Incomplete(BoundaryBackend):
        def __call__(self, messages):
            if 'Global Program Planner' in messages[0]['content']:
                return json.dumps({'program': [{'tool': 'INTERPRET_DOMAIN', 'arguments': {}}]})
            return json.dumps({'summary': '合成解释', 'data': {'domain_semantics': '金融业务'}})
    output = tmp_path / 'run'
    assert smoke.main(['--model', str(tmp_path), '--output', str(output)], backend_factory=Incomplete) == 0
    summary = read(output / 'summary.json')
    assert summary['status'] == 'completed' and summary['accepted_plans'] == 9 and summary['terminated_count'] == 0
    trace = read(output / 'samples' / 'local-balance.json')
    assert trace['programs'][0]['steps'][0]['status'] == 'success'
    assert trace['stop_reason'] == 'program_budget_exhausted'
    assert trace['state']['domain_semantics'] == '金融业务'


def test_downloader_pins_identity_before_snapshot_and_hashes(tmp_path):
    download = module('download_model')
    model = tmp_path / 'model'
    files = {'config.json': b'{"model_type":"qwen3_5"}', 'tokenizer.json': b'{}', 'model.safetensors': b'tensor'}
    class Api:
        def model_info(self, repo_id, **kw):
            assert repo_id == 'Qwen/Qwen3.5-9B' and kw == {'files_metadata': True, 'token': False}
            return SimpleNamespace(sha='a' * 40, siblings=[SimpleNamespace(rfilename=n, size=len(b)) for n, b in files.items()] + [SimpleNamespace(rfilename='modeling_remote.py', size=3)])
    def snapshot(**kw):
        identity = read(model / 'download-identity.json')
        assert kw['revision'] == identity['revision'] == 'a' * 40
        assert kw['token'] is False and kw['repo_id'] == 'Qwen/Qwen3.5-9B'
        assert set(kw['allow_patterns']) == set(files)
        for n, b in files.items():
            (model / n).write_bytes(b)
    manifest = download.download_model(model, api=Api(), snapshot=snapshot)
    assert manifest['revision'] == 'a' * 40 and len(manifest['files']) == 3
    assert all(e['sha256'] == hashlib.sha256(files[e['path']]).hexdigest() for e in manifest['files'])
    with pytest.raises(FileExistsError):
        download.download_model(model, api=Api(), snapshot=snapshot)


def test_interrupted_download_resumes_pinned_revision(tmp_path):
    download = module('download_model')
    model = tmp_path / 'model'
    class Api:
        def model_info(self, *a, **k):
            return SimpleNamespace(sha='b' * 40, siblings=[SimpleNamespace(rfilename=n, size=2) for n in ('config.json', 'tokenizer.json', 'model.safetensors')])
    def interrupted(**kw):
        raise RuntimeError('interrupted')
    with pytest.raises(RuntimeError, match='interrupted'):
        download.download_model(model, api=Api(), snapshot=interrupted)
    assert (model / 'download-identity.json').is_file()
    class NoResolve:
        def model_info(self, *a, **k):
            pytest.fail('resume must not resolve a fresh revision')
    def resumed(**kw):
        assert kw['revision'] == 'b' * 40
        for name in kw['allow_patterns']:
            (model / name).write_bytes(b'{}')
    assert download.download_model(model, api=NoResolve(), snapshot=resumed)['revision'] == 'b' * 40


def test_download_identity_tamper_and_size_mismatch_rejected(tmp_path):
    download = module('download_model')
    model = tmp_path / 'model'
    model.mkdir()
    (model / 'download-identity.json').write_text(json.dumps({'repo_id': 'evil/repo', 'revision': 'a' * 40, 'files': []}), encoding='utf-8')
    with pytest.raises(ValueError, match='identity'):
        download.download_model(model, api=object(), snapshot=lambda **k: None)


@pytest.mark.parametrize('mode', ['raw', 'context', 'generation_error'])
def test_cuda_backend_records_raw_devices_budget_and_offline_options(tmp_path, monkeypatch, mode):
    smoke = module('local_smoke')
    import sys
    import torch
    from types import ModuleType
    tf = ModuleType('transformers')
    config = SimpleNamespace(model_type='qwen3_5', eos_token_id=9, pad_token_id=0)
    class Tensor:
        shape = (1, 32257 if mode == 'context' else 12)
        device = torch.device('cuda:0')
        def to(self, device):
            assert device == 'cuda:0'
            return self
    class Output:
        def __getitem__(self, key):
            assert key == (0, slice(12, None))
            return [2] * 512
    class Tokenizer:
        eos_token_id = 9
        pad_token_id = 0
        def apply_chat_template(self, messages, **kw):
            assert kw['enable_thinking'] is False
            return 'prompt'
        def __call__(self, prompt, **kw):
            assert kw['truncation'] is False
            return {'input_ids': Tensor(), 'attention_mask': Tensor()}
        def decode(self, ids, **kw):
            assert kw == {'skip_special_tokens': True, 'clean_up_tokenization_spaces': False}
            return '  ```json\n{bad json}\n```  '
    class Model:
        generation_config = config
        def parameters(self):
            return iter([SimpleNamespace(device=Tensor.device, dtype=torch.float16, is_floating_point=lambda: True)])
        def eval(self):
            return self
        def generate(self, **kw):
            if mode == 'generation_error':
                raise RuntimeError('test generation failure')
            assert kw['do_sample'] is False and kw['max_new_tokens'] == 512 and kw['max_time'] == 90.0
            assert kw['eos_token_id'] == 9
            return Output()
    def config_load(path, **kw):
        assert kw == smoke.OFFLINE
        return config
    def tokenizer_load(path, **kw):
        assert kw == smoke.OFFLINE
        return Tokenizer()
    def model_load(path, **kw):
        assert kw['local_files_only'] is True and kw['trust_remote_code'] is False
        assert kw['device_map'] == {'': 'cuda:0'} and kw['dtype'] == torch.float16
        assert kw['attn_implementation'] == 'sdpa'
        return Model()
    tf.AutoConfig = SimpleNamespace(from_pretrained=config_load)
    tf.AutoTokenizer = SimpleNamespace(from_pretrained=tokenizer_load)
    tf.AutoModelForImageTextToText = SimpleNamespace(from_pretrained=model_load)
    monkeypatch.setitem(sys.modules, 'transformers', tf)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'set_device', lambda x: None)
    monkeypatch.setattr(torch.cuda, 'manual_seed_all', lambda x: None)
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda x: (10**10, 12 * 10**9))
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda x: 'CPU-test-only CUDA device replacement')
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda x: None)
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', lambda x: None)
    for name in ('memory_allocated', 'memory_reserved', 'max_memory_allocated'):
        monkeypatch.setattr(torch.cuda, name, lambda x: 100)
    monkeypatch.setattr(smoke, 'verify_model', lambda path: {'revision': 'a' * 40})
    backend = smoke.Backend(tmp_path, tmp_path / 'calls')
    if mode != 'raw':
        with pytest.raises((ValueError, RuntimeError)):
            backend.generate([{'role': 'user', 'content': 'unaltered'}])
        result = read(tmp_path / 'calls' / 'call-0001' / 'result.json')
        assert result['input_tokens'] == (32257 if mode == 'context' else 12)
        assert result['error'] and result['raw_text'] is None and len(backend.failures) == 1
        assert read(tmp_path / 'calls' / 'call-0001' / 'request.json')['messages'][0]['content'] == 'unaltered'
        return
    raw = backend.generate([{'role': 'user', 'content': 'unaltered'}])
    assert raw == '  ```json\n{bad json}\n```  '
    result = read(tmp_path / 'calls' / 'call-0001' / 'result.json')
    assert result['raw_text'] == raw and result['output_tokens'] == 512 and result['input_tokens'] == 12
    assert result['truncation_at_budget'] is True and result['input_devices'] == ['cuda:0']
    assert result['parameter_devices'] == ['cuda:0'] and result['error'] is None


@pytest.mark.parametrize('failed_metric', ['memory_allocated', 'memory_reserved', 'max_memory_allocated'])
def test_cuda_telemetry_failure_retains_context_error_and_result(tmp_path, failed_metric):
    smoke = module('local_smoke')
    class Tokenizer:
        def apply_chat_template(self, messages, **kw):
            return 'whole prompt'
        def __call__(self, prompt, **kw):
            return {'input_ids': SimpleNamespace(shape=(1, 32257))}
    def unavailable(device):
        raise RuntimeError('telemetry unavailable')
    metrics = ('memory_allocated', 'memory_reserved', 'max_memory_allocated')
    cuda = SimpleNamespace(**{name: unavailable if name == failed_metric else lambda device: 100 for name in metrics})
    # Replace only external tokenizer/CUDA I/O; execute real encode/generate/finalization.
    backend = smoke.Backend.__new__(smoke.Backend)
    backend.call_dir = tmp_path / 'calls'
    backend.call_dir.mkdir()
    backend.calls, backend.failures = [], []
    backend.parameter_devices, backend.parameter_dtypes = ['cuda:0'], ['torch.float16']
    backend.attention, backend.options = 'sdpa', {'max_new_tokens': 512}
    backend.tokenizer, backend.torch = Tokenizer(), SimpleNamespace(cuda=cuda)
    messages = [{'role': 'user', 'content': 'unaltered'}]
    with pytest.raises(smoke.ContextBudgetExceeded, match='32257'):
        backend.generate(messages)
    call = backend.call_dir / 'call-0001'
    assert read(call / 'request.json')['messages'] == messages
    result = read(call / 'result.json')
    assert result['input_tokens'] == 32257 and result['raw_text'] is None
    assert result['error'].startswith('ContextBudgetExceeded: context budget exceeded: 32257')
    assert result['telemetry_errors'] == {failed_metric: 'RuntimeError: telemetry unavailable'}
    assert result['memory'] == {name: None if name == failed_metric else 100 for name in metrics}
    assert backend.failures == [{'call': str(call), 'error': result['error']}]


def test_model_verification_detects_modified_file(tmp_path):
    download = module('download_model')
    model = tmp_path / 'model'
    class Api:
        def model_info(self, *a, **k):
            return SimpleNamespace(sha='c' * 40, siblings=[SimpleNamespace(rfilename=n, size=2) for n in ('config.json', 'tokenizer.json', 'model.safetensors')])
    def snapshot(**kw):
        for n in kw['allow_patterns']:
            (model / n).write_bytes(b'{}')
    download.download_model(model, api=Api(), snapshot=snapshot)
    assert download.verify_model(model)['revision'] == 'c' * 40
    (model / 'model.safetensors').write_bytes(b'XX')
    with pytest.raises(ValueError, match='hash'):
        download.verify_model(model)


def test_official_unusual_shard_name_and_trusted_lfs_hash(tmp_path):
    download = module('download_model')
    model = tmp_path / 'model'
    weight = 'model.safetensors-00001-of-00001.safetensors'
    payload = b'{}'
    digest = hashlib.sha256(payload).hexdigest()
    class Api:
        def model_info(self, *a, **k):
            return SimpleNamespace(sha='d' * 40, siblings=[SimpleNamespace(rfilename=n, size=2, lfs=SimpleNamespace(sha256=digest)) for n in ('config.json', 'tokenizer.json', weight)])
    def snapshot(**kw):
        assert weight in kw['allow_patterns']
        for n in kw['allow_patterns']:
            (model / n).write_bytes(payload)
    result = download.download_model(model, api=Api(), snapshot=snapshot)
    assert all(e['expected_sha256'] == digest for e in result['files'])
    assert all(e['sha256'] == e['expected_sha256'] for e in result['files'])


def test_trusted_lfs_hash_mismatch_retains_identity_without_manifest(tmp_path):
    download = module('download_model')
    model = tmp_path / 'model'
    class Api:
        def model_info(self, *a, **k):
            return SimpleNamespace(sha='d' * 40, siblings=[SimpleNamespace(rfilename=n, size=2, lfs=SimpleNamespace(sha256='0' * 64)) for n in ('config.json', 'tokenizer.json', 'model.safetensors')])
    def snapshot(**kw):
        for n in kw['allow_patterns']:
            (model / n).write_bytes(b'{}')
    with pytest.raises(ValueError, match='hash'):
        download.download_model(model, api=Api(), snapshot=snapshot)
    assert (model / 'download-identity.json').is_file() and not (model / 'model-manifest.json').exists()


def test_unlisted_executable_prevents_completed_manifest(tmp_path):
    download = module('download_model')
    model = tmp_path / 'model'
    class Api:
        def model_info(self, *a, **k):
            return SimpleNamespace(sha='e' * 40, siblings=[SimpleNamespace(rfilename=n, size=2) for n in ('config.json', 'tokenizer.json', 'model.safetensors')])
    def snapshot(**kw):
        for n in kw['allow_patterns']:
            (model / n).write_bytes(b'{}')
        (model / 'remote_code.py').write_bytes(b'{}')
    with pytest.raises(ValueError, match='inventory'):
        download.download_model(model, api=Api(), snapshot=snapshot)
    assert not (model / 'model-manifest.json').exists()


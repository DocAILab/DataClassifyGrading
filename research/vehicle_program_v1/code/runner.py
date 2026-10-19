"""Full Val inference, durable per-source receipts, identity-checked crash recovery."""
import argparse
from contextlib import contextmanager
from copy import deepcopy
from importlib.metadata import PackageNotFoundError, version
import os
from pathlib import Path
import platform
import time
from artifacts import BUDGET, GENERATION, MODEL_REVISION, fingerprint, load_controller, load_study, put, read, require_private_output, sha
from backends import ScriptedBackend, local_backend


@contextmanager
def exclusive_run(output):
    """OS releases the advisory lock on crashes; never signal another process."""
    path = Path(output) / '.run.lock'
    with path.open('a+b') as f:
        f.seek(0, 2)
        if f.tell() == 0:
            f.write(b'0')
            f.flush()
        f.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            # Interrupted atomic writes are never committed evidence.
            for tmp in Path(output).rglob('*.json.tmp'):
                tmp.unlink()
            yield
        finally:
            if os.name == 'nt':
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def members(folder):
    return {p.relative_to(folder).as_posix(): sha(p) for p in sorted(folder.rglob('*'))
            if p.is_file() and p.name != 'receipt.json' and not p.name.endswith('.tmp')}


def read_receipts(output, ids):
    result = {}
    samples = Path(output) / 'sources'
    expected = {f'{n:04d}': ident for n, ident in enumerate(ids, 1)}
    for folder in sorted(samples.iterdir()) if samples.exists() else []:
        if not folder.is_dir() or folder.name not in expected:
            raise ValueError('unknown source artifact')
        receipt_path = folder / 'receipt.json'
        if not receipt_path.exists():
            continue
        receipt = read(receipt_path)
        ident = expected[folder.name]
        if receipt['id'] != ident or receipt['files'] != members(folder):
            raise ValueError('saved source receipt identity/hash differs')
        if receipt['status'] not in ('completed', 'runner_error', 'backend_error', 'interrupted'):
            raise ValueError('invalid committed status')
        result[ident] = receipt
    return result


class Recorder:
    def __init__(self, generate, folder):
        self.generate = generate
        self.folder = Path(folder)
        self.calls = []
        self.failed = False
        self.last_generation_metadata = {}

    def __call__(self, messages):
        if self.failed:
            raise RuntimeError('backend failed earlier in this source; no further generation')
        record = {'messages': deepcopy(messages), 'raw_response': None, 'error_type': None}
        started = time.monotonic()
        try:
            record['raw_response'] = self.generate(messages)
            return record['raw_response']
        except Exception as exc:
            self.failed = True
            record['error_type'] = type(exc).__name__
            raise
        finally:
            self.last_generation_metadata = deepcopy(getattr(self.generate, 'last_generation_metadata', {}))
            record.update(generation=self.last_generation_metadata, seconds=time.monotonic() - started)
            self.calls.append(record)
            put(self.folder / f'{len(self.calls):04d}.json', record)


def environment():
    packages = {}
    for name in ('torch', 'transformers', 'accelerate', 'safetensors', 'tokenizers', 'jinja2'):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {'python': platform.python_version(), 'platform': platform.platform(), 'packages': packages}


def run(package, output, *, backend='mock', model=None, api_config=None, resume=False, max_samples=None, factory=None):
    study, manifest = load_study(package)
    ids = study['expected_ids']
    output = Path(output)
    require_private_output(output)
    if max_samples is not None and max_samples < 1:
        raise ValueError('max_samples must be positive')
    if backend not in ('mock', 'local', 'api') or (backend == 'local' and not model) or (backend == 'api' and not api_config):
        raise ValueError('select mock, local with --model, or api with --api-config')
    kind = {'mock': 'scripted_cpu_test', 'local': 'local_official_base', 'api': 'hosted_qwen35_9b_api'}[backend]
    config, service = None, None
    generation = GENERATION
    if backend == 'api':
        from api_backend import api_generation, api_identity, load_config
        config = load_config(api_config)
        service, generation = api_identity(config), api_generation()
    model_identity = None
    if backend == 'local':
        m = read(Path(model) / 'model-manifest.json')
        if m.get('repo_id') != 'Qwen/Qwen3.5-9B' or m.get('revision') != MODEL_REVISION:
            raise ValueError('base model revision differs')
        model_identity = sha(Path(model) / 'model-manifest.json')
    identity = {'schema': 1, 'arm': 'NONT-D', 'backend': kind,
                'study_sha256': manifest['study_sha256'], 'code_sha256': fingerprint(manifest['code']),
                'model_manifest_sha256': model_identity, 'generation': generation, 'budget': BUDGET,
                'expected_ids': ids, 'environment': environment()}
    if service is not None:
        identity['api_service'] = service
    if output.exists() and not resume:
        raise FileExistsError('run exists; explicitly resume the same immutable identity')
    output.mkdir(parents=True, exist_ok=resume)
    with exclusive_run(output):
        if (output / 'run.json').exists():
            if read(output / 'run.json') != identity:
                raise ValueError('run identity differs')
        else:
            if set(p.name for p in output.iterdir()) != {'.run.lock'}:
                raise ValueError('unidentified run artifacts')
            put(output / 'run.json', identity)
        receipts = read_receipts(output, ids)
        builder = load_controller()
        generator = None
        newly_finished = 0
        halted = None
        try:
            for n, row in enumerate(study['rows'], 1):
                if row['id'] in receipts:
                    continue
                folder = output / 'sources' / f'{n:04d}'
                if folder.exists():
                    # Preserve an interrupted attempt, including any partial model I/O. Never reroll it.
                    receipt = {'id': row['id'], 'status': 'interrupted', 'terminated': False,
                               'stop_reason': 'interrupted_before_commit', 'seconds': None,
                               'model_calls': len(list((folder / 'calls').glob('*.json'))), 'files': members(folder)}
                    put(folder / 'receipt.json', receipt)
                else:
                    if generator is None:
                        if factory:
                            generator = factory()
                        elif backend == 'api':
                            from api_backend import APIBackend
                            if not (output / 'backend/api-budget.json').exists() and any((output / 'sources').glob('*/calls/*.json')):
                                raise ValueError('API budget missing for existing requests')
                            generator = APIBackend(config, output / 'backend')
                        else:
                            generator = ScriptedBackend() if backend == 'mock' else local_backend(model, output / 'backend')
                    if hasattr(generator, 'before_source'):
                        generator.before_source()
                    folder.mkdir(parents=True)
                    put(folder / 'started.json', {'id': row['id'], 'arm': 'NONT-D'})
                    recorded = Recorder(generator, folder / 'calls')
                    started = time.monotonic()
                    receipt = {'id': row['id'], 'status': 'runner_error', 'terminated': False,
                               'stop_reason': 'runner_error'}
                    try:
                        controller = builder(recorded, study['registry'], study['rows'], BUDGET, skill_pool='F66')
                        if len(controller.library) != 66:
                            raise ValueError('F66 library size differs')
                        trace = controller.run({'source_id': row['id'], 'domain': row['domain'],
                                                'field_name': row['metadata']['field_name']}).to_dict()
                        trace['execution_mode'] = kind
                        put(folder / 'trace.json', trace)
                        receipt.update(status='backend_error' if recorded.failed else 'completed',
                                       terminated=trace['terminated'] and not recorded.failed,
                                       stop_reason=trace['stop_reason'])
                    except Exception as exc:
                        receipt['error_type'] = type(exc).__name__
                    receipt.update(seconds=time.monotonic() - started, model_calls=len(recorded.calls), files=members(folder))
                    receipt.update(input_tokens=sum(c['generation'].get('input_tokens') or 0 for c in recorded.calls),
                                   output_tokens=sum(c['generation'].get('output_tokens') or 0 for c in recorded.calls),
                                   token_counts_measured=bool(recorded.calls) and all(
                                       c['generation'].get('input_tokens') is not None and c['generation'].get('output_tokens') is not None
                                       for c in recorded.calls))
                    put(folder / 'receipt.json', receipt)
                receipts[row['id']] = receipt
                newly_finished += 1
                write_summary(output, ids, receipts, kind)
                if receipt['status'] in ('runner_error', 'backend_error'):
                    halted = receipt['status']
                    break
                if max_samples is not None and newly_finished >= max_samples:
                    break
        except Exception as exc:
            halted = type(exc).__name__
        result = write_summary(output, ids, receipts, kind, halted=halted)
        return result


def write_summary(output, ids, receipts, kind, *, halted=None):
    summary = {'backend': kind, 'quality_claim': False if kind == 'scripted_cpu_test' else 'pending_local_scoring',
               'expected_count': len(ids), 'committed_count': len(receipts), 'pending_count': len(ids) - len(receipts),
               'terminated_count': sum(r['terminated'] for r in receipts.values()),
               'error_count': sum(r['status'] != 'completed' for r in receipts.values()),
               'model_calls': sum(r['model_calls'] for r in receipts.values()), 'halted': halted,
               'input_tokens': sum(r.get('input_tokens') or 0 for r in receipts.values()),
               'output_tokens': sum(r.get('output_tokens') or 0 for r in receipts.values()),
               'token_counts_measured': bool(receipts) and all(r.get('token_counts_measured') for r in receipts.values()),
               'status': 'complete' if len(receipts) == len(ids) else 'incomplete'}
    put(Path(output) / 'summary.json', summary, replace=True)
    budget_path = Path(output) / 'backend/api-budget.json'
    if kind == 'hosted_qwen35_9b_api' and budget_path.exists():
        attempts = read(budget_path)['attempts']
        summary.update(api_http_attempts=len(attempts),
                       api_reserved_cost_upper_cny=sum(a['reserved_cny'] for a in attempts),
                       api_uncertain_or_failed_attempts=sum(a['state'] != 'response_received' for a in attempts),
                       api_cost_is_invoice=False)
        put(Path(output) / 'summary.json', summary, replace=True)
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--backend', choices=('mock', 'local', 'api'), default='mock')
    p.add_argument('--model', type=Path)
    p.add_argument('--api-config', type=Path)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--max-samples', type=int)
    a = p.parse_args()
    import json
    result = run(a.package, a.output, backend=a.backend, model=a.model, api_config=a.api_config,
                 resume=a.resume, max_samples=a.max_samples)
    print(json.dumps(result))
    raise SystemExit(0 if result['status'] == 'complete' and not result['halted'] else 2)

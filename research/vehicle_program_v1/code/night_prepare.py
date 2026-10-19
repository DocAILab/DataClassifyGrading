"""Prepare an offline night plan, preserving all pilot receipts and their charges."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import sys

from api_backend import api_generation, default_config, MODEL
from api_parallel import scheduler_config
from artifacts import BUDGET, code_inventory, fingerprint, put, read, sha
from prepare import prepare
from runner import read_receipts


def prepare_night(source, package, home, donors, *, mode='api'):
    package, home = Path(package), Path(home)
    prepare(source, package)
    study = read(package / 'inference/study.json')
    provenance = {'schema': 1, 'receipts': {}, 'donors': [], 'usage_plus_uncertain_reserves_cny': 0.}
    for donor_package, donor_run in donors:
        donor_package, donor_run = Path(donor_package), Path(donor_run)
        manifest = read(donor_package / 'manifest.json')
        identity = read(donor_run / 'run.json')
        if (identity['expected_ids'] != study['expected_ids'] or identity['budget'] != BUDGET
                or identity['study_sha256'] != sha(package / 'inference/study.json')
                or identity['generation'] != api_generation()
                or identity['code_sha256'] != fingerprint(manifest['code'])
                or identity['api_service']['model'] != MODEL
                or identity['api_service']['base_url'] != 'https://api.siliconflow.cn/v1'):
            raise ValueError('pilot_identity_differs')
        if {k: v for k, v in manifest['code'].items() if k.startswith('upstream/')} != {
                k: v for k, v in code_inventory().items() if k.startswith('upstream/')}:
            raise ValueError('pilot_controller_changed')
        receipts = read_receipts(donor_run, study['expected_ids'])
        if set(receipts) & set(provenance['receipts']):
            raise ValueError('duplicate_pilot_source')
        attempts = read(donor_run / 'backend/api-budget.json')['attempts']
        price = identity['api_service']['pricing_cny_per_million']
        carried = sum((a['usage']['prompt_tokens'] * price['input'] + a['usage']['completion_tokens'] * price['output']) / 1e6
                      if a['state'] == 'response_received' else a['reserved_cny'] for a in attempts)
        for n, ident in enumerate(study['expected_ids'], 1):
            if ident in receipts:
                src = donor_run / f'sources/{n:04d}'
                shutil.copytree(src, package / f'seed/sources/{n:04d}')
                provenance['receipts'][ident] = {'position': n, 'receipt_sha256': sha(src / 'receipt.json'),
                                                'origin_run_sha256': sha(donor_run / 'run.json'),
                                                'status': receipts[ident]['status']}
        provenance['donors'].append({'package': str(donor_package), 'run': str(donor_run),
                                     'manifest_sha256': sha(donor_package / 'manifest.json'),
                                     'run_sha256': sha(donor_run / 'run.json'),
                                     'api_budget_sha256': sha(donor_run / 'backend/api-budget.json'),
                                     'source_count': len(receipts), 'http_attempts': len(attempts),
                                     'carried_usage_plus_uncertain_reserves_cny': carried})
        provenance['usage_plus_uncertain_reserves_cny'] += carried
    put(package / 'seed-provenance.json', provenance)
    tokenizer = Path(os.environ.get('VEHICLE_NONTD_TOKENIZER', str(Path.home() / '.cache/vehicle-nontd/official-tokenizer')))
    config = default_config(tokenizer)
    config['limits'].update(min_interval_seconds=.000001, max_http_requests=(len(study['rows']) - len(provenance['receipts'])) * 16,
                            max_estimated_cny=70., max_wall_seconds=50400., request_timeout_seconds=180.)
    put(package / 'night-api-profile.json', config)
    scheduler = scheduler_config(); scheduler['workers'] = 2
    home.mkdir(parents=True, exist_ok=False)
    python = Path(sys.executable)
    plan = {'schema': 'vehicle-nontd-night-v1', 'mode': mode, 'package': str(package.resolve()),
            'package_manifest_sha256': sha(package / 'manifest.json'),
            'home': str(home.resolve()), 'output': str((home / 'run').resolve()), 'stop_file': str((home / 'STOP').resolve()),
            'files': {name: sha(package / name) for name in ('night-api-profile.json', 'seed-provenance.json')},
            'python': str(python), 'pythonw': str(python.with_name('pythonw.exe')), 'scheduler': scheduler,
            'supervisor_wall_seconds': 50400., 'manual_start_only': True, 'scheduled_trigger': None,
            'task_name': 'VeRL-NONTD-Night-' + fingerprint(str(home.resolve()))[:10],
            'policy': {'heartbeat_seconds': 5., 'transient_cooldown_seconds': 30.,
                       'rate_limit_cooldown_seconds': 120., 'consecutive_source_error_limit': 3,
                       'shutdown_grace_seconds': 210.}}
    put(home / 'plan.json', plan)
    return {'prepared': True, 'model_calls': 0, 'preserved_sources': len(provenance['receipts']),
            'pending_sources': len(study['rows']) - len(provenance['receipts']),
            'carried_usage_plus_uncertain_reserves_cny': provenance['usage_plus_uncertain_reserves_cny']}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--package', type=Path, required=True)
    p.add_argument('--home', type=Path, required=True)
    p.add_argument('--donors', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(prepare_night(a.source, a.package, a.home, read(a.donors)['donors'])))

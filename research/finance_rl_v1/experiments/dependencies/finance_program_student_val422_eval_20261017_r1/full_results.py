"""Local post-exit aggregation of two immutable archives; model-free controller replay."""
import os
from pathlib import Path
import subprocess
import sys
from io_utils import ROOT, read, put, sha, verify
from full_validation import ARMS, partition, merge_traces, combined_metrics, anchor_identity
from parent_acceptance import parent_evidence, member_recheck, archive_identity


def local_replay(extract, out, cohort):
    """Execute the replay script whose ROOT is the extracted package, never the live package."""
    extract, out = Path(extract), Path(out)
    folder = out/(cohort+'-replay-%03d' % len(list(out.glob(cohort+'-replay-*'))))
    folder.mkdir(parents=True, exist_ok=False); report = folder/'proof.json'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    p = subprocess.run([sys.executable,'-X','utf8','-B',str(extract/'replay.py'),
        '--run',str(extract/'runs'/cohort),'--output',str(report)], cwd=extract, env=env,
        text=True, capture_output=True, encoding='utf-8')
    put(folder/'exit.json', {'exit_code':p.returncode, 'stdout':p.stdout, 'stderr':p.stderr})
    if p.returncode: raise RuntimeError('local full replay failed: '+cohort)
    value = read(report); remote = read(extract/'reviews'/(cohort+'-audit.json'))
    if remote.get('actual_route_and_tokens_verified') is not True: raise ValueError('remote tokenizer replay missing')
    for key, item in value.items():
        if key != 'actual_route_and_tokens_verified' and remote.get(key) != item:
            raise ValueError('remote/local replay differs: '+key)
    return {'result':value, 'proof_sha256':sha(report), 'exit_sha256':sha(folder/'exit.json'),
            'remote_tokenizer_proof_sha256':sha(extract/'reviews'/(cohort+'-audit.json'))}


def collect(root, cohort, ids, *, complete):
    root = Path(root); traces = {a:{} for a in ARMS}; hashes = {a:{} for a in ARMS}
    source_root = root/'runs'/cohort/'sources'
    if source_root.exists():
        actual = {p.name for p in source_root.iterdir()}
        if not actual <= set(ids) or (complete and actual != set(ids)):
            raise ValueError('source directory coverage differs: '+cohort)
        for source in source_root.iterdir():
            arms = {p.name for p in source.iterdir()} if source.is_dir() else set()
            if not source.is_dir() or not arms <= set(ARMS) or (complete and arms != set(ARMS)):
                raise ValueError('source arm directories differ: '+source.name)
    resources = {}; per_id = {}
    for arm in ARMS:
        total = {'calls':0, 'input_tokens':0, 'output_tokens':0, 'generation_seconds':0.,
                 'source_execution_seconds':0., 'planner_adapter_calls':0, 'backend_errors':0}
        per_id[arm] = {}
        for ident in ids:
            folder = root/'runs'/cohort/'sources'/ident/arm
            path = folder/'trace.json'
            if not path.exists():
                if complete: raise ValueError('missing trace: '+ident+'/'+arm)
                continue
            traces[arm][ident] = read(path); hashes[arm][ident] = sha(path)
            cost = dict.fromkeys(total, 0); receipt = folder/'receipt.json'
            if receipt.exists():
                r = read(receipt)
                if r['id'] != ident or r['arm'] != arm: raise ValueError('source receipt identity differs')
                cost['source_execution_seconds'] = r['seconds']
            elif complete: raise ValueError('missing source receipt')
            for call in sorted((folder/'calls').glob('call-*')):
                if not (call/'result.json').is_file() or not (call/'route.json').is_file():
                    if complete: raise ValueError('incomplete call records')
                    continue
                result, route = read(call/'result.json'), read(call/'route.json')
                cost['calls'] += 1; cost['planner_adapter_calls'] += int(route['enabled'])
                cost['backend_errors'] += int(bool(result.get('error')))
                cost['generation_seconds'] += result['seconds']
                for key in ('input_tokens','output_tokens'): cost[key] += result.get(key) or 0
            per_id[arm][ident] = cost
            for key in total: total[key] += cost[key]
        resources[arm] = total
    return traces, hashes, resources, per_id


def result(extract, out, receipt, *, root=ROOT):
    root, extract, out = Path(root), Path(extract), Path(out)
    verify(extract)
    if sha(extract/'manifest.json') != sha(root/'manifest.json'): raise ValueError('new frozen manifest differs')
    members = read(out/'members.json'); member_recheck(extract, members)
    study = read(extract/'data/study.json'); parts = partition(study)
    oldout = root.parent/'finance_program_student_eval_20261016_r2/reviews/final-01'; old = oldout/'extract'
    parent = parent_evidence(old, oldout/'student-val80-evidence.tar.gz', oldout/'members.json',
                             read(oldout/'archive.json'), read(oldout/'archive.exit.json'), current=extract)
    if parent != read(extract/'data/reuse-parent.json'): raise ValueError('local original archive differs from frozen parent')
    replays = {name:local_replay(old, out, name) for name in ('smoke','val80')}
    failures = []
    for name in ('val342','val342-audit','exit'):
        p = extract/'runs'/(name+'.exit.json' if name != 'exit' else 'exit.json')
        if not p.is_file() or read(p).get('exit_code') != 0: failures.append(name)
    if read(out/'archive.exit.json').get('exit_code') != 0 or receipt['exit']['exit_code'] != 0:
        failures.append('archive')
    complete = not failures
    if complete:
        completion = read(extract/'runs/completion.json')
        expected = {'supplementary_sources':342, 'supplementary_traces':684, 'reused_formal_sources':80,
                    'reused_formal_traces':160, 'expected_full_sources':422, 'expected_full_traces':844,
                    'supplementary_complete':True, 'full_replay_accepted':True}
        if any(completion.get(k) != v for k,v in expected.items()): raise ValueError('supplement completion differs')
        if completion['evaluation_manifest_sha256'] != sha(extract/'manifest.json'): raise ValueError('completion manifest differs')
        anchor_identity(parent['training_anchor'], read(extract/'runs/training-anchor.json'))
        acceptance = read(extract/'runs/parent-acceptance.json')
        for key in ('parent_manifest_sha256','parent_archive_sha256','parent_members_sha256','parent_anchor_sha256'):
            if acceptance[key] != parent[key]: raise ValueError('remote parent acceptance differs')
        from parent_acceptance import TRAINING_FILES
        for name,key in TRAINING_FILES.items():
            if sha(extract/'evidence/training'/name) != parent['training_anchor'][key]: raise ValueError('new training copy differs')
        replays['val342'] = local_replay(extract, out, 'val342')
        if replays['val342']['result']['sources'] != 342 or replays['val342']['result']['traces'] != 684:
            raise ValueError('new replay coverage differs')
    oldtraces, oldhashes, oldcost, oldper = collect(old, 'val80', parts['val80'], complete=True)
    newtraces, newhashes, newcost, newper = collect(extract, 'val342', parts['val342'], complete=complete)
    traces = merge_traces(parts, oldtraces, newtraces, complete=complete)
    localgold = root/'reviews/local-gold422.json'; selection = read(root/'reviews/full422-selection-proof.json')
    if sha(localgold) != selection['local_gold422_sha256']: raise ValueError('local gold changed since preparation')
    gold = read(localgold)
    value = combined_metrics(parts, traces, gold, study['registry'], complete=complete)
    allhashes = {i:{a:(oldhashes if i in set(parts['val80']) else newhashes)[a].get(i) for a in ARMS}
                 for i in parts['full422']}
    for ident in parts['val80']:
        if allhashes[ident] != parent['old80_trace_sha256'][ident]: raise ValueError('old formal trace SHA differs')
    costs = {a:{key:oldcost[a][key]+newcost[a][key] for key in oldcost[a]} for a in ARMS}
    if complete:
        for cohort,cost in (('val80',oldcost),('val342',newcost)):
            for arm in ARMS:
                for key in ('calls','input_tokens','output_tokens'):
                    if cost[arm][key] != replays[cohort]['result'][arm+'_'+key]: raise ValueError('resource replay totals differ')
    # Recheck all extracted bytes independently after replay and scoring as well.
    member_recheck(old, parent['parent_members']); member_recheck(extract, members)
    wall = {'reused_val80_wall_seconds':read(old/'runs/val80/summary.json')['seconds'],
            'supplement342_wall_seconds':read(extract/'runs/val342/summary.json')['seconds']
                if (extract/'runs/val342/summary.json').exists() else None,
            'these_are_separate_stages_not_one_wall_measurement':True}
    value.update(fixed_full422_ids=parts['full422'], supported_classes=sorted({v['category'] for v in gold.values()}),
        local_gold_sha256=sha(localgold), gold_local_only=True, resources=costs,
        resources_by_cohort={'val80':oldcost,'val342':newcost},
        resources_per_id={a:{**oldper[a],**newper[a]} for a in ARMS}, wall=wall,
        trace_sha256_per_id_arm=allhashes, local_full_replays=replays,
        provenance={'parent_archive_sha256':parent['parent_archive_sha256'],
            'parent_members_sha256':parent['parent_members_sha256'], 'parent_manifest_sha256':parent['parent_manifest_sha256'],
            'parent_anchor_sha256':parent['parent_anchor_sha256'],
            'new_archive_sha256':receipt['archive_sha256'], 'new_members_sha256':receipt['members_sha256'],
            'new_manifest_sha256':sha(extract/'manifest.json'),
            'frozen_reuse_proof_sha256':sha(extract/'data/reuse-parent.json'),
            'parent_all_members_independently_rechecked':True, 'new_all_members_independently_rechecked':True},
        limits=['Development Val422, single seed; not held-out Test or independent generalization.',
                'Only eight of twelve classes represented; sources may share tables.',
                'Old80 reused exactly including failures; old engineering smoke excluded.',
                'No further training, outcome-based tuning, or automatic RL continuation.'])
    if failures:
        value.update(failed_stages=failures, must_not_report_partial_as_completed=True,
                     engineering_failure=read(extract/'runs/failure.json') if (extract/'runs/failure.json').exists() else None)
    return value

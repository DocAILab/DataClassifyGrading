"""Read-only acceptance of the completed first-SFT evaluation and its training anchor."""
from pathlib import Path
import tarfile
from io_utils import ROOT, read, sha, verify
from full_validation import anchor_identity, reuse_identity, safe_members, partition
from gate import cpu_proof_identity

PARENT_MANIFEST = 'ff9882ae30893ee623f1528e17510c4bb1bd441733c6fc7083246fdb5efa7711'
PARENT_ARCHIVE = '3c093681b1e2e406e33799609a4955d1b4d0d7c8a01649d83471e7c67dcda1be'
ADAPTER = '81b18e85c5ce9ceb9585cf069a87d389567a2be47198c35a0d79d73f850a2abc'
ADAPTER_CONFIG = 'df63e488cafe6b804c78a5f289e2d61d0ffae17893e8aebb9cfe6fe53f5bf757'
TRAINING_FILES = {'manifest.json':'training_manifest_sha256', 'completion.json':'completion_sha256',
                  'cpu-proof.json':'cpu_proof_sha256', 'adapter_model.safetensors':'adapter_sha256',
                  'adapter_config.json':'adapter_config_sha256'}
STAGES = ('smoke', 'smoke-audit', 'val80', 'val80-audit')


def execution_config_identity(original, current):
    overrides = {'pair_count':844, 'stage_timeout_seconds':43200}
    for key, value in original.items():
        if current.get(key) != overrides.get(key, value):
            raise ValueError('original execution setting differs: ' + key)


def process_live(pid, needle):
    p = Path('/proc') / str(pid)
    try:
        return bool(p.exists() and needle in (p/'cmdline').read_bytes().decode(errors='replace')
                    and (p/'stat').read_text().split()[2] != 'Z')
    except (FileNotFoundError, ProcessLookupError): return False


def member_recheck(root, entries):
    root = Path(root).resolve()
    for item in entries:
        p = root / item['path']
        if not p.resolve().is_relative_to(root) or not p.is_file() or p.stat().st_size != item['bytes'] or sha(p) != item['sha256']:
            raise ValueError('independent member recheck differs: ' + item['path'])


def archive_identity(archive, members, receipt, *, expected_archive=None):
    archive, members = Path(archive), Path(members)
    if (expected_archive and sha(archive) != expected_archive) or sha(archive) != receipt['archive_sha256']:
        raise ValueError('archive SHA differs')
    if sha(members) != receipt['members_sha256'] or archive.stat().st_size != receipt['bytes']:
        raise ValueError('archive receipt differs')
    entries = read(members); expected = {x['path']:x for x in entries}
    if len(entries) != len(expected): raise ValueError('duplicate archive manifest entries')
    with tarfile.open(archive, 'r:gz') as tar:
        if len(safe_members(tar, expected, receipt['members_sha256'])) != receipt['members']:
            raise ValueError('archive member count differs')
    return entries


def writers_stopped(root):
    root = Path(root)
    writers = [(root/'runs/worker.json', 'watch.py')]
    writers.extend((root/'runs'/(s+'.pid.json'), 'replay.py' if s.endswith('-audit') else 'evaluate.py') for s in STAGES)
    for path, needle in writers:
        if process_live(read(path)['pid'], needle): raise ValueError('parent writer is still live')


def parent_evidence(root, archive, members, receipt, archive_exit, *, current=ROOT, check_writers=False, study_override=None):
    """Accept original80 including failed classifications; engineering stages must all succeed."""
    root, current = Path(root), Path(current)
    old = verify(root)
    if sha(root/'manifest.json') != PARENT_MANIFEST: raise ValueError('original evaluation manifest differs')
    entries = archive_identity(archive, members, receipt, expected_archive=PARENT_ARCHIVE)
    member_recheck(root, entries)
    if archive_exit.get('exit_code') != 0 or receipt['exit']['exit_code'] != 0:
        raise ValueError('parent archive or writer failed')
    for name in (*STAGES, 'exit'):
        path = root/'runs'/(name+'.exit.json' if name != 'exit' else 'exit.json')
        if read(path)['exit_code'] != 0: raise ValueError('parent stage failed: ' + name)
    if check_writers:
        writers_stopped(root)
    completion = read(root/'runs/completion.json')
    if (completion.get('formal_traces') != 160 or completion.get('smoke_traces') != 4
            or completion.get('paired_val80_sources') != 80 or completion.get('full_replay_accepted') is not True
            or completion.get('evaluation_manifest_sha256') != PARENT_MANIFEST):
        raise ValueError('parent formal completion differs')
    study = read(root/'data/study.json'); newstudy = study_override if study_override is not None else read(current/'data/study.json')
    oldfiles = {x['path']:x['sha256'] for x in old['files']}
    newfiles = {n:sha(current/n) for n in oldfiles if (current/n).is_file()}
    reuse_identity(study, newstudy, oldfiles, newfiles)
    parts = partition(study); anchor = read(root/'runs/training-anchor.json')
    anchor_identity(anchor, completion['training_anchor'])
    if anchor['adapter_sha256'] != ADAPTER or anchor['adapter_config_sha256'] != ADAPTER_CONFIG:
        raise ValueError('first SFT adapter identity differs')
    for name, key in TRAINING_FILES.items():
        if sha(root/'evidence/training'/name) != anchor[key]: raise ValueError('training evidence differs: ' + name)
    proof = read(root/'evidence/training/cpu-proof.json')
    training = read(root/'evidence/training/completion.json')
    if not cpu_proof_identity(proof, anchor['adapter_config_sha256'], training['adapter_config_sha256'],
                              sha(root/'verify_training.py'), PARENT_MANIFEST):
        raise ValueError('original training validator identity differs')
    if not proof.get('independent_cpu_audit_passed') or not proof.get('base_all_files_sha_unchanged'):
        raise ValueError('original training proof did not pass')
    trace_hashes = {}
    for cohort, count in (('smoke', 2), ('val80', 80)):
        audit = read(root/'reviews'/(cohort+'-audit.json'))
        summary = read(root/'runs'/cohort/'summary.json')
        if (audit.get('sources') != count or audit.get('traces') != count*2
                or not all(audit.get(k) is True for k in ('full_trace_exact', 'all_calls_consumed', 'actual_route_and_tokens_verified'))
                or summary['expected_ids'] != study['cohorts'][cohort] or summary['paired_sources'] != count
                or summary['status'] != 'completed'):
            raise ValueError('parent full replay/coverage differs: ' + cohort)
        if cohort == 'val80':
            trace_hashes = {i:{arm:sha(root/'runs/val80/sources'/i/arm/'trace.json') for arm in ('base','sft')}
                            for i in parts['val80']}
    return {'parent_manifest_sha256':PARENT_MANIFEST, 'parent_archive_sha256':PARENT_ARCHIVE,
            'parent_members_sha256':receipt['members_sha256'], 'parent_member_count':len(entries),
            'parent_anchor_sha256':sha(root/'runs/training-anchor.json'), 'training_anchor':anchor,
            'parent_frozen_files':oldfiles, 'parent_members':entries, 'old80_trace_sha256':trace_hashes,
            'parent_replay_sha256':{s:sha(root/'reviews'/(s+'-audit.json')) for s in ('smoke','val80')},
            'val80':parts['val80'], 'old_smoke_excluded_from_formal':True}


def actual_training(c, anchor):
    root = Path(c['training_root']); adapter = Path(c['adapter'])
    sources = {'training_manifest_sha256':root/'manifest.json', 'completion_sha256':root/'formal/completion.json',
               'cpu_proof_sha256':root/'reviews/post-training-cpu-proof.json',
               'adapter_sha256':adapter/'adapter_model.safetensors', 'adapter_config_sha256':adapter/'adapter_config.json'}
    anchor_identity(anchor, {key:sha(path) for key,path in sources.items()})


def model_identity(c, root=ROOT):
    model = Path(c['model']); identity = read(Path(root)/'model-identity.json')
    if read(model/'model-manifest.json') != identity: raise ValueError('actual base model manifest differs')
    for item in identity['files']:
        p = model/item['path']
        if p.stat().st_size != item['bytes'] or sha(p) != item['sha256']:
            raise ValueError('actual base model file SHA differs: ' + item['path'])
    return {'all_base_files_sha_verified':True, 'files':len(identity['files']), 'revision':identity['revision']}

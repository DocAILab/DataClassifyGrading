"""Offline audit/materialization only. No train, launch or remote command exists."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

from dataset_contract import encode_completion, load_rows, partition, sha, validate_program

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
MERGE = PARENT / 'finance_program_teacher_merge_20261022_r1'
EVAL = PARENT / 'finance_program_student_r9_eval80_20261022_r1'
MERGE_MANIFEST_SHA = 'c72268a776cf6bccbbe45423881094f181f5c83dcc3090b9632f2c2adedb578d'
IDENTITY = PARENT / 'finance_program_student_b80_20261021_r3/model-identity.json'
IDENTITY_SHA = '5b0a6b5e607d5305b2d59dd3e04bbd0f8b04af4502c389be3a5d83d01eddc8c9'


def read_bound(path, ref):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('bound regular local file required: ' + str(path))
    raw = path.read_bytes()
    if sha(raw) != ref['sha256'] or len(raw) != ref['bytes']:
        raise ValueError('bound file SHA/size changed: ' + str(path))
    return raw


def binding(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return dict(path=str(path), bytes=len(raw), sha256=sha(raw))


def load_bundle():
    manifest_path = MERGE / 'reviews/delivery-manifest-01.json'
    raw = manifest_path.read_bytes()
    if sha(raw) != MERGE_MANIFEST_SHA:
        raise ValueError('original merge delivery manifest SHA changed')
    manifest = json.loads(raw)
    refs = {str(Path(r['path']).resolve()): r for r in manifest['files']}
    paths = [MERGE / 'data/planner-curated.jsonl', MERGE / 'reviews/curation-index-01.json', EVAL / 'selection.json']
    payloads = [read_bound(p, refs[str(p.resolve())]) for p in paths]
    rows = load_rows(payloads[0], json.loads(payloads[1]))
    if len(rows) != 162:
        raise ValueError('the sealed 162-row data pool is required')
    selection = json.loads(payloads[2])
    if len(selection['selected']) != 80:
        raise ValueError('original fixed 80-source evaluation required')
    return rows, selection['selected'], [binding(p) for p in [manifest_path] + paths]


def audit():
    rows, eval_rows, refs = load_bundle()
    disjoint, overlap = partition(rows, 'exclude_eval_groups', eval_rows)
    summary = dict(schema='student-merge162-offline-audit-v1', utc=datetime.now(timezone.utc).isoformat(),
                   inputs=refs, examples=len(rows), states=dict(Counter(r['supervision_state'] for r in rows)),
                   sources=len({r['source_id'] for r in rows}), groups=len({r['group'] for r in rows}),
                   program_steps=dict(Counter(len(validate_program(r['completion'])) for r in rows)),
                   program_step_cap=16, six_step_cap=False,
                   current_eval_overlap_examples=len(overlap),
                   current_eval_overlap_groups=len({r['group'] for r in overlap}),
                   current_eval_overlap_ids=[r['id'] for r in overlap],
                   exclude_eval_groups_candidate_examples=len(disjoint),
                   exclude_eval_groups_candidate_groups=len({r['group'] for r in disjoint}),
                   note='Neither policy is frozen for training. Original A/B80 exposure remains separately relevant; group disjointness does not imply fresh Test.',
                   tokenizer_checked=False, train_ready=False, training_started=False, GPU_calls=0)
    return rows, eval_rows, summary


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as out:
        json.dump(value, out, ensure_ascii=False, indent=2)
        out.write('\n')


def write_prepared(rows, report, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    path = output / 'train-candidates.jsonl'
    with path.open('x', encoding='utf-8', newline='\n') as out:
        for row in rows:
            material = {k: row[k] for k in ('id', 'messages', 'completion')}
            out.write(json.dumps(material, ensure_ascii=False, separators=(',', ':')) + '\n')
    result = dict(report, candidates=len(rows), candidate_file=binding(path),
                  candidate_ids=[r['id'] for r in rows], tokenizer_checked=False, train_ready=False)
    write_json(output / 'preparation.json', result)
    return result


def load_original_tokenizer(directory):
    """Only the six pinned local tokenizer/config files; never download/model-load."""
    directory = Path(directory).resolve(strict=True)
    raw = IDENTITY.read_bytes()
    if sha(raw) != IDENTITY_SHA:
        raise ValueError('original model identity SHA changed')
    identity = json.loads(raw)
    names = {'config.json', 'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja', 'vocab.json', 'merges.txt'}
    refs = {r['path']: r for r in identity['files']}
    for name in names:
        read_bound(directory / name, refs[name])
    for path in directory.iterdir():
        n = path.name.lower()
        if n.startswith(('tokenizer', 'tokenization_', 'configuration_', 'chat_template',
                         'added_token', 'special_token', 'vocab', 'merges', 'spiece', 'sentencepiece')) and path.name not in names:
            raise ValueError('unbound tokenizer helper/override: ' + path.name)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError('original fast tokenizer required')
    return tokenizer, [binding(directory / name) for name in sorted(names)]


def tokenizer_audit(rows, tokenizer, max_length):
    details = []
    for row in rows:
        try:
            encoded = encode_completion(tokenizer, row, max_length)
            details.append(dict(id=row['id'], passed=True, total_tokens=len(encoded['input_ids']),
                                prompt_tokens=encoded['prompt_length'], target_tokens=encoded['target_length'],
                                truncated=False, suffix_exact=True, prompt_masked=True))
        except ValueError as error:
            details.append(dict(id=row['id'], passed=False, error=str(error), truncated=False))
    return details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['audit', 'prepare', 'tokenize'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--eval-policy', choices=['diagnostic_only', 'exclude_eval_groups'])
    parser.add_argument('--tokenizer-dir', type=Path)
    parser.add_argument('--max-length', type=int)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output already exists; never overwrite')
    if args.command in ('prepare', 'tokenize') and not args.eval_policy:
        parser.error('--eval-policy must be explicit')
    rows, eval_rows, report = audit()
    if args.command == 'audit':
        write_json(args.output, report)
    else:
        selected, overlap = partition(rows, args.eval_policy, eval_rows)
        report.update(eval_policy=args.eval_policy, selected_overlap_examples=sum(r in overlap for r in selected),
                      selected_ids=[r['id'] for r in selected])
        if args.command == 'prepare':
            report = write_prepared(selected, report, args.output)
        else:
            if args.tokenizer_dir is None or args.max_length is None:
                parser.error('tokenize requires original --tokenizer-dir and explicit --max-length')
            tokenizer, refs = load_original_tokenizer(args.tokenizer_dir)
            details = tokenizer_audit(selected, tokenizer, args.max_length)
            report.update(tokenizer_checked=True, tokenizer_files=refs, per_example=details,
                          max_length=args.max_length, all_selected_fit=all(r['passed'] for r in details),
                          train_ready=False)
            write_json(args.output, report)
            if not report['all_selected_fit']:
                raise SystemExit('tokenizer audit failed; evidence retained, no truncation or training')
    print(json.dumps({k: report[k] for k in ('examples', 'states', 'current_eval_overlap_examples',
                                             'current_eval_overlap_groups', 'train_ready')}))


if __name__ == '__main__': main()

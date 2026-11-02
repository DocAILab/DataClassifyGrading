"""Fixed CoreSkill validation: model Top-5 retrieval, then model Top-1 and grade."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request
import urllib.error

ROOT = Path(__file__).resolve().parent
REVISION = '84a9a462dcb828332f980ef121f40e1281ce1794'
MODEL = 'Qwen/Qwen3.5-9B'
ENDPOINT = 'https://api.siliconflow.cn/v1/chat/completions'
PATH_KEYS = ('category_root_level', 'category_branch_level', 'category_subbranch_level', 'category_leaf_level')


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def category_path(row):
    return [row[k] for k in PATH_KEYS if isinstance(row.get(k), str) and row[k].strip() and row[k] != '--']


def prepare():
    details = {}
    for domain, folder, count in [('VEH', 'DCG_VEHICLE', 397), ('FIN', 'DCG_FIN', 422)]:
        hashes = {}
        for suffix in ('CORPUS', 'VAL'):
            name = f'{folder}/{folder}_{suffix}.json'
            path = ROOT / domain / 'inputs' / f'{suffix}.json'
            if not path.exists():
                url = f'https://huggingface.co/datasets/DocAILab/DCG/resolve/{REVISION}/{name}'
                from huggingface_hub import get_token
                token = get_token()
                request = urllib.request.Request(url, headers={'Authorization': 'Bearer ' + token} if token else {})
                with urllib.request.urlopen(request, timeout=120) as response:
                    content = response.read()
                json.loads(content.decode('utf-8-sig'))
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
            hashes[name] = digest(path)
        rows = read(ROOT / domain / 'inputs/VAL.json')
        corpus = read(ROOT / domain / 'inputs/CORPUS.json')
        assert len(rows) == count and len({r['id'] for r in rows}) == count
        registry = []
        for i, entry in enumerate(corpus, 1):
            registry.append({'choice_id': str(i), 'standard_id': entry['id'], 'path': category_path(entry),
                             'description': entry['data_description_and_example'],
                             'reference_standard': entry['reference_standard'],
                             'level': entry['sensitivity_level']})
        assert len(registry) >= 5 and len({r['standard_id'] for r in registry}) == len(registry)
        for row in rows:
            assert any(e['path'] == category_path(row['classification']) and e['level'] == row['grading']['sensitivity_level'] for e in registry)
        put(ROOT / domain / 'registry.json', registry)
        details[domain] = {'count': count, 'standards': len(registry),
                           'leaf_paths': len({tuple(e['path']) for e in registry}), 'files': hashes}
    manifest = {'experiment': 'NOTRAIN-B / NONT-B / CoreSkill', 'dataset': 'DocAILab/DCG',
                'revision': REVISION, 'split': 'VAL', 'model': MODEL, 'endpoint': ENDPOINT,
                'temperature': 0, 'max_tokens': 512, 'enable_thinking': False,
                'initial_metadata_fields': ['domain', 'field_name'],
                'candidate_unit': 'standard_entry; duplicate leaf paths may have different levels',
                'algorithm': 'Fixed two model Skills: retrieve exactly 5 unique standard entries from the full catalog; select top-1 from those entries and output its corresponding sensitivity level. No Planner, other Skills, SFT or RL. Both Skills see only domain and field_name; stage 2 additionally sees the selected standards descriptions and grades. Labels are used only by the evaluator.',
                'parse_policy': 'strict JSON, exact keys, no repairs or semantic retries',
                'http_retry': 'up to 3 attempts only for 429, 5xx or network errors; all attempts retained',
                'denominator': 'all validation sources including failed and invalid predictions',
                'provider_revision': 'not exposed; deterministic output not guaranteed', 'datasets': details,
                'runner_sha256': digest(Path(__file__)),
                'prompts_sha256': hashlib.sha256((S1 + S2).encode()).hexdigest()}
    target = ROOT / 'manifest.json'
    if target.exists() and read(target) != manifest:
        raise ValueError('existing manifest differs; do not mix experiments')
    put(target, manifest)
    return manifest


S1 = ('你是数据分类分级系统的候选标准检索Skill。依据领域和字段名，从完整标准目录中选择最相关的5个标准条目，按相关性降序。'
      '仅输出JSON对象，恰好包含candidates键，值为5个互不重复的目录id字符串。不得输出说明、Markdown或其他键。')
S2 = ('你是数据分类分级系统的候选决策Skill。依据领域、字段名和5个候选的标准描述，选择最匹配的一个标准条目，'
      '并输出该条目对应的敏感级别。仅输出JSON对象，恰好包含answer和level两个键。answer必须是候选id字符串，'
      'level必须严格等于所选条目的level。不得输出说明、Markdown或其他键。')


def messages(stage, metadata, registry):
    if stage == 1:
        catalog = [{'id': e['choice_id'], 'category_path': e['path']} for e in registry]
        body = {'metadata': metadata, 'standard_catalog': catalog}
    else:
        catalog = [{'id': e['choice_id'], 'category_path': e['path'], 'description': e['description'],
                    'reference_standard': e['reference_standard'], 'level': e['level']} for e in registry]
        body = {'metadata': metadata, 'candidates': catalog}
    return [{'role': 'system', 'content': S1 if stage == 1 else S2},
            {'role': 'user', 'content': json.dumps(body, ensure_ascii=False, separators=(',', ':'))}]


def call(msg, folder):
    body = {'model': MODEL, 'messages': msg, 'temperature': 0, 'max_tokens': 512, 'enable_thinking': False}
    put(folder / 'request.json', body)
    for attempt in range(1, 4):
        started = time.monotonic()
        req = urllib.request.Request(ENDPOINT, data=json.dumps(body).encode(),
                                     headers={'Authorization': 'Bearer ' + os.environ['SILICONFLOW_KEY'], 'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=180) as response:
                data = json.load(response)
            put(folder / f'attempt-{attempt}.json', {'status': 'success', 'seconds': time.monotonic() - started, 'response': data})
            return data['choices'][0]['message']['content']
        except urllib.error.HTTPError as exc:
            status = exc.code
            put(folder / f'attempt-{attempt}.json', {'status': 'http_error', 'http_status': status, 'seconds': time.monotonic() - started})
            if status != 429 and status < 500:
                raise RuntimeError(f'HTTP_{status}') from None
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            put(folder / f'attempt-{attempt}.json', {'status': 'network_error', 'error_type': type(exc).__name__, 'seconds': time.monotonic() - started})
        if attempt < 3:
            time.sleep(5 * attempt)
    raise RuntimeError('transport_retries_exhausted')


def parse(text, keys):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate_key')
            result[key] = value
        return result
    obj = json.loads(text, object_pairs_hook=unique)
    if not isinstance(obj, dict) or set(obj) != set(keys):
        raise ValueError('invalid_keys')
    return obj


def one(domain, index, source, registry):
    folder = ROOT / domain / 'sources' / f'{index:04d}'
    if (folder / 'result.json').exists():
        result = read(folder / 'result.json')
        assert result['source_id'] == source['id']
        return result
    if folder.exists():
        # Preserve an interrupted request without silently re-generating the source.
        result = {'source_id': source['id'], 'status': 'interrupted', 'predicted_path': None, 'predicted_level': None}
        put(folder / 'result.json', result)
        return result
    folder.mkdir(parents=True)
    metadata = {'domain': source['domain'], 'field_name': source['metadata']['field_name']}
    result = {'source_id': source['id'], 'status': 'started', 'predicted_path': None, 'predicted_level': None}
    started = time.monotonic()
    stage = 1
    try:
        output = call(messages(1, metadata, registry), folder / 'stage1')
        candidates = parse(output, ['candidates'])['candidates']
        ids = {e['choice_id']: e for e in registry}
        if not isinstance(candidates, list) or len(candidates) != 5 or any(not isinstance(x, str) or x not in ids for x in candidates) or len(set(candidates)) != 5:
            raise ValueError('invalid_candidates')
        result['candidates'] = candidates
        stage = 2
        output = call(messages(2, metadata, [ids[x] for x in candidates]), folder / 'stage2')
        answer = parse(output, ['answer', 'level'])
        if not isinstance(answer['answer'], str) or answer['answer'] not in candidates or not isinstance(answer['level'], str):
            raise ValueError('invalid_answer')
        selected = ids[answer['answer']]
        result.update(predicted_path=selected['path'], predicted_level=answer['level'], standard_id=selected['standard_id'],
                      grade_matches_standard=answer['level'] == selected['level'], status='completed')
    except Exception as exc:
        result.update(status='failed', failed_stage=stage, error_type=type(exc).__name__)
        if isinstance(exc, RuntimeError):
            result['error_code'] = str(exc)
    result['seconds'] = time.monotonic() - started
    put(folder / 'result.json', result)
    return result


def f1_stats(truths, predictions):
    labels = set(truths)
    support = Counter(truths)
    scores = []
    for label in labels:
        tp = sum(t == label and p == label for t, p in zip(truths, predictions))
        fp = sum(t != label and p == label for t, p in zip(truths, predictions))
        fn = support[label] - tp
        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / support[label]
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0
        scores.append((precision, recall, f1, support[label]))
    return {'macro_precision': sum(x[0] for x in scores) / len(labels),
            'macro_recall': sum(x[1] for x in scores) / len(labels),
            'macro_f1': sum(x[2] for x in scores) / len(labels),
            'weighted_f1': sum(x[2] * x[3] for x in scores) / len(truths), 'gt_labels': len(labels)}


def score(domain):
    rows = read(ROOT / domain / 'inputs/VAL.json')
    registry = read(ROOT / domain / 'registry.json')
    results = []
    for i, row in enumerate(rows, 1):
        file = ROOT / domain / 'sources' / f'{i:04d}/result.json'
        results.append(read(file) if file.exists() else {'status': 'pending'})
    truths = [tuple(category_path(r['classification'])) for r in rows]
    levels = [r['grading']['sensitivity_level'] for r in rows]
    predictions = [tuple(r['predicted_path']) if r.get('predicted_path') else None for r in results]
    predicted_levels = [r.get('predicted_level') for r in results]
    usage = Counter()
    http_attempts = 0
    for path in (ROOT / domain / 'sources').glob('*/stage*/attempt-*.json'):
        http_attempts += 1
        data = read(path)
        if data['status'] == 'success':
            usage.update({k: v for k, v in data['response'].get('usage', {}).items() if k in ('prompt_tokens', 'completion_tokens', 'total_tokens') and isinstance(v, int)})
    n = len(rows)
    category_correct = sum(t == p for t, p in zip(truths, predictions))
    level_correct = sum(t == p for t, p in zip(levels, predicted_levels))
    joint_correct = sum(t == p and l == q for t, p, l, q in zip(truths, predictions, levels, predicted_levels))
    known_paths = {tuple(e['path']) for e in registry}
    known_levels = {e['level'] for e in registry}
    metrics = {'domain': domain, 'split': 'VAL', 'denominator': n, 'statuses': dict(Counter(r['status'] for r in results)),
               'exact_match_category': category_correct / n, 'level_accuracy': level_correct / n, 'exact_match_joint': joint_correct / n,
               'category_correct': category_correct, 'level_correct': level_correct, 'joint_correct': joint_correct,
               'terminal_valid_count': sum(r['status'] == 'completed' for r in results),
               'terminal_parse_rate': sum(r['status'] == 'completed' for r in results) / n,
               'category_valid_rate': sum(p in known_paths for p in predictions) / n,
               'level_valid_rate': sum(p in known_levels for p in predicted_levels) / n,
               'stage1_recall_at5': sum(any(tuple(e['path']) == t for e in registry if e['choice_id'] in r.get('candidates', [])) for t, r in zip(truths, results)) / n,
               'category': f1_stats(truths, predictions), 'level': f1_stats(levels, predicted_levels),
               'joint': f1_stats(list(zip(truths, levels)), [(p, l) if p else None for p, l in zip(predictions, predicted_levels)]),
               'http_attempts': http_attempts, 'usage': dict(usage),
               'average_input_tokens_per_sample': usage['prompt_tokens'] / n,
               'average_output_tokens_per_sample': usage['completion_tokens'] / n}
    put(ROOT / domain / 'metrics.json', metrics)
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--limit', type=int)
    a = p.parse_args()
    manifest = prepare()
    print(json.dumps(manifest['datasets'], ensure_ascii=False), flush=True)
    if a.prepare_only:
        return
    if not os.environ.get('SILICONFLOW_KEY'):
        raise RuntimeError('SILICONFLOW_KEY missing')
    tasks = []
    for domain in ('VEH', 'FIN'):
        rows = read(ROOT / domain / 'inputs/VAL.json')
        registry = read(ROOT / domain / 'registry.json')
        for i, source in enumerate(rows, 1):
            if a.limit and i > a.limit:
                break
            tasks.append((domain, i, source, registry))
    begin = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures = {pool.submit(one, *task): task[:2] for task in tasks}
        for done, f in enumerate(as_completed(futures), 1):
            result = f.result()
            print(f'{done}/{len(tasks)} {futures[f]} {result["status"]}', flush=True)
            if done % 20 == 0:
                put(ROOT / 'progress.json', {'finished': done, 'scheduled': len(tasks), 'elapsed_seconds': time.time() - begin})
    metrics = {domain: score(domain) for domain in ('VEH', 'FIN')}
    put(ROOT / 'summary.json', {'metrics': metrics, 'wall_seconds_this_invocation': time.time() - begin})
    print(json.dumps(metrics, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

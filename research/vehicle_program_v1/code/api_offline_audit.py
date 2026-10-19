"""Audit API preparation with local tokenizer and scripted Val traces; never opens a network."""
import argparse
from collections.abc import Mapping
import json
from pathlib import Path
from unittest.mock import patch

from api_backend import ReferenceCounter, load_config
from api_check import synthetic_messages
from artifacts import load_study, put, sha
from replay import replay
from runner import run


def audit(package, output):
    package, output = Path(package), Path(output)
    config = load_config(package / 'api-profile.json')
    study, _ = load_study(package)
    counter = ReferenceCounter(config['tokenizer_path'])
    prompts = synthetic_messages(counter)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(config['tokenizer_path'], local_files_only=True, trust_remote_code=False)
    comparisons = []
    for messages in prompts:
        encoded = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
        ids = encoded['input_ids'] if isinstance(encoded, Mapping) else encoded
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        comparisons.append({'reference_tokens': counter(messages), 'transformers_tokens': len(ids)})
    if any(v['reference_tokens'] != v['transformers_tokens'] for v in comparisons):
        raise ValueError('reference counter differs from official AutoTokenizer')
    def no_network(*args, **kwargs):
        raise AssertionError('offline audit attempted network access')
    original_open = Path.open
    def no_inference_gold(path, *args, **kwargs):
        if path.name in ('gold.json', '敏感信息交接.md'):
            raise AssertionError('offline inference opened gold or credentials')
        return original_open(path, *args, **kwargs)
    with patch('urllib.request.OpenerDirector.open', no_network), patch.object(Path, 'open', no_inference_gold):
        first = run(package, output, max_samples=7)
        first_hashes = {str(i): sha(output / f'sources/{i:04d}/receipt.json') for i in range(1, 8)}
        finished = run(package, output, resume=True)
        hashes = {str(i): sha(output / f'sources/{i:04d}/receipt.json') for i in range(1, 8)}
        replayed = replay(package, output)
    lengths = []
    for call in sorted((output / 'sources').glob('*/calls/*.json')):
        data = json.loads(call.read_text(encoding='utf-8'))
        lengths.append(counter(data['messages']))
    report = {'scope': 'offline synthetic transport tests and scripted Val paths; not real model accuracy',
              'http_requests': 0, 'key_read': False, 'gold_opened_by_inference': False,
              'reference_tokenizer_vs_transformers': comparisons,
              'first_batch_committed': first['committed_count'], 'resume_receipts_unchanged': hashes == first_hashes,
              'scripted_run': finished, 'replay': replayed,
              'scripted_path_capacity': {'requests': len(lengths), 'min_input_tokens': min(lengths),
                  'max_input_tokens': max(lengths), 'over_32256': sum(n > 32256 for n in lengths),
                  'not_exhaustive_for_real_model_routes': True},
              'expected_count': len(study['rows']), 'quality_claim': False}
    if not report['resume_receipts_unchanged'] or finished['pending_count'] or not replayed['pass']:
        raise ValueError('offline receipt/replay validation failed')
    put(output.parent / 'api-offline-audit.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.package, args.output)))

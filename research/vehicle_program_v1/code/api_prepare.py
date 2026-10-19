"""Prepare a new immutable vehicle package and key-free API profile, without network access."""
import argparse
from pathlib import Path
import json

from api_backend import ReferenceCounter, default_config
from artifacts import put
from prepare import prepare


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--tokenizer', type=Path, required=True)
    args = p.parse_args()
    counter = ReferenceCounter(args.tokenizer)
    report = prepare(args.source, args.output)
    put(args.output / 'api-profile.json', default_config(args.tokenizer))
    put(args.output / 'api-preparation.json', {
        'offline_only': True, 'http_requests': 0, 'key_read': False,
        'reference_tokenizer_verified': True, 'synthetic_reference_tokens': counter([
            {'role': 'system', 'content': 'Return only JSON.'},
            {'role': 'user', 'content': 'Return {"ok": true}.'}]),
        'hosted_precision': 'not_verified', 'hosted_weight_hash_verified': False,
        'full_validation_authorized_by_this_script': False})
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()

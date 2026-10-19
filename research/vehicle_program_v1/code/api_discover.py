"""Read the authenticated SiliconFlow model list; no generation or private sample upload."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.error
import urllib.request

from api_backend import APIError, CREDENTIAL_FILE, NoRedirect, credential_from_text
from artifacts import put, require_private_output


def discover(output):
    output = Path(output)
    require_private_output(output)
    key = credential_from_text(CREDENTIAL_FILE.read_text(encoding='utf-8'), '汽车 NONT-D SiliconFlow API')
    report = {'provider': 'siliconflow', 'base_url': 'https://api.siliconflow.cn/v1',
              'operation': 'GET /models', 'generation_requests': 0,
              'private_samples_sent': 0, 'exact_model': 'Qwen/Qwen3.5-9B'}
    request = urllib.request.Request('https://api.siliconflow.cn/v1/models',
                                     headers={'Authorization': 'Bearer ' + key})
    try:
        with urllib.request.build_opener(NoRedirect()).open(request, timeout=30) as response:
            raw = response.read(8 * 1024 * 1024 + 1)
            if len(raw) > 8 * 1024 * 1024:
                raise APIError('response_too_large')
            body = json.loads(raw)
        ids = sorted({r['id'] for r in body['data'] if isinstance(r.get('id'), str)})
        if any(key in ident for ident in ids):
            raise APIError('unsafe_model_list_content')
        report.update(http_status=200, model_count=len(ids),
                      public_model_ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                      qwen35_models=[s for s in ids if 'Qwen3.5' in s],
                      exact_model_listed=report['exact_model'] in ids)
    except urllib.error.HTTPError as exc:
        report.update(http_status=exc.code, error='http_' + str(exc.code))
        exc.close()
    except Exception:
        report.update(http_status=None, error='model_list_transport_or_contract_error')
    put(output, report)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(discover(args.output)))

"""Build a controlled inference-only archive; evaluator gold stays local."""
import argparse
from pathlib import Path
import zipfile
from artifacts import ROOT, code_inventory, load_study, require_private_output, sha


def bundle(package, output):
    _, manifest = load_study(package)
    package, output = Path(package), Path(output)
    require_private_output(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(code_inventory()):
            archive.write(ROOT / name, 'code/' + name)
        for name in ('manifest.json', 'inference/study.json'):
            archive.write(package / name, 'package/' + name)
    return {'archive_sha256': sha(output), 'bytes': output.stat().st_size,
            'study_sha256': manifest['study_sha256'], 'evaluator_gold_included': False}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    import json
    print(json.dumps(bundle(a.package, a.output)))

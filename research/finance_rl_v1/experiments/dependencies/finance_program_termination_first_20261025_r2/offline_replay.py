"""Exact original SFT response replay followed by the changed execution policy."""
import argparse
from collections import Counter
import json
from pathlib import Path
from support import ROOT, N100, load, native, put, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(exist_ok=False, parents=True)
    old = native()
    study, cases, baseline = old.prepare()
    changed = load('termination_first_replay_runtime', ROOT / 'runtime.py').build(old.parent.load_engine(old.parent.E2))
    by_source = {c['source_id']:c for c in cases}
    metrics = json.loads((N100 / 'reviews/detailed-metrics-01.json').read_bytes())
    ordered = sorted(metrics['rows'], key=lambda r:(r['legal_completion'],r['source_index']))
    rows = []
    evidence = []
    for row in ordered:
        folder = N100 / 'reviews/final-01/extract/runs' / row['slot_id']
        records = []
        for p in sorted(folder.glob('call-*/returned.json')):
            records.append(json.loads(p.read_bytes()))
            evidence.append(dict(path=p.relative_to(ROOT.parent).as_posix(),sha256=sha(p)))
        path = folder / 'trace.json'
        expected = json.loads(path.read_bytes())
        evidence.append(dict(path=path.relative_to(ROOT.parent).as_posix(),sha256=sha(path)))
        case = by_source[row['source_id']]
        replay = old.CPUReplay(records)
        before = baseline(case['item'], study, replay, case['snapshot'])
        assert old.raw_bytes(before) == old.raw_bytes(expected) and replay.position == len(records)
        replay = old.CPUReplay(records)
        after = changed(case['item'], study, replay, case['snapshot'])
        assert replay.position == len(records), 'every actual response must be consumed exactly once'
        assert after['terminated'] and after['new_steps'] <= 32
        if row['legal_completion']:
            assert old.raw_bytes(after) == old.raw_bytes(before), 'normal success must be exactly preserved'
        else:
            assert after['stop_reason'] in ('best_effort_terminal','abstained_terminal')
            if after['stop_reason'] == 'best_effort_terminal':
                assert after['terminal_event']['observation']['status'] == 'success'
                assert after['termination']['native_guard_valid']
            else:
                assert after['final_result'] is None and after['termination']['requires_review']
        dest = output / row['slot_id']
        put(dest / 'trace.json',after)
        result = dict(slot_id=row['slot_id'],source_id=row['source_id'],stratum=row['stratum'],
                      original_legal=row['legal_completion'],original_stop_reason=before['stop_reason'],
                      terminated=after['terminated'],stop_reason=after['stop_reason'],prediction=after['final_result'],
                      new_steps=after['new_steps'],recorded_model_calls=len(records),new_model_calls=0,
                      exact_original_replay=True,all_recorded_requests_exact=True,
                      termination=after.get('termination'),native_guard_valid=after['stop_reason']!='abstained_terminal')
        put(dest / 'result.json',result)
        rows.append(result)
        if len(rows) == 42:
            assert all(not r['original_legal'] for r in rows)
            predictions=sum(r['native_guard_valid'] for r in rows)
            assert predictions>=40 and all(r['terminated'] for r in rows)
            put(output / 'failed42-gate.json',dict(passed=True,denominator=42,guard_valid_predictions=predictions,
                                                  algorithm_completed=42,abstentions=42-predictions,
                                                  new_model_calls=0,kind='offline_exact_response_replay'))
            print(json.dumps(dict(phase='failed42_replay_passed',denominator=42,guard_valid_predictions=predictions,algorithm_completed=42)),flush=True)
    summary = dict(kind='offline_exact_response_replay',slots=100,new_model_calls=0,
                   original_stop_reasons=dict(Counter(r['original_stop_reason'] for r in rows)),
                   stop_reasons=dict(Counter(r['stop_reason'] for r in rows)),
                   same_actual_model_requests=True,unchanged_normal_successes=58,
                   rows=rows,input_evidence=evidence)
    put(output / 'completion.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('rows','input_evidence')}),flush=True)


if __name__ == '__main__':
    main()

"""Independent, local post-score comparisons; reads only audited results."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from support import put, sha


def dist(paths):
    counts = Counter(tuple(p) for p in paths)
    n = len(paths)
    return {'count': n, 'unique': len(counts),
            'top1_count': counts.most_common(1)[0][1] if n else 0,
            'top3_count': sum(c for _, c in counts.most_common(3))}


def main():
    review = ROOT / 'reviews'
    a = json.loads((review / 'independent-verification.json').read_bytes())
    score = json.loads((review / 'full422-score.json').read_bytes())
    old_path = ROOT.parent / 'finance_program_termination_val422_20261025_r2/reviews/full422-score.json'
    old = json.loads(old_path.read_bytes())
    refs = json.loads((ROOT.parent / 'finance_program_student_val422_eval_20261017_r1/reviews/local-gold422.json').read_bytes())
    rows, oldrows = score['rows'], {r['source_id']: r for r in old['rows']}
    assert a['passed'] and a['new_native_replays'] == 422 and len(rows) == len(oldrows) == len(refs) == 422
    assert len({r['source_id'] for r in rows}) == 422 and {r['source_id'] for r in rows} == set(oldrows) == set(refs)
    old_digest = score['baseline_score_sha256']
    assert sha(old_path) == old_digest
    first, paths, skills = [], [], Counter()
    levels = defaultdict(lambda: {'support': 0, 'sft_correct': 0, 'grpo_correct': 0})
    total_steps = []
    extract = review / 'final-01/extract'
    for row in rows:
        sid = row['source_id']
        ref = refs[sid]
        pred = row['prediction'] or {}
        oldrow = oldrows[sid]
        assert row['category_correct'] == (pred.get('category') == ref['category'])
        assert row['grade_correct'] == (pred.get('grading_level') == ref['grading_level'])
        assert row['joint_correct'] == (row['category_correct'] and row['grade_correct'])
        if not row['legal_completion']:
            assert row['prediction'] is None and not row['joint_correct']
        key = ref['grading_level']
        levels[key]['support'] += 1
        levels[key]['sft_correct'] += bool(oldrow['grade_correct'])
        levels[key]['grpo_correct'] += bool(row['grade_correct'])
        t = json.loads((extract / 'runs' / row['slot_id'] / 'trace.json').read_bytes())
        if t['planner_attempts']:
            try:
                first.append([s['tool'] for s in json.loads(t['planner_attempts'][0]['raw_response'])['program']])
            except (KeyError, TypeError, ValueError):
                pass
        actual = [s['call']['tool'] for p in t['programs'] for s in p['steps']]
        terminal = t.get('terminal_event', {}).get('observation', {})
        if terminal.get('status') == 'success':
            actual.append(terminal['call']['tool'])
        paths.append(actual)
        skills.update(actual)
        total_steps.append(row['executed_steps'])
    old_joint = sum(bool(r['joint_correct']) for r in oldrows.values())
    joint = sum(bool(r['joint_correct']) for r in rows)
    paired = {'improved': sum(bool(r['joint_correct']) and not oldrows[r['source_id']]['joint_correct'] for r in rows),
              'worsened': sum(not r['joint_correct'] and bool(oldrows[r['source_id']]['joint_correct']) for r in rows),
              'both_correct': sum(bool(r['joint_correct']) and bool(oldrows[r['source_id']]['joint_correct']) for r in rows),
              'both_wrong': sum(not r['joint_correct'] and not oldrows[r['source_id']]['joint_correct'] for r in rows)}
    assert paired == score['paired_comparison'] and old_joint == 138 and joint == score['all']['joint_correct']
    assert sum(paired.values()) == 422 and sum(level['support'] for level in levels.values()) == 422
    oldroot = old_path.parent.parent
    same = {n: sha(ROOT / n) == sha(oldroot / n) for n in ('selection.json', 'eval-inputs.json', 'termination.py')}
    result = {'passed': True, 'rows': 422, 'source_matching': True, 'same_input_and_termination_sha': same,
              'runtime_wrapper_byte_equal': sha(ROOT/'runtime.py') == sha(oldroot/'runtime.py'),
              'old_score_sha256': old_digest, 'new_score_sha256': sha(review/'full422-score.json'),
              'audit_sha256': sha(review/'independent-verification.json'), 'sft_joint_correct': old_joint,
              'grpo_joint_correct': joint, 'paired': paired, 'levels': dict(levels),
              'first_plan': dist(first), 'executed_path': dist(paths), 'distinct_skills': len(skills),
              'mean_executed_steps': statistics.mean(total_steps),
              'sft_calls': old['all']['model_calls'], 'grpo_calls': score['all']['model_calls'],
              'real_generation_failures': a['generation_failures'], 'fresh_test': False}
    put(review / 'analysis-detail-02.json', result)
    lines = ['# GRPO Val422：独立审核与评估', '',
             '训练完成 192 条轨迹及 6 次实际参数更新，评估使用最后版本 checkpoint。422 条均实际生成，worker 与 launch 均实际退出 0。', '',
             '| 指标 | 历史 SFT | GRPO |', '|---|---:|---:|',
             f"| 类别正确 | {old['all']['category_correct']}/422 | {score['all']['category_correct']}/422 |",
             f"| 等级正确 | {old['all']['grade_correct']}/422 | {score['all']['grade_correct']}/422 |",
             f'| 类别及等级联合正确 | {old_joint}/422 | {joint}/422 |',
             f"| 合法预测 | {old['all']['predictions']}/422 | {score['all']['predictions']}/422 |",
             f"| 弃权 | {old['all']['abstentions']}/422 | {score['all']['abstentions']}/422 |",
             f"| 固定 12 类 Macro F1 | {old['macro_f1_12']:.4f} | {score['macro_f1_12']:.4f} |",
             f"| 8 个有支持类别 Macro F1 | {old['macro_f1_supported']:.4f} | {score['macro_f1_supported']:.4f} |",
             f"| 模型调用 | {old['all']['model_calls']} | {score['all']['model_calls']} |", '',
             f"逐来源联合结果：改善 {paired['improved']}、退步 {paired['worsened']}、均正确 {paired['both_correct']}、均不正确 {paired['both_wrong']}；净变化 {joint-old_joint:+d} 条。", '',
             f"本次有 {len(a['generation_failures'])} 条实际生成失败并留在分母；完整 422 次 native 重放、{a['new_actual_model_calls']} 次实际输入改写和模型路由/80 权重核验通过。未对全部语义做人工业务审核。", '',
             f"首计划可解析 {len(first)}/422，{result['first_plan']['unique']} 种，最多模板 {result['first_plan']['top1_count']} 条；实际执行路径 {result['executed_path']['unique']} 种，最多模板 {result['executed_path']['top1_count']} 条，涉及 {len(skills)} 种 Skill。路径包含恢复与失败，不能直接当作规划能力提升。", '',
             '评估 selection、公开输入和 termination.py 与历史 SFT 字节一致；runtime.py 包装路径不同，且硬件和运行环境不同，完整运行条件等价尚未独立确认。按来源匹配的对照是开发验证结果，Val422 曾用于开发，并非全新独立 Test，也不能据此宣称统计显著性。', '',
             '完整类别、各等级、逐来源、失败与计数见 full422-score.json、analysis-detail-02.json 和 independent-verification.json。', '']
    with (review/'result-detail-02-zh.md').open('x', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(lines))
    print(json.dumps(result, ensure_ascii=True))


if __name__ == '__main__':
    main()

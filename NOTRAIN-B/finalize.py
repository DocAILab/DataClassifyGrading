"""Independently audit all receipts and write the completed experiment report."""
from collections import Counter
import json
import os
from pathlib import Path
from run import ROOT, read, put, category_path, digest


def main():
    manifest = read(ROOT / 'manifest.json')
    assert manifest['runner_sha256'] == digest(ROOT / 'run.py')
    audits = {}
    summary = read(ROOT / 'summary.json')
    for domain, expected in [('VEH', 397), ('FIN', 422)]:
        folder = ROOT / domain
        rows = read(folder / 'inputs/VAL.json')
        registry = {e['choice_id']: e for e in read(folder / 'registry.json')}
        metrics = read(folder / 'metrics.json')
        assert len(list((folder / 'sources').glob('*/result.json'))) == expected
        counts = Counter()
        for index, source in enumerate(rows, 1):
            path = folder / 'sources' / f'{index:04d}'
            result = read(path / 'result.json')
            assert result['source_id'] == source['id']
            counts[result['status']] += 1
            if result['status'] == 'completed':
                assert len(result['candidates']) == 5
                assert len(set(result['candidates'])) == 5
                for stage in ('stage1', 'stage2'):
                    request = read(path / stage / 'request.json')
                    assert request['model'] == manifest['model']
                    body = json.loads(request['messages'][1]['content'])
                    assert body['metadata'] == {'domain': source['domain'], 'field_name': source['metadata']['field_name']}
                    attempts = [read(p) for p in sorted((path / stage).glob('attempt-*.json'))]
                    success = [a for a in attempts if a['status'] == 'success']
                    assert len(success) == 1
                    if stage == 'stage1':
                        assert len(body['standard_catalog']) == len(registry)
                    else:
                        assert [e['id'] for e in body['candidates']] == result['candidates']
                        response = json.loads(success[0]['response']['choices'][0]['message']['content'])
                        assert registry[response['answer']]['path'] == result['predicted_path']
                        assert response['level'] == result['predicted_level']
            category_correct = result.get('predicted_path') == category_path(source['classification'])
            level_correct = result.get('predicted_level') == source['grading']['sensitivity_level']
            counts['category_correct'] += category_correct
            counts['level_correct'] += level_correct
            counts['joint_correct'] += category_correct and level_correct
        for key in ('category_correct', 'level_correct', 'joint_correct'):
            assert counts[key] == metrics[key]
        assert metrics == summary['metrics'][domain]
        assert not metrics['statuses'].get('pending', 0)
        audits[domain] = {'sources': expected, 'counts': dict(counts), 'metrics_sha256': digest(folder / 'metrics.json')}
    secret = os.environ.get('SILICONFLOW_KEY', '')
    if secret:
        needle = secret.encode()
        for file in ROOT.rglob('*'):
            if file.is_file() and file.suffix in ('.json', '.md', '.log', '.py'):
                assert needle not in file.read_bytes(), 'credential found in artifact'
    put(ROOT / 'audit.json', {'status': 'PASS', 'datasets': audits,
                             'checks': ['source identity and complete denominator', 'raw responses match predictions',
                                        'fixed two-stage candidate lineage', 'metadata allowlist',
                                        'independent exact-match recomputation', 'summary consistency', 'credential scan']})
    lines = ['# NOTRAIN-B 验证集实验结果', '',
             '已完成 VEH Val397 与 FIN Val422，共 819 条。Qwen/Qwen3.5-9B（SiliconFlow），无训练，固定 Top-5 检索 → Top-1 分类分级。', '',
             '| 指标 | VEH | FIN |', '|---|---:|---:|']
    a, b = summary['metrics']['VEH'], summary['metrics']['FIN']
    for name, key in [('分类准确率', 'exact_match_category'), ('级别准确率', 'level_accuracy'), ('联合准确率', 'exact_match_joint'),
                      ('Top-5 分类召回率', 'stage1_recall_at5'), ('终局解析率', 'terminal_parse_rate')]:
        lines.append(f'| {name} | {a[key]:.2%} | {b[key]:.2%} |')
    for name, head in [('分类', 'category'), ('级别', 'level'), ('联合', 'joint')]:
        for label, key in [('Macro-F1', 'macro_f1'), ('Weighted-F1', 'weighted_f1'), ('Macro-Precision', 'macro_precision'), ('Macro-Recall', 'macro_recall')]:
            lines.append(f'| {name} {label} | {a[head][key]:.2%} | {b[head][key]:.2%} |')
    for label, key in [('样本数', 'denominator'), ('合法终局数', 'terminal_valid_count'), ('HTTP 请求数（含重试）', 'http_attempts'),
                       ('每样本平均输入 tokens', 'average_input_tokens_per_sample'), ('每样本平均输出 tokens', 'average_output_tokens_per_sample')]:
        lines.append(f'| {label} | {a[key]:,.2f} | {b[key]:,.2f} |')
    lines.extend(['', '## 实验口径', '',
                  '- 数据固定版本：`' + manifest['revision'] + '`；只评估 VAL，不使用 TEST。',
                  '- 初始输入只有 domain、field_name；不额外调用元数据技能取得 field_description。',
                  '- 候选按标准条目计数：VEH 59 条目 / 44 叶子类，FIN 12 条目 / 12 叶子类。',
                  '- 第一步完整目录仅呈现条目编号与类别路径；第二步提供实际检索的五个条目的标准描述、参考标准和对应等级。',
                  '- 宏指标按验证集出现的 GT 标签计算；失败及非法输出保留在完整分母。分类比较完整非空类别路径。',
                  '- 两域各一条预检样本直接纳入正式结果，没有重新生成；不根据验证集结果修改提示词。',
                  '- 托管权重 revision 与精度不可核实；不能声称与本地 FP16 权重严格等价。', '',
                  '## 产物与核验', '',
                  '`manifest.json` 保存方法、配置、输入与源码哈希；`VEH/metrics.json`、`FIN/metrics.json` 为详细指标；',
                  '`sources/` 保存逐样本请求、原始响应、用量和预测；`inputs/` 保存评估输入与 gold（未发送给模型）。',
                  '`audit.json` 记录独立核验通过，`summary.json` 保存两域汇总，`run.log` 保存运行进度。',
                  '密钥从用户环境变量读取，未写入产物。数据及结果已通过本目录 `.gitignore` 排除，供本地查阅。', ''])
    (ROOT / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'audit': 'PASS', 'results': {d: {k: summary['metrics'][d][k] for k in ('statuses', 'exact_match_category', 'level_accuracy', 'exact_match_joint', 'http_attempts', 'usage')} for d in ('VEH', 'FIN')}}, ensure_ascii=False))


if __name__ == '__main__':
    main()

"""Local posthoc only: fixed denominator, unchanged native terminal semantics."""

def evaluate(rows, source_ids):
    if len(source_ids) != 20 or len(set(source_ids)) != 20:
        raise ValueError('fixed20 unique source identities required')
    if len(rows) != 20 or [r['source_id'] for r in rows] != source_ids:
        raise ValueError('all twenty outcomes in original source order required')
    count = sum(r.get('legal_completion') is True and r.get('native_replay_exact') is True
                and r.get('stop_reason') == 'validated_terminal' for r in rows)
    return dict(denominator=20, normal_completions=count, normal_termination_rate=count/20,
                target_rate=0.75, minimum_normal_completions=15, target_reached=count>=15,
                correctness_is_separate=True, semantic_quality_verified=False)

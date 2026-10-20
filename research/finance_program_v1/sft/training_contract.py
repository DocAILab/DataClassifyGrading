"""Pure gates shared by the full-length trial, training, and monitoring."""
# 试跑证明须与数据、最长来源和资源配置一致，防止用较短样本或其他运行的结果放行正式训练。
import math


def expected_steps(rows, accumulation):
    if rows < 1 or accumulation < 1:
        raise ValueError('positive source count and accumulation required')
    return math.ceil(rows / accumulation)


def validate_trial(proof, config):
    if (proof.get('data_sha256') != config['data_sha256']
            or proof.get('tokens') != config['max_length']
            or proof.get('source_id') != config['longest_source_id']
            or proof.get('gpu_uuids') != config['gpu_uuids']
            or proof.get('target_modules') != 40
            or any(proof.get(k) is not True for k in (
                'base_and_vision_frozen', 'lora_updated', 'gradients_finite', 'loss_finite'))):
        raise ValueError('actual longest-source finite backward/update proof required')


def validate_gpu_snapshot(expected, observed, free_mb, busy_uuids):
    if set(expected) != {'0', '1', '2'} or observed != expected:
        raise ValueError('three exact physical GPU identities required')
    if any(free_mb.get(k, 0) < 30000 for k in expected):
        raise ValueError('less than 30000 MiB free on a requested GPU')
    if set(expected.values()).intersection(busy_uuids):
        raise ValueError('requested GPU has another compute process; do not interfere')

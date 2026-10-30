"""Offline preparation contracts; no model, network, GPU or original-file writes."""
import hashlib
import json


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def message_sha(value):
    return sha(json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode())


def validate_program(raw):
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != {'program'}:
        raise ValueError('original Program object required')
    steps = value['program']
    if not isinstance(steps, list) or not 1 <= len(steps) <= 16:
        raise ValueError('Program must have 1..16 steps; NOT six-step limited')
    for step in steps:
        if (not isinstance(step, dict) or not isinstance(step.get('tool'), str)
                or not step['tool'] or not isinstance(step.get('arguments'), dict)):
            raise ValueError('Program step requires tool and arguments')
    return steps


def load_rows(raw, index):
    identities = [r for r in index['rows'] if r.get('exported') is True]
    by_id = {r['id']: r for r in identities}
    if len(by_id) != len(identities):
        raise ValueError('duplicate exported index identity')
    result = []
    for line in raw.splitlines():
        row = json.loads(line)
        if set(row) != {'id', 'messages', 'completion'} or row['id'] not in by_id:
            raise ValueError('unexpected dataset row/identity')
        identity = by_id[row['id']]
        messages = row['messages']
        if (not isinstance(messages, list) or [m.get('role') for m in messages] != ['system', 'user']
                or any(set(m) != {'role', 'content'} or not isinstance(m['content'], str) for m in messages)):
            raise ValueError('exact original system/user messages required')
        if not isinstance(row['completion'], str) or not row['completion']:
            raise ValueError('nonempty original completion required')
        if (message_sha(messages) != identity['messages_sha256']
                or sha(row['completion'].encode()) != identity['completion_utf8_sha256']):
            raise ValueError('original messages/target byte identity changed')
        if identity['decision'] not in ('accept', 'accept_with_notes'):
            raise ValueError('unapproved row cannot enter preparation')
        validate_program(row['completion'])
        result.append(dict(row, source_id=identity['source_id'], group=identity['group'],
                           supervision_state=identity['supervision_state']))
    if (len(result) != len(identities) or len({r['id'] for r in result}) != len(result)
            or {r['id'] for r in result} != set(by_id)):
        raise ValueError('all exported rows exactly once required')
    return result


def partition(rows, policy, eval_rows):
    if policy not in ('exclude_eval_groups', 'diagnostic_only'):
        raise ValueError('explicit evaluation overlap policy required')
    sources = {r['source_id'] for r in eval_rows}
    groups = {r['group_sha256'] for r in eval_rows}
    overlap = [r for r in rows if r['source_id'] in sources or r['group'] in groups]
    blocked = {r['id'] for r in overlap}
    kept = list(rows) if policy == 'diagnostic_only' else [r for r in rows if r['id'] not in blocked]
    if not kept:
        raise ValueError('no training candidates after group isolation')
    return kept, overlap


def plan_batches(ids, accumulation, max_updates):
    if (type(accumulation) is not int or accumulation < 1 or type(max_updates) is not int
            or max_updates < 1 or not ids or len(set(ids)) != len(ids)):
        raise ValueError('nonempty unique IDs, positive accumulation/update budget required')
    batches = [list(ids[i:i + accumulation]) for i in range(0, len(ids), accumulation)]
    if len(batches) > max_updates:
        raise ValueError('complete epoch exceeds explicit update budget; dropping rows forbidden')
    return batches


def loss_divisor(batch):
    if not batch:
        raise ValueError('cannot normalize empty accumulation batch')
    return len(batch)


def encode_completion(tokenizer, row, max_length):
    if type(max_length) is not int or max_length < 2:
        raise ValueError('explicit positive total context length required')
    eos, eos_id = tokenizer.eos_token, tokenizer.eos_token_id
    if not isinstance(eos, str) or not eos or type(eos_id) is not int:
        raise ValueError('original EOS string/ID required')
    prompt = tokenizer.apply_chat_template(row['messages'], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False)
    prefix = tokenizer(prompt, add_special_tokens=False)['input_ids']
    full = tokenizer(prompt + row['completion'] + eos, add_special_tokens=False)['input_ids']
    if (not prefix or any(type(i) is not int for i in full + prefix)
            or full[:len(prefix)] != prefix):
        raise ValueError('prompt must be exact token prefix')
    if len(full) > max_length:
        raise ValueError('context overflow; truncation forbidden')
    target = full[len(prefix):]
    if len(target) < 2 or target[-1] != eos_id:
        raise ValueError('full completion and EOS required')
    decoded = tokenizer.decode(target, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    if decoded != row['completion'] + eos:
        raise ValueError('decoded completion differs from original raw target plus EOS')
    return dict(input_ids=full, attention_mask=[1] * len(full),
                labels=[-100] * len(prefix) + target,
                prompt_length=len(prefix), target_length=len(target))

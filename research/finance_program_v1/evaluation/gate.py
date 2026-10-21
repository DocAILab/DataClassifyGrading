"""Pure fail-closed evaluation authorization and routing gates."""
# 评估放行同时绑定训练完成状态、独立审计及适配器身份；路由依据可信 system 消息识别 Planner。
def training_ready(state, signature):
    complete, proof = state.get('completion') or {}, state.get('cpu_proof') or {}
    return bool(state.get('manifest_sha256') == signature and state.get('writer_live') is False
        and all(state.get('exits', {}).get(k) == 0 for k in ('trial', 'formal', 'pipeline', 'archive'))
        and complete.get('manifest_sha256') == proof.get('manifest_sha256') == signature
        and complete.get('global_step') == 13 and complete.get('sources') == 200 and complete.get('epoch') == 1.
        and proof.get('independent_cpu_audit_passed') is True and state.get('adapter_sha256')
        and state['adapter_sha256'] == complete.get('adapter_sha256') == proof.get('adapter_sha256'))


def enabled_for(arm, messages, planner_system):
    if arm not in ('base', 'sft'):
        raise ValueError('unknown paired condition')
    return bool(arm == 'sft' and messages and messages[0].get('role') == 'system'
        and messages[0].get('content') == planner_system)


def validate_ids(ids, smoke, train):
    if (len(ids) != 80 or len(set(ids)) != 80 or len(smoke) != 2 or len(set(smoke)) != 2
            or set(ids) & set(smoke) or (set(ids) | set(smoke)) & set(train)):
        raise ValueError('fixed80 and smoke must be unique and disjoint from training')


def eos_options(model_eos, tokenizer_eos):
    values = list(model_eos) if isinstance(model_eos, (list, tuple)) else [model_eos]
    values.append(tokenizer_eos)
    if any(type(x) is not int or x < 0 for x in values):
        raise ValueError('actual model and tokenizer EOS required')
    return sorted(set(values))


def validate_adapter_files(folder, anchor):
    from pathlib import Path
    from io_utils import sha
    folder=Path(folder)
    if (sha(folder/'adapter_model.safetensors')!=anchor['adapter_sha256']
            or sha(folder/'adapter_config.json')!=anchor['adapter_config_sha256']):
        raise ValueError('loaded adapter/config bytes differ from accepted training anchor')


def validate_lora_inventory(config,inventory,original_targets,linear_shapes):
    """Compare expanded module identity and both tensor shapes, not serialized suffix count."""
    import re
    if (config.get('r')!=8 or config.get('lora_alpha')!=16 or config.get('lora_dropout')!=.05
            or config.get('peft_type')!='LORA' or config.get('task_type')!='CAUSAL_LM'
            or config.get('bias')!='none' or config.get('modules_to_save')
            or config.get('use_dora') or config.get('use_rslora') or config.get('rank_pattern')
            or config.get('alpha_pattern')):
        raise ValueError('unexpected LoRA recipe')
    suffixes=config.get('target_modules')
    if not isinstance(suffixes,list) or not suffixes or any(not isinstance(s,str) or not s for s in suffixes):
        raise ValueError('nonempty serialized target suffix list required')
    wanted=set(original_targets)
    if (len(wanted)!=len(original_targets) or len(wanted)!=40
            or any(not n.startswith('model.language_model.layers.') for n in wanted)):
        raise ValueError('original forty language targets required')
    expanded={n for n in linear_shapes if any(n==s or n.endswith('.'+s) for s in suffixes)}
    if expanded!=wanted:raise ValueError('serialized targets expand beyond/differ from original forty layers')
    found={};names=set()
    for item in inventory:
        name=item['name'];match=re.fullmatch(r'base_model\.model\.(.+)\.lora_([AB])\.weight',name)
        if name in names or not match:raise ValueError('unexpected or duplicate adapter tensor')
        names.add(name);target,side=match.groups()
        if target not in wanted:raise ValueError('adapter tensor outside original language targets')
        out_features,in_features=linear_shapes[target]
        shape=[8,in_features] if side=='A' else [out_features,8]
        if item['shape']!=shape:raise ValueError('adapter tensor shape differs from original linear layer')
        found.setdefault(target,set()).add(side)
    if len(inventory)!=80 or set(found)!=wanted or any(sides!={'A','B'} for sides in found.values()):
        raise ValueError('exactly forty complete A/B pairs required')
    return {'serialized_target_suffixes':suffixes,'resolved_targets':40,'tensor_count':80,
        'original_targets_and_shapes_exact':True}


def cpu_proof_identity(proof,actual_config_sha,completion_config_sha,verifier_sha,validator_manifest_sha):
    return bool(proof.get('cpu_validator_version')==2
        and proof.get('adapter_config_sha256')==actual_config_sha==completion_config_sha
        and proof.get('verifier_sha256')==verifier_sha
        and proof.get('validator_manifest_sha256')==validator_manifest_sha)

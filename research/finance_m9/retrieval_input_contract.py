from copy import deepcopy
import json

def disclose_descriptions(messages,registry):
    out=deepcopy(messages)
    if not out or not out[0].get('content'):return out
    first=out[0]['content'].splitlines()[0]
    if not first.startswith('Execute only the selected Skill SEARCH_STANDARD'):return out
    if len(out)!=2 or [x['role'] for x in out]!=['system','user']:raise ValueError('unlabelled retrieval request required')
    by={}
    for item in registry:
        option=item.get('option_id')
        if not isinstance(option,str) or option in by:raise ValueError('unique registry IDs required')
        if not all(isinstance(item.get(k),str) and item[k].strip() for k in ['label','description']):raise ValueError('frozen nonempty standard label/description required')
        by[option]=item
    try:user=json.loads(out[-1]['content'])
    except (ValueError,TypeError) as exc:raise ValueError('invalid retrieval JSON') from exc
    if not isinstance(user,dict) or not isinstance(user.get('category_options'),list) or not user['category_options']:raise ValueError('typed supplied category options required')
    seen=set()
    for item in user['category_options']:
        option=item.get('option_id')
        if not isinstance(option,str) or option in seen or option not in by:raise ValueError('supplied option must be unique and in frozen registry')
        seen.add(option);standard=by[option]
        if item.get('label')!=standard['label']:raise ValueError('frozen label mismatch')
        if 'description' in item and item['description']!=standard['description']:raise ValueError('conflicting existing description')
        item['description']=standard['description']
    out[-1]['content']=json.dumps(user,ensure_ascii=False,sort_keys=True,separators=(',',':'))
    return out

"""Transport/identity/accounting health is separate from original JSON syntax."""
import json
def classify(outcome,model):
 r=outcome['result'];u=r.get('usage');known=isinstance(u,dict) and all(type(u.get(k)) is int and u[k]>=0 for k in ('prompt_tokens','completion_tokens','total_tokens')) and u['prompt_tokens']+u['completion_tokens']==u['total_tokens']
 wire=bool(r.get('http_status')==200 and r.get('transport_complete') is True and r.get('stream_done') is True and r.get('models')==[model] and r.get('finish_reason') in ('stop','length') and r.get('child_exitcode')==0 and r.get('child_stopped') is True and r.get('timed_out') is not True and r.get('fatal_capture') is not True and known)
 try:valid=isinstance(json.loads(outcome['provider_raw_content']),dict)
 except (ValueError,TypeError):valid=False
 return dict(wire_complete=wire,raw_JSON_object=valid,original_provider_complete=r.get('complete'),known_usage=known,response_not_repaired=True)

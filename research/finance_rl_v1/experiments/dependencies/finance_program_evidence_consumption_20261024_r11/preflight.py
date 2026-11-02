import importlib.util,json
from common import ROOT,PREVIOUS
spec=importlib.util.spec_from_file_location('r11_parent_preflight',PREVIOUS/'preflight.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
def run():
 proof=old.run()
 import interface
 tok=old.tokenizer();rows=[]
 path=PREVIOUS/'reviews/targeted-01/extract/runs'
 if not path.exists():path=PREVIOUS/'runs'
 for folder in sorted(path.iterdir()):
  if not folder.is_dir():continue
  session=interface.RecoverySession()
  for p in sorted(folder.glob('call-*/returned.json')):
   messages=[dict(role=m['role'],content=m['content']) for m in json.loads(p.read_bytes())['messages']]
   actual=session.rewrite(messages)
   if not messages[0]['content'].startswith('Execute only the selected Skill SELECT_CATEGORY_TOPK.'):continue
   body=json.loads(actual[-1]['content']);index=body['selector_source_index']
   text=tok.apply_chat_template(actual,tokenize=False,add_generation_prompt=True,enable_thinking=False)
   n=len(tok(text,add_special_tokens=False)['input_ids']);assert n+512<=49152
   assert not any(k in json.dumps(index) for k in ('option_id','preferred_category'))
   rows.append(dict(slot=folder.name,call=p.parent.name,tokens=n,index_facts=len(index['target_record'])+len(index['same_table_context'])))
 assert len(rows)==16
 proof['actual_selector_tokens']=rows;proof['inherited_deadline_unix']=1792837212.3172429
 return proof

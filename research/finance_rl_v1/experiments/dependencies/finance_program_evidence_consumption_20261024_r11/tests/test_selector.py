import importlib.util,json,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
PARENT=ROOT.parent/'finance_program_evidence_consumption_20261024_r10'
def load(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
previous=load(PARENT/'interface.py','selector_original_r10')
def active():return load(ROOT/'interface.py','selector_new_interface') if (ROOT/'interface.py').exists() else previous
def folder():
 path=PARENT/'reviews/targeted-01/extract/runs'
 return path if path.exists() else PARENT/'runs'
class SelectorTests(unittest.TestCase):
 def test_all16_select_requests_have_exact_current_source_values_no_hypotheses(self):
  count=0
  for f in sorted(folder().iterdir()):
   if not f.is_dir():continue
   new=active().RecoverySession();old=previous.RecoverySession()
   for p in sorted(f.glob('call-*/returned.json')):
    messages=[dict(role=m['role'],content=m['content']) for m in json.loads(p.read_bytes())['messages']]
    actual=new.rewrite(messages);wanted=old.rewrite(messages)
    if not messages[0]['content'].startswith('Execute only the selected Skill SELECT_CATEGORY_TOPK.'):continue
    body=json.loads(actual[-1]['content']);expected=json.loads(wanted[-1]['content'])
    self.assertTrue('selector_source_index' in body,'selector needs actual acquired-source presentation instead of old semantic query only')
    index=body.pop('selector_source_index');facts=expected['evidence_boundary']['facts']
    eligible={k:v for k,v in facts.items() if v.get('role')=='raw_fact' and v.get('decisive_eligible') is True}
    pairs=index['target_record']+index['same_table_context']
    self.assertEqual({k:v for k,v in pairs},{k:v['value'] for k,v in eligible.items()})
    self.assertEqual(len(pairs),len(eligible));self.assertFalse(index['sufficiency_adjudicated'])
    self.assertEqual(body,expected);count+=1
  self.assertEqual(count,16)
 def test_nonselector_requests_exact_r10(self):
  seen=0
  for f in sorted(folder().iterdir()):
   if not f.is_dir():continue
   new=active().RecoverySession();old=previous.RecoverySession()
   for p in sorted(f.glob('call-*/returned.json')):
    messages=[dict(role=m['role'],content=m['content']) for m in json.loads(p.read_bytes())['messages']]
    actual,wanted=new.rewrite(messages),old.rewrite(messages)
    if messages[0]['content'].startswith('Execute only the selected Skill SELECT_CATEGORY_TOPK.'):continue
    self.assertEqual(actual,wanted);seen+=1
  self.assertEqual(seen,67)
 def test_campaign_anchor_not_new_clock(self):
  import common
  self.assertEqual(common.CAMPAIGN,PARENT)
if __name__=='__main__':unittest.main()

"""Offline shared projector: distinguish actual failed step from unexecuted tail."""
import copy,importlib.util,json
from pathlib import Path
DECODE=Path(__file__).resolve().parent.parent/'finance_teacher_program_coldstart_20261024_r3/reviews/availability_index.py'
spec=importlib.util.spec_from_file_location('R8_pinned_public_lossless_decoder',DECODE);decoder=importlib.util.module_from_spec(spec);spec.loader.exec_module(decoder)
GUIDANCE=('\nrecovery_checkpoint_index_v1 copies only the preceding public program checkpoint. '
 'The unexecuted tail is NOT an executed observation. A canonical unavailable acquisition is not proof '
 'that later search or inference calls in that program ran. The index supplies no new evidence, '
 'automatic action, candidate preference or sufficiency decision. Keep all typed, availability, '
 'repeat, citation and terminal guards unchanged.')

def rewrite(messages,actor):
 if actor not in ('student','teacher'):raise ValueError('unknown shared actor')
 out=copy.deepcopy(messages)
 if not out or out[0]['role']!='system':raise ValueError('trusted first system message required')
 if out[0]['content'].startswith('Execute only the selected Skill '):return out
 body=json.loads(out[-1]['content'])
 if 'recovery_checkpoint_index_v1' in body:raise ValueError('projection already applied')
 if body.get('encoding')=='lossless-message-values-v1':root=body['payload'];pool=body['lossless_message_values']
 elif 'encoding' in body:raise ValueError('unknown public encoding')
 else:root=body;pool={}
 feedback=decoder._resolve(root.get('feedback'),pool);failed=None;tail=[];status=None
 if feedback is not None:
  if not isinstance(feedback,dict):raise ValueError('public program feedback not an object')
  status=feedback.get('status');tail=feedback.get('unexecuted_tail',[]);steps=feedback.get('steps',[]);index=feedback.get('failure_index')
  if not isinstance(tail,list) or not isinstance(steps,list):raise ValueError('invalid public step list')
  if index is not None:
   if type(index) is not int or not 0<=index<len(steps):raise ValueError('invalid public failure index')
   step=steps[index];call=step.get('call',{})
   if not isinstance(call,dict) or not isinstance(call.get('tool'),str):raise ValueError('public failed call absent')
   failed=dict(tool=call['tool'],arguments=copy.deepcopy(call.get('arguments',{})),status=step.get('status'),summary=step.get('summary'))
 body['recovery_checkpoint_index_v1']=dict(schema='observed-program-checkpoint-v1',previous_program_status=status,failed_step=failed,unexecuted_tail=copy.deepcopy(tail),tail_is_executed=False,evidence_added=False,automatic_actions=False,candidate_preference_added=False,sufficiency_decision_added=False)
 out[-1]['content']=json.dumps(body,ensure_ascii=False,separators=(',',':'));out[0]['content']+=GUIDANCE;return out

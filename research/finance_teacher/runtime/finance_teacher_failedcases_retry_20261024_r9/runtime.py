"""Pinned native guards, R4 actor, one prospectively shared checkpoint projector."""
import copy
from pathlib import Path
from campaign import load,audit_shared,message_sha
BASE=Path(__file__).resolve().parent
original=load('R9_original_R4_native',BASE.parent/'finance_teacher_program_coldstart_20261024_r4/runtime.py')
projector=load('R9_fixed_R8_checkpoint',BASE.parent/'finance_teacher_failedcase_probe_20261024_r8/checkpoint_index.py')
native_context,build=original.native_context,original.build
class CommonActor(original.CommonActor):
 def project(self,messages):
  base=super().project(messages);student=projector.rewrite(base['messages'],'student');teacher=projector.rewrite(base['messages'],'teacher');audit_shared(student,teacher)
  return dict(messages=teacher,input_sha256=message_sha(teacher),native_input_sha256=message_sha(messages),same_information_exact=True,base_R4_input_sha256=base['input_sha256'])

"""Same positive-scope policy for all actual F66 classifier/search entry points."""
import copy,importlib.util
from pathlib import Path
SCOPE=Path(__file__).resolve().parent.parent/'finance_teacher_same_information_20261024_r3/scope_contract.py'
spec=importlib.util.spec_from_file_location('fullteacher_pinned_shared_scope_R3',SCOPE)
scope=importlib.util.module_from_spec(spec);spec.loader.exec_module(scope)
SEARCH=frozenset(('SELECT_CATEGORY_TOPK','SEARCH_STANDARD','SEARCH_STANDARD_BY_FIELD','SEARCH_STANDARD_BY_CONTEXT','SEARCH_STANDARD_BY_TABLE','SEARCH_STANDARD_BY_DATABASE','RESEARCH_STANDARD'))
INFER=frozenset(('INFER_CATEGORY','INFER_CATEGORY_FROM_TOPK'))
def project_messages(messages,actor):
 if actor not in ('student','teacher'):raise ValueError('unknown shared actor')
 head=messages[0]['content'].splitlines()[0]
 prefix='Execute only the selected Skill '
 stage=head[len(prefix):-1] if head.startswith(prefix) and head.endswith('.') else None
 if stage in ('SELECT_CATEGORY_TOPK','INFER_CATEGORY_FROM_TOPK'):return scope.project_messages(messages,actor)
 if stage=='INFER_CATEGORY':
  # The pinned static projector validates its canonical stage. Dispatch internally
  # under that name, then retain the actual native Skill header in the request.
  temporary=copy.deepcopy(messages);_,tail=temporary[0]['content'].split('\n',1)
  temporary[0]['content']='Execute only the selected Skill INFER_CATEGORY_FROM_TOPK.\n'+tail
  out=scope.project_messages(temporary,actor);_,tail=out[0]['content'].split('\n',1)
  out[0]['content']=head+'\n'+tail;return out
 if stage in SEARCH:return scope.parent_projection(messages,actor)
 return copy.deepcopy(messages)

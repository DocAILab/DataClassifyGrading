"""R9-compatible public recovery projection and comparison output guidance."""
import importlib.util,json,re
from pathlib import Path
from feedback import project
path=Path(__file__).resolve().parent.parent/'finance_program_skill_recovery20_20261022_r9/interface.py'
spec=importlib.util.spec_from_file_location('preserved_R9_recovery_interface',path)
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)

RECOVERY=('Recovery feedback below is an index of observations and static contract options already in this request. '
 'Read unresolved_gaps and latest_failures before selecting a new plan. A classification rejection for insufficient '
 'evidence cannot be repaired by repeating classification, reinterpreting the same facts, or adding grading/finalization. '
 'Choose a relevant acquisition only if its actual required inputs and availability observations permit it; '
 'inspect missing_required_inputs and input_contracts possible_producers when a locator is absent. '
 'These are alternative options, not a sequence. Do not call every listed Skill. '
 'An empty missing_required_inputs list proves presence only, not typed validity, data availability or relevance. '
 'A producer may return no data. Previously obtained/unavailable information is not an unqueried resource. '
 'Only actually returned eligible raw facts can resolve an evidence gap; never assume success. '
 'If available acquisition cannot resolve the distinction, preserve insufficient rather than force a terminal. '
 'Native evidence/terminal guards and remaining budgets still apply; no automatic actions or evidence were added.')

class RecoverySession:
    def __init__(self):self.gaps={}
    def rewrite(self,messages):return rewrite(messages,session=self)

class RecoveryAdapter:
    """Instantiate once per rollout; its backend retains the real call/resource budget."""
    def __init__(self,backend):
        self.backend=backend
        self.session=RecoverySession()
        self.last_generation_metadata={}
        self.last_call_evidence={}
    def generate(self,messages,arm,stage,destination):
        try:
            return self.backend.generate(self.session.rewrite(messages),arm,stage,destination)
        finally:
            self.last_generation_metadata=self.backend.last_generation_metadata.copy()
            self.last_call_evidence=self.backend.last_call_evidence.copy()

def rewrite(messages,*,session=None):
    actual=previous.rewrite(messages)
    stage=re.match(r'Execute only the selected Skill ([A-Z_]+)\.',actual[0]['content'])
    if stage:
        if stage.group(1)=='COMPARE_STANDARD_CANDIDATES':
            actual[0]['content']+=('\nOutput contract reminder: reason must be ONE nonempty JSON string of at most 240 characters; '
                'aim for no more than 120 characters to leave generation room. Escape quotes. '
                'Return only decision, option_id, reason and close the JSON object. '
                'Do not add prose, markdown, multiple candidate explanations or extra keys. '
                'Use match only with actual supplied evidence supporting an actually supplied option; otherwise preserve no_match. '
                'The length requirement never authorizes dropping decisive uncertainty or guessing a category.')
        return actual
    original=json.loads(messages[-1]['content'])
    if session is not None and not original.get('feedback'):
        current=original.get('current',{});key=(current.get('input',{}).get('domain'),current.get('input',{}).get('field_name'))
        session.gaps.pop(key,None)
    if original.get('feedback') or original.get('recovery_context'):
        current=original.get('current',{});key=(current.get('input',{}).get('domain'),current.get('input',{}).get('field_name'))
        prior=session.gaps.get(key,[]) if session is not None else []
        view=project(original,prior_gaps=prior)
        if session is not None:
            unique={(g.get('skill'),g.get('missing_evidence')):g for g in view['unresolved_gaps']}
            session.gaps[key]=list(unique.values())
        body=json.loads(actual[-1]['content']);body['recovery_feedback_v1']=view
        actual[-1]['content']=json.dumps(body,ensure_ascii=False,separators=(',',':'))
        actual[0]['content']+='\n'+RECOVERY
        actual[0]['content']+= ('\npreparation_options lists alternative producers needed to make acquired context usable. '
            'A list of column names is not interpreted context. Check the public required_inputs/produces '
            'before using it; selection, expansion and interpretation have different typed outputs. '
            'Historical gaps were actually reported earlier in this case, not newly adjudicated now. '
            'A rejected plan adds no observations and does not erase a previous evidence gap. '
            'Resolve newly reported missing dependencies before retrying the same rejected suffix. '
            'Do not select the same already blocked retrieval or classification without a relevant changed input.')
    elif session is not None:
        # A fresh rollout cannot inherit another rollout of the same source.
        current=original.get('current',{});key=(current.get('input',{}).get('domain'),current.get('input',{}).get('field_name'))
        session.gaps.pop(key,None)
    return actual

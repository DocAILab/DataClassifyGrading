"""Two-card inference only:16GiB cap/card,2GiB physical headroom,12GiB placement."""
from pathlib import Path
path=Path(__file__).resolve().parent.parent/'finance_program_student_b80_eval_20261021_r2/resource_gate.py'
source=path.read_text(encoding='utf-8')
old='0 < max_sequence_tokens <= 32768'
if source.count(old)!=1:raise ValueError('unknown original resource context guard')
source=source.replace(old,'0 < max_sequence_tokens <= 49152')
namespace={'__file__':str(path),'__name__':'two_card_inference_envelope'}
exec(compile(source,str(path)+'[inference48k-two-only]','exec'),namespace)
namespace['PROFILES']={2:{'allocator':16*1024**3,'model_weights':12*1024**3}}
def candidates(rows,*,max_sequence_tokens):
 result=namespace['candidates'](rows,max_sequence_tokens=max_sequence_tokens)
 for r in result:r['scientific_updates_allowed']=0;r['admission_kind']='two_card_inference_provisional'
 return result
def recheck(rows,anchor):
 result=namespace['recheck'](rows,anchor)
 result.update(scientific_updates_allowed=0,admission_kind='two_card_inference_provisional')
 return result

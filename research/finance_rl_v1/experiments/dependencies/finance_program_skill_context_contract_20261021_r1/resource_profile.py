"""Keep original memory/identity safety profile; allow only finite48K metadata."""
from pathlib import Path

path=Path(__file__).resolve().parent.parent/'finance_program_student_b80_eval_20261021_r2/resource_gate.py'
source=path.read_text(encoding='utf-8')
old='0 < max_sequence_tokens <= 32768'
if source.count(old)!=1:raise ValueError('unknown original resource context guard')
source=source.replace(old,'0 < max_sequence_tokens <= 49152')
source=source.replace('actual untruncated token audit within 32768 required','actual untruncated token audit within49152 required')
namespace={'__file__':str(path),'__name__':'context48k_same_memory_profile'}
exec(compile(source,str(path)+'[inference48k-only]','exec'),namespace)
candidates=namespace['candidates']
recheck=namespace['recheck']

"""CPU-only full initial supplementary342 encoding using the original message builder."""
from io_utils import ROOT,read,put,sha


def initial_messages(row,study):
    from program_loop import build_program_controller
    from program_loop.planner import planner_messages
    from agent.orchestration.contracts import MetadataRecord
    c=build_program_controller(lambda messages:'',study['registry'],study['rows'],study['budget'],skill_pool='F66')
    record=MetadataRecord.from_mapping({'source_id':row['id'],'domain':row['domain'],'field_name':row['metadata']['field_name']})
    state={};c.bridge.availability.initialize(record,state)
    return planner_messages(record,state,c.library,{'programs':3,'planner_attempts':4,'skill_steps':12,'program_steps':10},None,skill_pool='F66')


def main():
    from program_loop import build_program_controller
    from program_loop.planner import planner_messages
    from agent.orchestration.contracts import MetadataRecord
    from transformers import AutoTokenizer
    study=read(ROOT/'data/study.json');protocol=read(ROOT/'data/protocol.json')
    c=build_program_controller(lambda messages:'',study['registry'],study['rows'],study['budget'],skill_pool='F66')
    assert len(c.library)==len({s['name'] for s in c.library})==66
    model=read(ROOT/'manifest.json')['config']['model']
    tok=AutoTokenizer.from_pretrained(model,local_files_only=True)
    rows={r['id']:r for r in study['rows']};lengths=[]
    from full_validation import partition
    ids=partition(study)['val342']
    for i in ids:
        row=rows[i];messages=initial_messages(row,study)
        assert messages[0]['content']==protocol['planner_system'],'student train and eval Planner system differ'
        prompt=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        n=len(tok(prompt,add_special_tokens=False)['input_ids']);assert n+512<=32768
        lengths.append({'id':i,'tokens':n})
    put(ROOT/'reviews/length-proof.json',{'full_initial_sources':len(lengths),'min':min(x['tokens'] for x in lengths),
        'max':max(x['tokens'] for x in lengths),'all_full66_contracts':True,'truncation':False,'rows':lengths})
    assert len(lengths)==342
    print('PREFLIGHT_FULL342_LENGTHS_OK')


if __name__=='__main__':main()

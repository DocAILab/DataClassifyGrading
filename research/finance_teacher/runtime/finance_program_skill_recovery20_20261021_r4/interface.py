"""Exact JSON subtree interning and one typed prompt, never output repair."""
from collections import Counter
import importlib.util
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('recovery20_previous_interface',ROOT.parent/'finance_program_skill_recovery20_20261021_r1/interface.py')
previous=importlib.util.module_from_spec(s);s.loader.exec_module(previous)

ENCODING_RULE=('Lossless message encoding v1: the user JSON has payload and lossless_message_values. '
 'Interpret payload as the original full planning input. A JSON object with exactly one key '
 '"$message_ref" means the exact value stored under that ID in lossless_message_values; '
 'resolve such references recursively, including within stored values. An object with exactly '
 '"$message_concat" contains an ordered list of string pieces/references; concatenate their resolved '
 'strings with no added separator to restore the exact original text. Restore values before reading any contract, '
 'An object with exactly "$message_json_string" and "style" represents an original JSON string: '
 'restore its nested value then serialize it as JSON with original key order, Unicode preserved, '
 'indent=2 for style pretty2 or no whitespace separators for style compact. Its parsed contents '
 'remain the originally recorded message, not a new observation. '
 'state, evidence, feedback or budget. No original facts or Skill contracts have been removed. '
 'These are message-only references, never executable Program arguments. Existing prompt_ref/shared_values '
 'references inside the restored input retain their original documented meaning. Output only the original Program JSON.')
SELECTOR_RULE=('Context selection output contract: selected_context_fields MUST be a nonempty JSON array of distinct field-name strings, '
 'not objects, not dictionaries of field names and reasons. Every string must exactly equal a field_name from the supplied table_fields. '
 'Put any explanation only in the root summary string. data has only selected_context_fields. '
 'Return exactly {"summary":"brief source-supported explanation","data":{"selected_context_fields":["FIELD_NAME"]}} '
 'with actual supplied field names, not the example placeholder. No extra keys, Markdown, invented fields or repaired metadata. '
 'Selection is not proof of business ownership or category; preserve all existing evidence boundaries.')

def dumps(value):
    return json.dumps(value,ensure_ascii=False,separators=(',',':'),allow_nan=False)

def json_string(value):
    if not isinstance(value,str) or len(value)<160 or value[:1] not in ('{','['):return None
    try:
        obj=json.loads(value)
        if json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)==value:return obj,'pretty2'
        if dumps(obj)==value:return obj,'compact'
    except (ValueError,TypeError):pass
    return None

def pack(value):
    counts=Counter()
    def count(v):
        if isinstance(v,dict) and set(v) in ({'$message_ref'},{'$message_concat'},{'$message_json_string','style'}):
            raise ValueError('reserved message-reference shape in original input')
        text=dumps(v)
        if len(text)>=160:counts[text]+=1
        if isinstance(v,dict):
            for x in v.values():count(x)
        elif isinstance(v,list):
            for x in v:count(x)
        elif isinstance(v,str):
            parsed=json_string(v)
            if parsed is not None:
                count(parsed[0]);return
            parts=re.split(r'(?<=\n)|(?<=\. )',v)
            if len(parts)>1:
                for line in parts:
                    if len(line)>=48:counts[dumps(line)]+=1
    count(value)
    values={};ids={}
    def encode(v):
        text=dumps(v);intern=len(text)>=160 and counts[text]>1
        if intern and text in ids:return {'$message_ref':ids[text]}
        if isinstance(v,dict):body={k:encode(x) for k,x in v.items()}
        elif isinstance(v,list):body=[encode(x) for x in v]
        elif isinstance(v,str):
            parsed=json_string(v)
            if parsed is not None:body={'$message_json_string':encode(parsed[0]),'style':parsed[1]}
            else:
                parts=re.split(r'(?<=\n)|(?<=\. )',v);pieces=[];changed=False
                for line in parts:
                    code=dumps(line)
                    if len(parts)>1 and len(line)>=48 and counts[code]>1:
                        if code not in ids:
                            key='r'+str(len(values));ids[code]=key;values[key]=line
                        pieces.append({'$message_ref':ids[code]});changed=True
                    else:pieces.append(line)
                body={'$message_concat':pieces} if changed else v
        else:body=v
        if intern:
            key='r'+str(len(values));ids[text]=key;values[key]=body
            return {'$message_ref':key}
        return body
    body=encode(value)
    return {'encoding':'lossless-message-values-v1','payload':body,'lossless_message_values':values}

def expand_packet(packet):
    values=packet['lossless_message_values']
    def expand(v,trail=()):
        if isinstance(v,dict):
            if set(v)=={'$message_ref'}:
                key=v['$message_ref']
                if not isinstance(key,str) or key in trail or key not in values:
                    raise ValueError('cyclic or unknown message reference')
                return expand(values[key],trail+(key,))
            if set(v)=={'$message_concat'}:
                pieces=[expand(x,trail) for x in v['$message_concat']]
                if not all(isinstance(x,str) for x in pieces):raise ValueError('non-string concatenation')
                return ''.join(pieces)
            if set(v)=={'$message_json_string','style'}:
                body=expand(v['$message_json_string'],trail)
                if v['style']=='pretty2':return json.dumps(body,ensure_ascii=False,indent=2,allow_nan=False)
                if v['style']=='compact':return dumps(body)
                raise ValueError('unknown original JSON string style')
            return {k:expand(x,trail) for k,x in v.items()}
        if isinstance(v,list):return [expand(x,trail) for x in v]
        return v
    return expand(packet['payload'])

def rewrite(messages):
    result=previous.rewrite(messages)
    match=re.match(r'Execute only the selected Skill ([A-Z_]+)\.',result[0]['content'])
    if match:
        if match.group(1)=='SELECT_CONTEXT_FIELDS':result[0]['content']+='\n'+SELECTOR_RULE
        return result
    payload=json.loads(result[-1]['content']);encoded=pack(payload)
    if dumps(expand_packet(encoded))!=dumps(payload):raise ValueError('lossless message roundtrip differs')
    result[0]['content']+='\n'+ENCODING_RULE
    result[-1]['content']=dumps(encoded)
    return result

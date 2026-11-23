"""Deterministic cache placement only; never edit token inputs or decoding budgets."""
LONG_CONTEXT_THRESHOLD=32768
MAX_CONTEXT_TOKENS=49152

def generation_options(input_tokens,max_new_tokens):
 if type(input_tokens) is not int or type(max_new_tokens) is not int or min(input_tokens,max_new_tokens)<=0:raise ValueError('positive integer token lengths required')
 if input_tokens+max_new_tokens>MAX_CONTEXT_TOKENS:raise ValueError('native prompt overflow without truncation')
 out={'do_sample':False,'max_new_tokens':max_new_tokens,'max_time':120,'eos_token_id':[248044,248046],'use_cache':True}
 if input_tokens>=LONG_CONTEXT_THRESHOLD:out['cache_implementation']='offloaded'
 return out

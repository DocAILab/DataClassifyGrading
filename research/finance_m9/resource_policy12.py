"""Deterministic cache placement only; never edit token inputs or decoding budgets."""
LONG_CONTEXT_THRESHOLD=32768
MAX_CONTEXT_TOKENS=49152

class ContextBudgetExceeded(ValueError):
    def __init__(self, input_tokens, max_new_tokens):
        super().__init__("native prompt overflow without truncation")
        self.evidence=dict(prompt_tokens=input_tokens,reserved_response_tokens=max_new_tokens,limit=MAX_CONTEXT_TOKENS)

def generation_options(input_tokens,max_new_tokens):
 if type(input_tokens) is not int or type(max_new_tokens) is not int or min(input_tokens,max_new_tokens)<=0:raise ValueError('positive integer token lengths required')
 if input_tokens+max_new_tokens>MAX_CONTEXT_TOKENS:raise ContextBudgetExceeded(input_tokens,max_new_tokens)
 out={'do_sample':False,'max_new_tokens':max_new_tokens,'max_time':120,'eos_token_id':[248044,248046],'use_cache':True}
 if input_tokens>=LONG_CONTEXT_THRESHOLD:out['cache_implementation']='offloaded'
 return out

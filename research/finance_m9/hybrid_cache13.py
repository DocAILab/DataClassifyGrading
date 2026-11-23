"""Instance-local full-attention-aware prefetch for a linear/full hybrid cache."""
def next_full_attention(layer_types,start):
 types=tuple(layer_types)
 if not types or any(x not in ('linear_attention','full_attention') for x in types) or 'full_attention' not in types:raise ValueError('supported linear/full hybrid layer contract required')
 if type(start) is not int or start<0:raise ValueError('nonnegative layer index required')
 if start>=len(types):start=0
 return next((i for i in range(start,len(types)) if types[i]=='full_attention'),types.index('full_attention'))

def cache_class(base):
 class AttentionAwareOffloadedCache(base):
  def __init__(self,config):
   self._attention_layer_types=tuple(config.layer_types);next_full_attention(self._attention_layer_types,0)
   super().__init__(config=config,offloading=True,offload_only_non_sliding=False)
  def prefetch(self,layer_idx,only_non_sliding=True):
   # Native Qwen linear-attention layers update state directly, without Cache.update.
   # Therefore generic '+1 layer' prefetch never reaches the next full-attention KV layer.
   # D2H offload runs on the producing stream. Fence it before the other stream reads host KV.
   import torch
   self.prefetch_stream.wait_stream(torch.cuda.current_stream())
   return super().prefetch(next_full_attention(self._attention_layer_types,layer_idx),only_non_sliding=False)
 return AttentionAwareOffloadedCache

def prepare_options(model,options):
 out=dict(options)
 if out.get('cache_implementation')!='offloaded':return out
 if 'past_key_values' in out:raise ValueError('never overwrite a caller cache')
 from transformers.cache_utils import DynamicCache
 cfg=model.config.get_text_config(decoder=True)
 out.pop('cache_implementation');out['past_key_values']=cache_class(DynamicCache)(cfg)
 return out

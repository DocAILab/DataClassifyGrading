"""Exact public-candidate token trie; no reference labels or new business facts."""
import math
def compile_trie(encoded):
 if len(encoded)!=12 or len(set(encoded))!=12:raise ValueError('the full12 catalog is required')
 sequences=list(encoded.values())
 if any(not s for s in sequences):raise ValueError('empty option encoding')
 common=[]
 for values in zip(*sequences):
  if len(set(values))!=1:break
  common.append(values[0])
 if not common:raise ValueError('common source prompt required')
 tree={};paths={}
 for option,seq in encoded.items():
  suffix=seq[len(common):]
  if not suffix:raise ValueError('option is a prefix of another')
  cursor=tree;path=[]
  for token in suffix:
   path.append(token);cursor=cursor.setdefault(token,{})
  if cursor:raise ValueError('option prefix ambiguity')
  paths[option]=suffix
 # Stable traversal, one independent causal forward for each internal prefix.
 nodes=[]
 def visit(node,branch):
  if not node:return
  nodes.append({'node_id':'n'+str(len(nodes)),'input_ids':common+branch,'branch':branch,'next_tokens':sorted(node)})
  for token,child in sorted(node.items()):visit(child,branch+[token])
 visit(tree,[])
 lookup={tuple(n['branch']):n['node_id'] for n in nodes};leaves={}
 for option,suffix in paths.items():
  leaves[option]=[{'node_id':lookup[tuple(suffix[:i])],'token_id':token} for i,token in enumerate(suffix)]
 return {'common_input_ids':common,'nodes':nodes,'leaves':leaves}
def rank_options(trie,node_logprobs,registry_order):
 if set(trie['leaves'])!=set(registry_order) or len(registry_order)!=12:raise ValueError('public option pool mismatch')
 scores={}
 for option,path in trie['leaves'].items():
  values=[node_logprobs[x['node_id']][str(x['token_id'])] for x in path]
  if not all(math.isfinite(v) for v in values):raise ValueError('nonfinite candidate score')
  scores[option]=sum(values)
 indices={v:i for i,v in enumerate(registry_order)}
 ordered=sorted(registry_order,key=lambda v:(-scores[v],indices[v]))
 return {'option_ids':ordered[:5],'scores':scores,'full_order':ordered}

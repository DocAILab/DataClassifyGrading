import random
def make_batches(n, accumulation, seed):
 if n<=0 or accumulation<=0:raise ValueError('positive training size and accumulation required')
 order=list(range(n));random.Random(seed).shuffle(order)
 return [order[i:i+accumulation] for i in range(0,n,accumulation)]
def loss_divisor(batch):
 if not batch:raise ValueError('empty training batch')
 return len(batch)

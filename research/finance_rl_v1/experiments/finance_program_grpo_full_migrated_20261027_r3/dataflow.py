"""Versioned executed data dependencies, separate from semantic usefulness."""
from copy import deepcopy


def leaves(value, prefix=''):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from leaves(item, prefix+'.'+str(key) if prefix else str(key))
    elif isinstance(value, list):
        for index, item in enumerate(value): yield from leaves(item, prefix+'.'+str(index))
    else: yield prefix, value


class VersionedFlow:
    def __init__(self): self.owners={}; self.nodes={}

    def producers(self, paths):
        return {owner for path in paths for field,owner in self.owners.items()
                if field==path or field.startswith(path+'.')}

    def invalidate(self, paths):
        for field in list(self.owners):
            if any(field==p or field.startswith(p+'.') for p in paths): del self.owners[field]

    def add(self, node_id, reads, writes, *, success=True, terminal=False, automatic=False):
        if node_id in self.nodes: raise ValueError('call identity must be unique')
        parents=self.producers(reads)
        self.nodes[node_id]=dict(parents=sorted(parents),reads=list(reads),writes=list(writes),
                                 success=success,terminal=terminal,automatic=automatic)
        if success:
            self.invalidate(writes)
            for path in writes:self.owners[path]=node_id

    def ancestors(self, node_id):
        seen=set();pending=list(self.nodes[node_id]['parents'])
        while pending:
            node=pending.pop()
            if node in seen:continue
            seen.add(node);pending.extend(self.nodes[node]['parents'])
        return seen

    def rewarded(self, terminal):
        return {n for n in self.ancestors(terminal) if self.nodes[n]['success']
                and not self.nodes[n]['automatic'] and not self.nodes[n]['terminal']}

    def snapshot(self):return deepcopy(dict(nodes=self.nodes,owners=self.owners))


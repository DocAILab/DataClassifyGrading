"""Private-identity, label-free access to one canonical Val metadata source."""
import copy
import json
import math
from pathlib import Path
from collections.abc import Mapping

FIELDS=('field_name','database_name','database_description','table_name','table_description','field_description','field_type','value')
MISSING={'','none','null','n/a','nan'}


def present(value):
    if value is None:
        return False
    if isinstance(value,str):
        return value.strip().casefold() not in MISSING
    if isinstance(value,float) and math.isnan(value):
        return False
    return bool(value) or value==0


class PoolMetadata:
    def __init__(self,rows):
        if not isinstance(rows,list):
            raise ValueError('canonical source must be a list')
        self._rows={}
        for row in rows:
            if not isinstance(row,Mapping) or not isinstance(row.get('id'),str) or not row['id'] or row['id'] in self._rows:
                raise ValueError('canonical rows require unique nonempty private IDs')
            raw=row.get('metadata')
            if not isinstance(raw,Mapping) or not present(row.get('domain')) or not present(raw.get('field_name')):
                raise ValueError('canonical row requires real domain and field_name')
            fields={k:copy.deepcopy(raw[k]) for k in FIELDS if present(raw.get(k))}
            self._rows[row['id']]={'domain':row['domain'],'fields':fields}

    @classmethod
    def from_canonical(cls,path):
        return cls(json.loads(Path(path).read_text(encoding='utf-8')))

    def _entry(self,record):
        entry=self._rows.get(record.source_id)
        if entry is None:
            raise ValueError('source id is absent from canonical metadata lookup')
        if entry['domain']!=record.domain or entry['fields']['field_name']!=record.field_name:
            raise ValueError('private lookup does not match domain and field_name')
        return entry

    def get(self,record):
        return copy.deepcopy(self._entry(record)['fields'])

    def _same_table(self,record):
        entry=self._entry(record); fields=entry['fields']
        db,table=fields.get('database_name'),fields.get('table_name')
        if not present(db) or not present(table):
            return []
        return sorted([r['fields'] for r in self._rows.values() if r['domain']==record.domain and r['fields'].get('database_name')==db and r['fields'].get('table_name')==table],key=lambda r:(str(r['field_name']),json.dumps(r,sort_keys=True,ensure_ascii=False)))

    def table_fields(self,record):
        own=self.get(record)
        neighbors=[r for r in self._same_table(record) if r['field_name']!=record.field_name]
        # One actual field name can have several source rows. Keep deterministic first.
        unique={}
        for row in neighbors:
            unique.setdefault(row['field_name'],row)
        neighbors=list(unique.values())
        visible=[{k:copy.deepcopy(v) for k,v in r.items() if k in ('field_name','field_description','field_type','value')} for r in neighbors[:30]]
        return {'database_name':own.get('database_name'),'table_name':own.get('table_name'),'fields':visible,
                'available_count':len(neighbors),'returned_count':len(visible),'truncated':len(neighbors)>30,
                'omitted_count':max(0,len(neighbors)-30),'coverage':'canonical-val-same-table-only',
                'missing_locators':[k for k in ('database_name','table_name') if k not in own]}

    def expand(self,record,names):
        if not isinstance(names,list) or not names or any(not isinstance(n,str) or not n.strip() for n in names) or len(set(names))!=len(names):
            raise ValueError('selected context must be a nonempty distinct list of actual field names')
        actual={r['field_name']:r for r in self.table_fields(record)['fields']}
        if any(name not in actual for name in names):
            raise ValueError('context field is outside actual bounded same-table candidates')
        return [copy.deepcopy(actual[n]) for n in names]

    def table_context(self,record):
        own=self.get(record)
        context=self.table_fields(record)
        return {'target':own,**context,'missing_target_attributes':[k for k in FIELDS if k not in own]}

    def database_context(self,record):
        own=self.get(record)
        db=own.get('database_name')
        if not present(db):
            return {'tables':[],'coverage':'canonical-val-available-tables-only','missing_locators':['database_name'],'truncated':False,'omitted_count':0,'available_count':0}
        tables={}
        omitted_rows=0
        for entry in self._rows.values():
            fields=entry['fields']
            if entry['domain']!=record.domain or fields.get('database_name')!=db:
                continue
            if not present(fields.get('table_name')):
                omitted_rows+=1
                continue
            table=tables.setdefault(fields['table_name'],{'table_name':fields['table_name'],'field_names':set(),'descriptions':set()})
            table['field_names'].add(fields['field_name'])
            if present(fields.get('table_description')):
                table['descriptions'].add(fields['table_description'])
        ordered=[]
        for name in sorted(tables):
            table=tables[name]; fields=sorted(table['field_names'])
            ordered.append({'table_name':name,'table_descriptions':sorted(table['descriptions']),
                'field_names':fields[:30],'available_field_count':len(fields),'omitted_field_count':max(0,len(fields)-30)})
        return {'database_name':db,'database_description':own.get('database_description'),
                'tables':ordered[:10],'available_count':len(ordered),'returned_count':min(10,len(ordered)),
                'truncated':len(ordered)>10,'omitted_count':max(0,len(ordered)-10),'omitted_unlocated_rows':omitted_rows,
                'coverage':'canonical-val-available-tables-only','missing_locators':[]}

"""Private declared-source-group lookup; no inferred database or category facts."""
import copy,importlib.util,json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
NATIVE=ROOT.parent/'finance_program_student_b80_eval_20261021_r2/vendor/finance_program_student_grpo_20261017_r3/runtime/vendor/pool_metadata.py'
spec=importlib.util.spec_from_file_location('scope_original_pure_metadata',NATIVE)
native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)

class SourceScopedMetadata(native.PoolMetadata):
    def __init__(self,rows,scope_rows):
        super().__init__(rows)
        originals={r['id']:r for r in rows}
        if len(scope_rows)!=len(originals) or {r.get('id') for r in scope_rows}!=set(originals):
            raise ValueError('source scope must bind exactly the original allowed metadata rows')
        self._scope={};signatures={}
        for scope in scope_rows:
            sid=scope['id'];row=originals[sid];fields=row['metadata']
            if scope.get('metadata')!=fields or scope.get('domain')!=row['domain']:
                raise ValueError('source scope metadata/domain differs from original source')
            table=scope.get('source_table_name');db=scope.get('source_database_name')
            if table!=fields.get('table_name','') or db!=fields.get('database_name',''):
                raise ValueError('source group locator differs from actual metadata')
            group=scope.get('source_table_id')
            if not isinstance(group,str) or not group.strip():raise ValueError('explicit declared source group required')
            signature=(row['domain'],db,table)
            if group in signatures and signatures[group]!=signature:
                raise ValueError('declared source group has conflicting domain/database/table identity')
            signatures[group]=signature;self._scope[sid]=group

    def _uses_declared_group(self,record):
        fields=self._entry(record)['fields']
        return not native.present(fields.get('database_name')) and native.present(fields.get('table_name'))

    def _same_table(self,record):
        if not self._uses_declared_group(record):return super()._same_table(record)
        entry=self._entry(record);fields=entry['fields'];group=self._scope[record.source_id]
        matched=[copy.deepcopy(row['fields']) for sid,row in self._rows.items()
                 if self._scope[sid]==group and row['domain']==record.domain
                 and row['fields'].get('table_name')==fields['table_name']]
        return sorted(matched,key=lambda row:(str(row['field_name']),json.dumps(row,sort_keys=True,ensure_ascii=False)))

    def table_fields(self,record):
        result=super().table_fields(record)
        if self._uses_declared_group(record):
            result['coverage']='declared-source-table-only-v1'
            result['scope_resolution']=dict(status='declared_named_source_table',physical_database_known=False)
        return result

import copy,importlib.util,json,sys,unittest
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
NATIVE=ROOT.parent/'finance_program_student_b80_eval_20261021_r2/vendor/finance_program_student_grpo_20261017_r3/runtime/vendor/pool_metadata.py'
spec=importlib.util.spec_from_file_location('original_pure_metadata',NATIVE)
native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
def make(rows,scopes):
    try:from source_scope import SourceScopedMetadata
    except ModuleNotFoundError:return native.PoolMetadata(rows)
    return SourceScopedMetadata(rows,scopes)
def row(sid,name,description,db='',table='T',domain='finance'):
    return dict(id=sid,domain=domain,metadata=dict(field_name=name,field_description=description,database_name=db,table_name=table))
def scope(rows,groups=None):
    return [dict(id=r['id'],domain=r['domain'],source_table_id=(groups or {}).get(r['id'],'source-table-a'),source_database_name=r['metadata'].get('database_name',''),source_table_name=r['metadata'].get('table_name',''),metadata=copy.deepcopy(r['metadata'])) for r in rows]
class ScopeTests(unittest.TestCase):
    def setUp(self):
        self.rows=[row('a','TARGET','资源全称'),row('b','ACTION','功能路径(action)')]
        self.record=SimpleNamespace(source_id='a',domain='finance',field_name='TARGET')
    def test_named_group_returns_actual_neighbor_without_inventing_db_or_exposing_private_id(self):
        lookup=make(self.rows,scope(self.rows));actual=lookup.table_fields(self.record)
        self.assertEqual(actual['fields'],[dict(field_name='ACTION',field_description='功能路径(action)')])
        self.assertIsNone(actual['database_name']);self.assertEqual(actual['missing_locators'],['database_name'])
        self.assertEqual(actual['coverage'],'declared-source-table-only-v1')
        self.assertNotIn('source-table-a',json.dumps(actual));self.assertNotIn('database_name',lookup.get(self.record))
    def test_unknown_none_group_never_supplies_neighbors(self):
        rows=[row('a','TARGET','成交数量',db='FA',table='NONE'),row('b','OTHER','股票',db='FA',table='NONE')]
        self.assertEqual(make(rows,scope(rows)).table_fields(self.record)['fields'],[])
    def test_same_table_name_different_declared_group_never_joined(self):
        self.assertEqual(make(self.rows,scope(self.rows,{'a':'source-table-a','b':'source-table-b'})).table_fields(self.record)['fields'],[])
    def test_same_group_conflicting_named_table_identity_rejected(self):
        rows=self.rows+[row('c','OTHER','其他',table='U')]
        with self.assertRaises(ValueError):make(rows,scope(rows))
    def test_scope_metadata_mismatch_rejected(self):
        scopes=scope(self.rows);scopes[1]['metadata']['field_description']='invented'
        with self.assertRaises(ValueError):make(self.rows,scopes)
    def test_existing_complete_locator_queries_remain_exact(self):
        rows=[row('a','TARGET','原文',db='DB'),row('b','OTHER','旁字段',db='DB')]
        old=native.PoolMetadata(rows);new=make(rows,scope(rows))
        self.assertEqual(old.get(self.record),new.get(self.record))
        self.assertEqual(old.table_fields(self.record),new.table_fields(self.record))
        self.assertEqual(old.database_context(self.record),new.database_context(self.record))
if __name__=='__main__':unittest.main()

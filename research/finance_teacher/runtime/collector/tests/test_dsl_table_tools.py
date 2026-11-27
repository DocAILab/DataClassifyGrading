"""Exercise the shared public table contract against the native metadata provider."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dsl_table_tools import install
from smoke import science


def fixture(entries):
    rows, scopes, records = [], [], []
    for n, (db, table, name) in enumerate(entries):
        sid = str(n)
        fields = dict(database_name=db, table_name=table, field_name=name, field_description="description " + name)
        row = dict(id=sid, domain="finance", metadata=fields)
        rows.append(row)
        scopes.append(dict(**row, source_database_name=db, source_table_name=table, source_table_id=sid))
        records.append(SimpleNamespace(source_id=sid, domain="finance", field_name=name))
    lookup = science.runtime().coverage_pool.scope.SourceScopedMetadata(rows, scopes)
    controller = SimpleNamespace(bridge=SimpleNamespace(executor=SimpleNamespace(metadata_lookup=lookup)))
    install(controller)
    return lookup, records


class DSLTableTests(unittest.TestCase):
    def test_public_names_then_details(self):
        lookup, (target, _, missing) = fixture([("", "A", "target"), ("", "A", "neighbor"), ("", "", "other")])
        self.assertEqual(lookup.table_fields(target)["fields"], [{"field_name": "neighbor"}])
        details = [{"field_name": "neighbor", "field_description": "description neighbor"}]
        self.assertEqual(lookup.expand(target, ["neighbor"]), details)
        self.assertEqual(lookup.table_context(target)["fields"], details)
        self.assertEqual(lookup.table_fields(missing)["fields"], [])
        with self.assertRaises(ValueError):
            lookup.expand(missing, ["neighbor"])

    def test_table_name_collision_across_two_databases(self):
        lookup, (unknown, target, _, _) = fixture([
            ("", "T", "unknown"), ("A", "T", "id"), ("A", "T", "amount"), ("B", "T", "owner")])
        self.assertEqual(lookup.table_fields(unknown)["fields"], [])
        self.assertEqual(lookup.table_fields(target)["fields"], [{"field_name": "amount"}])

    def test_bounded_names_and_complete_context(self):
        lookup, records = fixture([("db", "T", f"f{i:02d}") for i in range(34)])
        first = lookup.table_fields(records[0])
        self.assertEqual((first["returned_count"], first["omitted_count"]), (30, 3))
        self.assertEqual(lookup.table_context(records[0])["returned_count"], 33)
        with self.assertRaises(ValueError):
            lookup.expand(records[0], ["f33"])

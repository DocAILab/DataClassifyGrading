"""Instance-local typed admission for actually acquired declared-source neighbors.

This extends the metadata transport type, not semantic sufficiency or terminal
acceptance. It never changes database_name, returned facts or native vendor code.
"""
import inspect
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROVIDER = ROOT.parent / 'finance_program_recovery_source_scope_20261024_r9/source_scope.py'

def install(controller):
    executor = controller.bridge.executor
    guard = controller.bridge.guard
    lookup = executor.metadata_lookup
    if (Path(inspect.getsourcefile(type(lookup).__init__)).resolve() != PROVIDER.resolve()
            or type(lookup).__name__ != 'SourceScopedMetadata'):
        raise ValueError('exact frozen private declared-source provider required')
    original = guard._normalize_policy_decision.__func__.__globals__['_binding']
    strict_equal = guard._normalize_policy_decision.__func__.__globals__['bind_call'].__globals__['strict_equal']

    def binding(name, record, state):
        canonical = original(name, record, state)
        if name != 'table_fields' or canonical is not None:
            return canonical
        if executor.metadata_lookup is not lookup:
            return None
        if state.get('_provenance', {}).get(name) != 'GET_TABLE_FIELDS':
            return None
        value = state.get(name)
        if not isinstance(value, dict) or not lookup._uses_declared_group(record):
            return None
        try:
            actual = lookup.table_fields(record)
        except (ValueError, KeyError, TypeError):
            return None
        if not strict_equal(value, actual):
            return None
        fields = value.get('fields')
        if (value.get('coverage') != 'declared-source-table-only-v1'
                or value.get('missing_locators') != ['database_name']
                or value.get('database_name') is not None
                or value.get('scope_resolution') != dict(status='declared_named_source_table', physical_database_known=False)
                or not isinstance(fields, list) or not 0 < len(fields) <= 30
                or value.get('returned_count') != len(fields)
                or type(value.get('returned_count')) is not int
                or type(value.get('available_count')) is not int
                or value['available_count'] < len(fields)):
            return None
        return '$state.table_fields'

    def rebound(method):
        function = method.__func__
        namespace = dict(function.__globals__, _binding=binding)
        clone = types.FunctionType(function.__code__, namespace, function.__name__, function.__defaults__, function.__closure__)
        clone.__kwdefaults__ = function.__kwdefaults__
        return types.MethodType(clone, method.__self__)

    # Both boundaries must use the same resolver; all other method code stays exact.
    guard._normalize_policy_decision = rebound(guard._normalize_policy_decision)
    executor._validate_consumed_raw = rebound(executor._validate_consumed_raw)
    controller.scope_binding_contract = 'exact-acquired-declared-source-v1'
    return controller

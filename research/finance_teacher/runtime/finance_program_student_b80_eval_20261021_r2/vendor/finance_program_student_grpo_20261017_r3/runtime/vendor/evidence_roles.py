"""Source roles and a copied Top1-only comparison explanation projection."""
from copy import deepcopy
from collections.abc import Mapping

from pool_catalog import pool_catalog
from pool_metadata import FIELDS
from grounded_semantics import valid_shape as grounded_shape


def acquired_source(key, source):
    """Only declared deterministic acquisition outputs establish raw facts."""
    spec = pool_catalog('F66').get(source)
    if spec is None or spec.role != 'metadata_acquisition':
        return False
    return key in spec.produces or (key in FIELDS and
        bool({'field_metadata', 'acquired_metadata'} & set(spec.produces)))


def evidence_roles(metadata, semantics, alignments, observations, provenance):
    result = {}
    for values, default in ((metadata, 'raw_fact'), (observations, 'raw_fact'),
                            (semantics, 'model_interpretation'), (alignments, 'comparison_inference')):
        for key, value in values.items():
            source = 'initial_record' if key == 'field_name' else provenance.get(key)
            role = default
            if default == 'model_interpretation' and grounded_shape(value):
                result[key] = {'role': 'source_anchored_description' if value['status']=='observed_description' else 'unresolved_hypothesis',
                    'value': deepcopy(value), 'provenance': source, 'semantic_consistency_verified': False}
                continue
            if default == 'model_interpretation' and source and (
                    source.endswith('_NAME') or 'ABBREVIATION' in source):
                role = 'model_hypothesis'
            result[key] = {'role': role, 'value': deepcopy(value), 'provenance': source}
    return result


def project_top1_evidence(evidence, comparison_reason):
    """Remove comparison reasons and value aliases, retaining acquired facts.

    Only callers constructing Top1 messages use this view. Shared publication,
    binding, Planner feedback and all other Skill messages retain their inputs.
    """
    omitted = object()
    catalog = pool_catalog('F66')
    independent_keys = {key for spec in catalog.values()
                        if spec.role in ('semantic_interpretation', 'standard_alignment', 'evidence_reasoning')
                        and spec.name != 'COMPARE_STANDARD_CANDIDATES' for key in spec.produces}

    def project(value, protected=False):
        if protected:
            return deepcopy(value)
        if isinstance(value, Mapping):
            provenance = value.get('provenance')
            source = provenance if isinstance(provenance, str) else None
            if value.get('value') == comparison_reason and (source == 'COMPARE_STANDARD_CANDIDATES' or
                    value.get('role') == 'comparison_inference' and source not in catalog):
                return omitted
            protected = protected or value.get('role') == 'raw_fact' or (
                source in catalog and source != 'COMPARE_STANDARD_CANDIDATES' and 'role' in value)
            if protected:
                return deepcopy(value)
            result = {}
            for key, item in value.items():
                if key == 'standard_comparison':
                    continue
                projected = project(item, protected or key in ('metadata', 'observations') or key in independent_keys)
                if projected is not omitted:
                    result[key] = projected
            return result
        if isinstance(value, list):
            return [item for original in value if (item := project(original, protected)) is not omitted]
        if not protected and isinstance(comparison_reason, str) and value == comparison_reason:
            return omitted
        return deepcopy(value)

    result = project(evidence)
    return {} if result is omitted else result

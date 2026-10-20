"""Explicit protocol variant: transport doesn't inject data or repair outputs."""
# 此协议变体只追加版本化的证据解释规则；原始消息保留，响应仍由执行器按合约校验。
from copy import deepcopy
import re


def apply_policy(messages):
    if not messages or messages[0].get('role') != 'system':raise ValueError('trusted first system required')
    head = messages[0]['content'].splitlines()[0]
    planner = 'You are the Global Program Planner for a database-field classification task.'
    if head != planner and not re.fullmatch(r'Execute only the selected Skill ([A-Z_]+)\.', head):raise ValueError('unknown role')
    value = deepcopy(messages)
    value[0]['content'] += ('\nEvidence protocol variant: Source-grounded semantic objects record observed descriptions or unknown '
        'without adjudicating name-description consistency. Follow each versioned Skill contract; never turn unknown into '
        'conflict or use model repetition as independent evidence. No fabricated source facts, labels, enum values or future observations. '
        'Planner remains autonomous and budgets/terminal rules are unchanged. Use actual acquired context and preserve typed outputs.')
    if head in {'Execute only the selected Skill ' + name + '.' for name in (
            'SELECT_CATEGORY_TOPK', 'INFER_CATEGORY_FROM_TOPK', 'SEARCH_STANDARD_BY_FIELD', 'INFER_CATEGORY')}:
        value[0]['content'] += ('\nClassification evidence calibration grounded-02: '
            'Candidate taxonomy descriptions are definitions, not observed source facts. '
            'A word or business event appearing only in a candidate definition is not evidence that this field belongs to that business. '
            'Choose the required option IDs using the available source evidence, but do not backfill specific entities, products, '
            'events or workflows into the source from the selected definition. The summary is a classification inference, not a new '
            'observation; indicate inference and any unresolved specificity within the existing brief summary. '
            'If source context cannot distinguish a more specific business meaning, keep that uncertainty explicit rather than '
            'asserting the specific meaning as verified. Do not change the required JSON keys, candidate choices or execution rules.')
    return value

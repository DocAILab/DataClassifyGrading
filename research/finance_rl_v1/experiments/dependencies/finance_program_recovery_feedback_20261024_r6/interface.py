"""Expose an index of current eligible raw facts; do not adjudicate sufficiency."""
import copy,importlib.util,json,re
from pathlib import Path
path=Path(__file__).resolve().parent.parent/'finance_program_recovery_feedback_20261024_r5/interface.py'
spec=importlib.util.spec_from_file_location('same_candidate_scope_interface',path)
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
def rewrite(messages,*,session=None):
    actual=previous.rewrite(messages,session=session)
    stage=re.match(r'Execute only the selected Skill ([A-Z_]+)\.',actual[0]['content'])
    if not stage or stage.group(1) not in ('INFER_CATEGORY','INFER_CATEGORY_FROM_TOPK'):return actual
    body=json.loads(actual[-1]['content']);facts=(body.get('evidence_boundary') or {}).get('facts',{})
    brief=[dict(fact_id=key,value=copy.deepcopy(fact['value']),provenance=fact.get('provenance'),source_scope=fact.get('source_scope'))
        for key,fact in facts.items() if fact.get('decisive_eligible') is True]
    body=dict(decision_source_brief=brief,source_brief_adds_evidence=False,source_brief_is_sufficiency_judgment=False,**body)
    actual[-1]['content']=json.dumps(body,ensure_ascii=False,separators=(',',':'))
    actual[0]['content']+= ('\n当前来源事实核对：decision_source_brief只是当前evidence_boundary.facts中可参与判断的原始资料索引，'
        '不是新证据，不表示足够。每个fact_id都是当前边界中真实存在的精确键，引用时逐字复制，不替换成历史checkpoint键或另一个字段名。'
        '先阅读当前所有来源说明，再对照候选的完整定义；同表实际取得的字段说明可以说明该表的业务背景，'
        '但不能把邻字段的属性或取值冒充目标字段的属性或取值，更不能使用其他表的信息。'
        '如果补证后出现了实际同表业务资料，不能只复述补证前的不足而忽略当前内容；历史gap没有否决新事实的效力。'
        '是否允许重判仍由原生persistent-gap guard决定，本索引不裁定事实是否新增、相关、真实或足够。'
        '业务类型可以由目标字段定义与有来源的同表背景共同支持，不要求目标字段单独重复写出全部业务类型。'
        '反之，通用的字段名、领域、计数、类型、示例值或模型解释不能补出缺失的资产类型/主体身份/内部或客户关系。'
        '必须保留真正未决的区别；不可为完成任务猜一个类别。输出仍严格遵守原JSON结构和充分/不足两分支，不增加输出键。')
    return actual
class RecoverySession(previous.RecoverySession):
    def rewrite(self,messages):return rewrite(messages,session=self)
class RecoveryAdapter(previous.RecoveryAdapter):
    def __init__(self,backend):
        super().__init__(backend);self.session=RecoverySession()

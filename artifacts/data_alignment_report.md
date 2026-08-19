# 数据对齐分析报告 (dataset ↔ corpus alignment)

> 方法约束：仅使用精确字符串与空白归一化匹配，不使用语义模型或模糊匹配修标签。
> 数据来源：`data/<dataset>/all.json`、`data/<dataset>/corpus.json`、`data/knowledge/standards_map/*.json`。

## 1. 数据集 (datasets)

| dataset | samples | L1 | L2 | L3 | L4 | 完整 4 级路径 | 最深级别分布 | data_level 字段分布 |
|---|---|---|---|---|---|---|---|---|
| finance | 568 | 3 (100.0%) | 7 (100.0%) | 10 (67.1%) | 25 (100.0%) | 381 | {'level_4': 568} | {'L2': 460, 'L4': 7, 'L3': 76, 'L1': 25} |
| infra | 64 | 3 (100.0%) | 3 (100.0%) | 3 (100.0%) | 4 (100.0%) | 64 | {'level_4': 64} | {'L3': 46, 'L2': 18} |
| pers_info | 176 | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 18 (100.0%) | 0 | {'level_4': 176} | {'L1': 31, 'L2': 98, 'L3': 47} |
| shougang | 19415 | 3 (100.0%) | 15 (100.0%) | 54 (100.0%) | 193 (100.0%) | 19415 | {'level_4': 19415} | {'L3': 4328, 'L2': 14188, 'L1': 899} |

### dataset `finance`

- 填充模式（样本数）：
  - `level_1/level_2/level_3/level_4` × 381
  - `level_1/level_2/level_4` × 187
- 唯一完整路径数：28
- leaf 同名不同父路径 (collision)：
  - level_2: 1 个 label 出现在不同父路径下
    - `单位` ← 业务 ; 客户
  - level_3: 1 个 label 出现在不同父路径下
    - `单位基本信息` ← 业务/单位 ; 客户/单位
  - level_4: 3 个 label 出现在不同父路径下
    - `交易清结算信息` ← 业务/交易信息/交易通用信息 ; 业务/账户信息/
    - `单位基本情况` ← 业务/单位/单位基本信息 ; 客户/单位/单位基本信息
    - `基本信息` ← 业务/合约协议/合同通用信息 ; 业务/账户信息/

### dataset `infra`

- 填充模式（样本数）：
  - `level_1/level_2/level_3/level_4` × 64
- 唯一完整路径数：4
- leaf 同名不同父路径：无

### dataset `pers_info`

- 填充模式（样本数）：
  - `level_4` × 176
- 唯一完整路径数：18
- leaf 同名不同父路径：无

### dataset `shougang`

- 填充模式（样本数）：
  - `level_1/level_2/level_3/level_4` × 19415
- 唯一完整路径数：203
- leaf 同名不同父路径 (collision)：
  - level_4: 1 个 label 出现在不同父路径下
    - `——` ← 生产数据域/产出管理/实重管理 ; 生产数据域/物流管理/销售物流管理 ; 生产数据域/生产合同（订单）/合同归并 ; 生产数据域/生产合同（订单）/合同跟踪 ; 生产数据域/生产合同（订单）/转用充当 ; 生产数据域/生产计划/中厚板作业计划 ; 生产数据域/生产计划/冷轧作业计划 ; 生产数据域/生产计划/撕分线作业计划 ; 生产数据域/生产计划/热轧作业计划 ; 管理数据域/环境管理/监测分析管理 ; 管理数据域/生产质量管理/质保书管理

## 2. Corpus / Standard 词典

| file | entries | 值类型 | category 字段 | 含 code | 多段名(可能 path) | 长描述 | 唯一 category | 重复 category | 唯一 name | 重复 name | malformed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| corpus:finance | 237 | dict | no | 0 | 0 | 0 | 220/237 | 12k/17i | 220/237 | 12k/17i | 0 |
| standard:beijing_fta_dict.json | 39 | dict | yes | 0 | 0 | 7 | 37/39 | 1k/2i | 37/39 | 1k/2i | 0 |
| standard:education_dict.json | 113 | dict | yes | 113 | 0 | 0 | 113/113 | 0k/0i | 109/113 | 3k/4i | 0 |
| standard:financial_standards_dict.json | 237 | dict | yes | 0 | 237 | 0 | 233/237 | 1k/4i | 220/237 | 12k/17i | 0 |
| standard:general_personal_info_dict.json | 7 | dict | yes | 0 | 0 | 0 | 7/7 | 0k/0i | 7/7 | 0k/0i | 0 |
| standard:general_personal_info_expanded.json | 6 | str | no | 0 | 0 | 0 | 6/6 | 0k/0i | 6/6 | 0k/0i | 0 |
| standard:guanji_dict.json | 234 | dict | yes | 233 | 18 | 0 | 234/234 | 0k/0i | 234/234 | 0k/0i | 0 |
| standard:iov_standards_dict.json | 33 | dict | yes | 0 | 0 | 0 | 11/33 | 11k/22i | 11/33 | 11k/22i | 0 |
| standard:iov_white_papers_dict.json | 33 | dict | yes | 0 | 0 | 0 | 28/33 | 5k/5i | 28/33 | 5k/5i | 0 |
| standard:iov_white_papers_dict_before.json | 133 | dict | yes | 0 | 0 | 0 | 99/133 | 21k/34i | 99/133 | 21k/34i | 0 |
| standard:sensitive_personal_info_dict.json | 8 | dict | yes | 0 | 0 | 0 | 8/8 | 0k/0i | 8/8 | 0k/0i | 0 |
| standard:sensitive_personal_info_expanded.json | 8 | str | no | 0 | 0 | 2 | 8/8 | 0k/0i | 8/8 | 0k/0i | 0 |
| standard:shanghai_fta_dict.json | 64 | dict | yes | 0 | 0 | 0 | 63/64 | 1k/1i | 63/64 | 1k/1i | 0 |

### corpus:finance

- 格式示例：
  - raw=`个人基本概况信息` name=`个人基本概况信息` code=`` dash_parts=['个人基本概况信息']
  - raw=`个人财产信息` name=`个人财产信息` code=`` dash_parts=['个人财产信息']
  - raw=`个人联系信息` name=`个人联系信息` code=`` dash_parts=['个人联系信息']
  - raw=`个人健康生理信息` name=`个人健康生理信息` code=`` dash_parts=['个人健康生理信息']
  - raw=`个人地理位置信息` name=`个人地理位置信息` code=`` dash_parts=['个人地理位置信息']
- 注：该 corpus 的 '重复 category' 是同一 level_4 下的多条文档（237 条文档 / 220 个唯一 label），不是 category 本身重复；每次出现只算一条 category。
- 重复完整 category：
  - `基本信息` × 7
  - `传统鉴别信息` × 2
  - `公私间关系信息` × 2
  - `行为信息` × 2
  - `基础标签信息` × 2
  - `关系标签信息` × 2
  - `签约标签信息` × 2
  - `交易类标签信息` × 2
  - `行为标签信息` × 2
  - `风险标签信息` × 2

### standard:guanji_dict.json

- 格式示例：
  - raw=`科研设备预约管理（A1-1-1）` name=`科研设备预约管理` code=`A1-1-1` dash_parts=['科研设备预约管理']
  - raw=`科研进程管控（A1-1-2）` name=`科研进程管控` code=`A1-1-2` dash_parts=['科研进程管控']
  - raw=`非常规委托检测管理（A1-1-3）` name=`非常规委托检测管理` code=`A1-1-3` dash_parts=['非常规委托检测管理']
  - raw=`考试管理（A1-1-4）` name=`考试管理` code=`A1-1-4` dash_parts=['考试管理']
  - raw=`薄板合同产线专业化设计（B1-1-1）` name=`薄板合同产线专业化设计` code=`B1-1-1` dash_parts=['薄板合同产线专业化设计']
- code 结构：233 个，唯一 233 个，深度分布 {2: 9, 3: 224}，示例 ['A1-1-1', 'A1-1-2', 'A1-1-3', 'A1-1-4', 'B1-1-1', 'B1-1-2', 'B1-2', 'B1-3-1']
  - 短 code（<3 段）类目：[{'code': 'B1-2', 'name': '合同归并'}, {'code': 'B1-5', 'name': '合同跟踪'}, {'code': 'B3-3', 'name': '热轧作业计划'}, {'code': 'B3-4', 'name': '冷轧作业计划'}, {'code': 'B3-5', 'name': '撕分线作业计划'}, {'code': 'B4-2', 'name': '实重管理'}, {'code': 'B6-2', 'name': '销售物流管理'}, {'code': 'C1-5', 'name': '质保书管理'}, {'code': 'C4-1', 'name': '监测分析管理'}]

## 3. dataset ↔ corpus 对齐矩阵（leaf_normalized 样本覆盖率）

| dataset | corpus:finance | standard:beijing_fta_dict.json | standard:education_dict.json | standard:financial_standards_dict.json | standard:general_personal_info_dict.json | standard:general_personal_info_expanded.json | standard:guanji_dict.json | standard:iov_standards_dict.json | standard:iov_white_papers_dict.json | standard:iov_white_papers_dict_before.json | standard:sensitive_personal_info_dict.json | standard:sensitive_personal_info_expanded.json | standard:shanghai_fta_dict.json |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| finance | 94.0% | 0.0% | 32.2% | 94.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| infra | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| pers_info | 0.0% | 0.0% | 19.9% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| shougang | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 94.7% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.3% |

| dataset | 最佳 corpus | leaf 匹配 (unique) | leaf_normalized 样本覆盖率 |
|---|---|---|---|
| finance | corpus:finance | 20/25 | 94.0% |
| infra | standard:guanji_dict.json | 4/4 | 100.0% |
| pers_info | standard:education_dict.json | 4/18 | 19.9% |
| shougang | standard:guanji_dict.json | 192/193 | 94.7% |

### dataset `finance` 对齐明细

- **corpus:finance**：
  - full_path_exact=0 normalized_path=0 leaf_exact=534 leaf_normalized=534 leaf_ancestor=534 ambiguous=0 unmatched=34
  - unmatched leaf 示例：['交易清金额信息', '单位基本信息', '单位基本情况', '单位联系人信息', '基本信息（公开']
- **standard:beijing_fta_dict.json**：
  - full_path_exact=0 normalized_path=0 leaf_exact=0 leaf_normalized=0 leaf_ancestor=0 ambiguous=0 unmatched=568
  - unmatched leaf 示例：['个人基本概况信息', '个人联系信息', '交易基本信息', '交易对手信息', '交易清结算信息', '交易清金额信息', '交易记账信息', '交易金额信息', '传统鉴别信息', '信托运用信息', '关系标签信息', '单位基本信息', '单位基本情况', '单位基本概况', '单位联系人信息']
- **standard:education_dict.json**：
  - full_path_exact=0 normalized_path=0 leaf_exact=183 leaf_normalized=183 leaf_ancestor=183 ambiguous=183 unmatched=385
  - ambiguous leaf 示例：[{'leaf': '基本信息', 'categories': ['A1-1:基本信息', 'A3-1:基本信息', 'A4-1:基本信息']}]
  - unmatched leaf 示例：['个人基本概况信息', '个人联系信息', '交易基本信息', '交易对手信息', '交易清结算信息', '交易清金额信息', '交易记账信息', '交易金额信息', '传统鉴别信息', '信托运用信息', '关系标签信息', '单位基本信息', '单位基本情况', '单位基本概况', '单位联系人信息']
- **standard:financial_standards_dict.json**：
  - full_path_exact=531 normalized_path=531 leaf_exact=534 leaf_normalized=534 leaf_ancestor=531 ambiguous=199 unmatched=34
  - ambiguous leaf 示例：[{'leaf': '传统鉴别信息', 'categories': ['客户-个人-传统鉴别信息', '客户-单位-传统鉴别信息']}, {'leaf': '关系标签信息', 'categories': ['客户-个人-关系标签信息', '客户-单位-关系标签信息']}, {'leaf': '基本信息', 'categories': ['业务-合约协议-基本信息', '业务-账户信息-基本信息', '经营管理-营销服务-基本信息']}]
  - unmatched leaf 示例：['交易清金额信息', '单位基本信息', '单位基本情况', '单位联系人信息', '基本信息（公开']
- **standard:general_personal_info_dict.json**：
  - full_path_exact=0 normalized_path=0 leaf_exact=0 leaf_normalized=0 leaf_ancestor=0 ambiguous=0 unmatched=568
  - unmatched leaf 示例：['个人基本概况信息', '个人联系信息', '交易基本信息', '交易对手信息', '交易清结算信息', '交易清金额信息', '交易记账信息', '交易金额信息', '传统鉴别信息', '信托运用信息', '关系标签信息', '单位基本信息', '单位基本情况', '单位基本概况', '单位联系人信息']
- **standard:general_personal_info_expanded.json**：
  - full_path_exact=0 normalized_path=0 leaf_exact=0 leaf_normalized=0 leaf_ancestor=0 ambiguous=0 unmatched=568
  - unmatched leaf 示例：['个人基本概况信息', '个人联系信息', '交易基本信息', '交易对手信息', '交易清结算信息', '交易清金额信息', '交易记账信息', '交易金额信息', '传统鉴别信息', '信托运用信息', '关系标签信息', '单位基本信息', '单位基本情况', '单位基本概况', '单位联系人信息']
- **standard:guanji_dict.json**：
  - full_path_exact=0 normalized_path=0 leaf_exact=0 leaf_normalized=0 leaf_ancestor=0 ambiguous=0 unmatched=568
  - unmatched leaf 示例：['个人基本概况信息', '个人联系信息', '交易基本信息', '交易对手信息', '交易清结算信息', '交易清金额信息', '交易记账信息', '交易金额信息', '传统鉴别信息', '信托运用信息', '关系标签信息', '单位基本信息', '单位基本情况', '单位基本概况', '单位联系人信息']
- **standard:iov_standards_dict.json**：
  - full_path_exact=0 normalized_path=0 leaf_exact=0 leaf_normalized=0 leaf_ancestor=0 ambiguous=0 unmatched=568
  - unmatched leaf 示例：['个人基本概况信息', '个人联系信息', '交易基本信息', '交易对手信息', '交易清结算信息', '交易清金额信息', '交易记账信息', '交易金额信息', '传统鉴别信息', '信托运用信息', '关系标签信息', '单位基本信息', '单位基本情况', '单位基本概况', '单位联系人信息']
- **standard:iov_white_papers_dict.json**：
  - full_path_exact=0 normalized_path=0 leaf_exact=0 leaf_normalized=0 leaf_ancestor=0 ambiguous=0 unmatched=568
  - unmatched leaf 示例：['个人基本概况信息', '个人联系信息', '交易基本信息', '交易对手信息', '交易清结算信息', '交易清金额信息', '交易记账信息', '交易金额信息', '传统鉴别信息', '信托运用信息', '关系标签信息', '单位基本信息', '单位基本情况', '单位基本概况', '单位联系人信息']
- **standard:iov_white_papers_dict_before.json**：
  - full_path_exact=0 normalized_path=0 leaf_exact=0 leaf_normalized=0 leaf_ancestor=0 ambiguous=0 unmatched=568
  - unmatched leaf 示例：['个人基本概况信息', '个人联系信息', '交易基本信息', '交易对手信息', '交易清结算信息', '交易清金额信息', '交易记账信息', '交易金额信息', '传统鉴别信息', '信托运用信息', '关系标签信息', '单位基本信息', '单位基本情况', '单位基本概况', '单位联系人信息']
- **standard:sensitive_personal_info_dict.json**：
  - full_path_exact=0 normalized_path=0 leaf_exact=0 leaf_normalized=0 leaf_ancestor=0 ambiguous=0 unmatched=568
  - unmatched leaf 示例：['个人基本概况信息', '个人联系信息', '交易基本信息', '交易对手信息', '交易清结算信息', '交易清金额信息', '交易记账信息', '交易金额信息', '传统鉴别信息', '信托运用信息', '关系标签信息', '单位基本信息', '单位基本情况', '单位基本概况', '单位联系人信息']
- **standard:sensitive_personal_info_expanded.json**：
  - full_path_exact=0 normalized_path=0 leaf_exact=0 leaf_normalized=0 leaf_ancestor=0 ambiguous=0 unmatched=568
  - unmatched leaf 示例：['个人基本概况信息', '个人联系信息', '交易基本信息', '交易对手信息', '交易清结算信息', '交易清金额信息', '交易记账信息', '交易金额信息', '传统鉴别信息', '信托运用信息', '关系标签信息', '单位基本信息', '单位基本情况', '单位基本概况', '单位联系人信息']
- **standard:shanghai_fta_dict.json**：
  - full_path_exact=0 normalized_path=0 leaf_exact=0 leaf_normalized=0 leaf_ancestor=0 ambiguous=0 unmatched=568
  - unmatched leaf 示例：['个人基本概况信息', '个人联系信息', '交易基本信息', '交易对手信息', '交易清结算信息', '交易清金额信息', '交易记账信息', '交易金额信息', '传统鉴别信息', '信托运用信息', '关系标签信息', '单位基本信息', '单位基本情况', '单位基本概况', '单位联系人信息']

### dataset `infra` 对齐明细

- **corpus:finance**：
  - full_path_exact=0 normalized_path=0 leaf_exact=0 leaf_normalized=0 leaf_ancestor=0 ambiguous=0 unmatched=64
  - unmatched leaf 示例：['厚板组板设计', '水质监测管理', '薄板合同产线专业化设计', '非常规委托检测管理']
- **standard:beijing_fta_dict.json**：
  - full_path_exact=0 normalized_path=0 leaf_exact=0 leaf_normalized=0 leaf_ancestor=0 ambiguous=0 unmatched=64
  - unmatched leaf 示例：['厚板组板设计', '水质监测管理', '薄板合同产线专业化设计', '非常规委托检测管理']
- **standard:education_dict.json**：
  - full_path_exact=0 normalized_path=0 leaf_exact=0 leaf_normalized=0 leaf_ancestor=0 ambiguous=0 unmatched=64
  - unmatched leaf 示例：['厚板组板设计', '水质监测管理', '薄板合同产线专业化设计', '非常规委托检测管理']
- **standard:financial_standards_dict.json**：
  - full_path_exact=0 normalized_path=0 leaf_exact=0 leaf_normalized=0 leaf_ancestor=0 ambiguous=0 unmatched=64
  - unmatched leaf 示例：['厚板组板设计', '水质监测管理', '薄板合同产线专业化设计', '非常规委托检测管理']
- **standard:general_personal_info_dict.json**：
  - full_path_exact=0 normalized_path=0 leaf_exact=0 leaf_normalized=0 leaf_ancestor=0 ambiguous=0 unmatched=64
  - unmatched leaf 示例：['厚板组板设计', '水质监测管理', '薄板合同产线专业化设计', '非常规委托检测管理']
- **standard:general_personal_info_expanded.json**：
  - full_path_exact=0 normalized_path=0 leaf_exact=0 leaf_normalized=0 leaf_ancestor=0 ambiguous=0 unmatched=64
  - unmatched leaf 示例：['厚板组板设计', '水质监测管理', '薄板合同产线专业化设计', '非常规委托检测管理']
- **standard:guanji_dict.json**：
  - full_path_exact=0 normalized_path=0 leaf_exact=64 leaf_normalized=64 leaf_ancestor=64 ambiguous=0 unmatched=0
- **standard:iov_standards_dict.json**：
  - full_path_exact=0 normalized_path=0 leaf_exact=0 leaf_normalized=0 leaf_ancestor=0 ambiguous=0 unmatched=64
  - unmatched leaf 示例：['厚板组板设计', '水质监测管理', '薄板合同产线专业化设计', '非常规委托检测管理']
- **standard:iov_white_papers_dict.json**：
  - full_path_exact=0 normalized_path=0 leaf_exact=0 leaf_normalized=0 leaf_ancestor=0 ambiguous=0 unmatched=64
  - unmatched leaf 示例：['厚板组板设计', '水质监测管理', '薄板合同产线专业化设计', '非常规委托检测管理']
- **standard:iov_white_papers_dict_before.json**：
  - full_path_exact=0 normalized_path=0 leaf_exact=0 leaf_normalized=0 leaf_ancestor=0 ambiguous=0 unmatched=64
  - unmatched leaf 示例：['厚板组板设计', '水质监测管理', '薄板合同产线专业化设计', '非常规委托检测管理']
- **standard:sensitive_personal_info_dict.json**：
  - full_path_exact=0 normalized_path=0 leaf_exact=0 leaf_normalized=0 leaf_ancestor=0 ambiguous=0 unmatched=64
  - unmatched leaf 示例：['厚板组板设计', '水质监测管理', '薄板合同产线专业化设计', '非常规委托检测管理']
- **standard:sensitive_personal_info_expanded.json**：
  - full_path_exact=0 normalized_path=0 leaf_exact=0 leaf_normalized=0 leaf_ancestor=0 ambiguous=0 unmatched=64
  - unmatched leaf 示例：['厚板组板设计', '水质监测管理', '薄板合同产线专业化设计', '非常规委托检测管理']
- **standard:shanghai_fta_dict.json**：
  - full_path_exact=0 normalized_path=0 leaf_exact=0 leaf_normalized=0 leaf_ancestor=0 ambiguous=0 unmatched=64
  - unmatched leaf 示例：['厚板组板设计', '水质监测管理', '薄板合同产线专业化设计', '非常规委托检测管理']

### dataset `pers_info` 对齐明细

- **corpus:finance**：
  - full_path_exact=0 normalized_path=0 leaf_exact=0 leaf_normalized=0 leaf_ancestor=0 ambiguous=0 unmatched=176
  - unmatched leaf 示例：['人力资源数据', '任课信息', '基本信息年级信息和班级信息', '学历学位信息', '学校概况基本信息', '学生个人基本信息', '学生敏感个人信息', '学科类科研管理', '学籍基本数据', '学籍管理信息', '教职工个人基本信息', '毕业就业信息', '离退休信息', '系统数据', '考核信息']
- **standard:beijing_fta_dict.json**：
  - full_path_exact=0 normalized_path=0 leaf_exact=0 leaf_normalized=0 leaf_ancestor=0 ambiguous=0 unmatched=176
  - unmatched leaf 示例：['人力资源数据', '任课信息', '基本信息年级信息和班级信息', '学历学位信息', '学校概况基本信息', '学生个人基本信息', '学生敏感个人信息', '学科类科研管理', '学籍基本数据', '学籍管理信息', '教职工个人基本信息', '毕业就业信息', '离退休信息', '系统数据', '考核信息']
- **standard:education_dict.json**：
  - full_path_exact=0 normalized_path=0 leaf_exact=35 leaf_normalized=35 leaf_ancestor=35 ambiguous=0 unmatched=141
  - unmatched leaf 示例：['人力资源数据', '基本信息年级信息和班级信息', '学校概况基本信息', '学生个人基本信息', '学生敏感个人信息', '学科类科研管理', '学籍基本数据', '学籍管理信息', '教职工个人基本信息', '毕业就业信息', '系统数据', '考试成绩', '职称信息', '课程信息']
- **standard:financial_standards_dict.json**：
  - full_path_exact=0 normalized_path=0 leaf_exact=0 leaf_normalized=0 leaf_ancestor=0 ambiguous=0 unmatched=176
  - unmatched leaf 示例：['人力资源数据', '任课信息', '基本信息年级信息和班级信息', '学历学位信息', '学校概况基本信息', '学生个人基本信息', '学生敏感个人信息', '学科类科研管理', '学籍基本数据', '学籍管理信息', '教职工个人基本信息', '毕业就业信息', '离退休信息', '系统数据', '考核信息']
- **standard:general_personal_info_dict.json**：
  - full_path_exact=0 normalized_path=0 leaf_exact=0 leaf_normalized=0 leaf_ancestor=0 ambiguous=0 unmatched=176
  - unmatched leaf 示例：['人力资源数据', '任课信息', '基本信息年级信息和班级信息', '学历学位信息', '学校概况基本信息', '学生个人基本信息', '学生敏感个人信息', '学科类科研管理', '学籍基本数据', '学籍管理信息', '教职工个人基本信息', '毕业就业信息', '离退休信息', '系统数据', '考核信息']
- **standard:general_personal_info_expanded.json**：
  - full_path_exact=0 normalized_path=0 leaf_exact=0 leaf_normalized=0 leaf_ancestor=0 ambiguous=0 unmatched=176
  - unmatched leaf 示例：['人力资源数据', '任课信息', '基本信息年级信息和班级信息', '学历学位信息', '学校概况基本信息', '学生个人基本信息', '学生敏感个人信息', '学科类科研管理', '学籍基本数据', '学籍管理信息', '教职工个人基本信息', '毕业就业信息', '离退休信息', '系统数据', '考核信息']
- **standard:guanji_dict.json**：
  - full_path_exact=0 normalized_path=0 leaf_exact=0 leaf_normalized=0 leaf_ancestor=0 ambiguous=0 unmatched=176
  - unmatched leaf 示例：['人力资源数据', '任课信息', '基本信息年级信息和班级信息', '学历学位信息', '学校概况基本信息', '学生个人基本信息', '学生敏感个人信息', '学科类科研管理', '学籍基本数据', '学籍管理信息', '教职工个人基本信息', '毕业就业信息', '离退休信息', '系统数据', '考核信息']
- **standard:iov_standards_dict.json**：
  - full_path_exact=0 normalized_path=0 leaf_exact=0 leaf_normalized=0 leaf_ancestor=0 ambiguous=0 unmatched=176
  - unmatched leaf 示例：['人力资源数据', '任课信息', '基本信息年级信息和班级信息', '学历学位信息', '学校概况基本信息', '学生个人基本信息', '学生敏感个人信息', '学科类科研管理', '学籍基本数据', '学籍管理信息', '教职工个人基本信息', '毕业就业信息', '离退休信息', '系统数据', '考核信息']
- **standard:iov_white_papers_dict.json**：
  - full_path_exact=0 normalized_path=0 leaf_exact=0 leaf_normalized=0 leaf_ancestor=0 ambiguous=0 unmatched=176
  - unmatched leaf 示例：['人力资源数据', '任课信息', '基本信息年级信息和班级信息', '学历学位信息', '学校概况基本信息', '学生个人基本信息', '学生敏感个人信息', '学科类科研管理', '学籍基本数据', '学籍管理信息', '教职工个人基本信息', '毕业就业信息', '离退休信息', '系统数据', '考核信息']
- **standard:iov_white_papers_dict_before.json**：
  - full_path_exact=0 normalized_path=0 leaf_exact=0 leaf_normalized=0 leaf_ancestor=0 ambiguous=0 unmatched=176
  - unmatched leaf 示例：['人力资源数据', '任课信息', '基本信息年级信息和班级信息', '学历学位信息', '学校概况基本信息', '学生个人基本信息', '学生敏感个人信息', '学科类科研管理', '学籍基本数据', '学籍管理信息', '教职工个人基本信息', '毕业就业信息', '离退休信息', '系统数据', '考核信息']
- **standard:sensitive_personal_info_dict.json**：
  - full_path_exact=0 normalized_path=0 leaf_exact=0 leaf_normalized=0 leaf_ancestor=0 ambiguous=0 unmatched=176
  - unmatched leaf 示例：['人力资源数据', '任课信息', '基本信息年级信息和班级信息', '学历学位信息', '学校概况基本信息', '学生个人基本信息', '学生敏感个人信息', '学科类科研管理', '学籍基本数据', '学籍管理信息', '教职工个人基本信息', '毕业就业信息', '离退休信息', '系统数据', '考核信息']
- **standard:sensitive_personal_info_expanded.json**：
  - full_path_exact=0 normalized_path=0 leaf_exact=0 leaf_normalized=0 leaf_ancestor=0 ambiguous=0 unmatched=176
  - unmatched leaf 示例：['人力资源数据', '任课信息', '基本信息年级信息和班级信息', '学历学位信息', '学校概况基本信息', '学生个人基本信息', '学生敏感个人信息', '学科类科研管理', '学籍基本数据', '学籍管理信息', '教职工个人基本信息', '毕业就业信息', '离退休信息', '系统数据', '考核信息']
- **standard:shanghai_fta_dict.json**：
  - full_path_exact=0 normalized_path=0 leaf_exact=0 leaf_normalized=0 leaf_ancestor=0 ambiguous=0 unmatched=176
  - unmatched leaf 示例：['人力资源数据', '任课信息', '基本信息年级信息和班级信息', '学历学位信息', '学校概况基本信息', '学生个人基本信息', '学生敏感个人信息', '学科类科研管理', '学籍基本数据', '学籍管理信息', '教职工个人基本信息', '毕业就业信息', '离退休信息', '系统数据', '考核信息']

### dataset `shougang` 对齐明细

- **corpus:finance**：
  - full_path_exact=0 normalized_path=0 leaf_exact=0 leaf_normalized=0 leaf_ancestor=0 ambiguous=0 unmatched=19415
  - unmatched leaf 示例：['MCCR生产实绩', 'MCCR计划', '——', '业务流程改善申报管理', '业务流程改善积分管理', '业务流程改善评分管理', '业务流程问题管理', '中厚板出入库管理', '中厚板库内管理', '中厚板生产实绩', '临时车辆管理', '产品规范管理', '产线生产物料', '产量指标计划', '人员统计']
- **standard:beijing_fta_dict.json**：
  - full_path_exact=0 normalized_path=0 leaf_exact=0 leaf_normalized=0 leaf_ancestor=0 ambiguous=0 unmatched=19415
  - unmatched leaf 示例：['MCCR生产实绩', 'MCCR计划', '——', '业务流程改善申报管理', '业务流程改善积分管理', '业务流程改善评分管理', '业务流程问题管理', '中厚板出入库管理', '中厚板库内管理', '中厚板生产实绩', '临时车辆管理', '产品规范管理', '产线生产物料', '产量指标计划', '人员统计']
- **standard:education_dict.json**：
  - full_path_exact=0 normalized_path=0 leaf_exact=0 leaf_normalized=0 leaf_ancestor=0 ambiguous=0 unmatched=19415
  - unmatched leaf 示例：['MCCR生产实绩', 'MCCR计划', '——', '业务流程改善申报管理', '业务流程改善积分管理', '业务流程改善评分管理', '业务流程问题管理', '中厚板出入库管理', '中厚板库内管理', '中厚板生产实绩', '临时车辆管理', '产品规范管理', '产线生产物料', '产量指标计划', '人员统计']
- **standard:financial_standards_dict.json**：
  - full_path_exact=0 normalized_path=0 leaf_exact=0 leaf_normalized=0 leaf_ancestor=0 ambiguous=0 unmatched=19415
  - unmatched leaf 示例：['MCCR生产实绩', 'MCCR计划', '——', '业务流程改善申报管理', '业务流程改善积分管理', '业务流程改善评分管理', '业务流程问题管理', '中厚板出入库管理', '中厚板库内管理', '中厚板生产实绩', '临时车辆管理', '产品规范管理', '产线生产物料', '产量指标计划', '人员统计']
- **standard:general_personal_info_dict.json**：
  - full_path_exact=0 normalized_path=0 leaf_exact=0 leaf_normalized=0 leaf_ancestor=0 ambiguous=0 unmatched=19415
  - unmatched leaf 示例：['MCCR生产实绩', 'MCCR计划', '——', '业务流程改善申报管理', '业务流程改善积分管理', '业务流程改善评分管理', '业务流程问题管理', '中厚板出入库管理', '中厚板库内管理', '中厚板生产实绩', '临时车辆管理', '产品规范管理', '产线生产物料', '产量指标计划', '人员统计']
- **standard:general_personal_info_expanded.json**：
  - full_path_exact=0 normalized_path=0 leaf_exact=0 leaf_normalized=0 leaf_ancestor=0 ambiguous=0 unmatched=19415
  - unmatched leaf 示例：['MCCR生产实绩', 'MCCR计划', '——', '业务流程改善申报管理', '业务流程改善积分管理', '业务流程改善评分管理', '业务流程问题管理', '中厚板出入库管理', '中厚板库内管理', '中厚板生产实绩', '临时车辆管理', '产品规范管理', '产线生产物料', '产量指标计划', '人员统计']
- **standard:guanji_dict.json**：
  - full_path_exact=0 normalized_path=0 leaf_exact=18393 leaf_normalized=18393 leaf_ancestor=18393 ambiguous=0 unmatched=1022
  - unmatched leaf 示例：['——']
- **standard:iov_standards_dict.json**：
  - full_path_exact=0 normalized_path=0 leaf_exact=0 leaf_normalized=0 leaf_ancestor=0 ambiguous=0 unmatched=19415
  - unmatched leaf 示例：['MCCR生产实绩', 'MCCR计划', '——', '业务流程改善申报管理', '业务流程改善积分管理', '业务流程改善评分管理', '业务流程问题管理', '中厚板出入库管理', '中厚板库内管理', '中厚板生产实绩', '临时车辆管理', '产品规范管理', '产线生产物料', '产量指标计划', '人员统计']
- **standard:iov_white_papers_dict.json**：
  - full_path_exact=0 normalized_path=0 leaf_exact=0 leaf_normalized=0 leaf_ancestor=0 ambiguous=0 unmatched=19415
  - unmatched leaf 示例：['MCCR生产实绩', 'MCCR计划', '——', '业务流程改善申报管理', '业务流程改善积分管理', '业务流程改善评分管理', '业务流程问题管理', '中厚板出入库管理', '中厚板库内管理', '中厚板生产实绩', '临时车辆管理', '产品规范管理', '产线生产物料', '产量指标计划', '人员统计']
- **standard:iov_white_papers_dict_before.json**：
  - full_path_exact=0 normalized_path=0 leaf_exact=0 leaf_normalized=0 leaf_ancestor=0 ambiguous=0 unmatched=19415
  - unmatched leaf 示例：['MCCR生产实绩', 'MCCR计划', '——', '业务流程改善申报管理', '业务流程改善积分管理', '业务流程改善评分管理', '业务流程问题管理', '中厚板出入库管理', '中厚板库内管理', '中厚板生产实绩', '临时车辆管理', '产品规范管理', '产线生产物料', '产量指标计划', '人员统计']
- **standard:sensitive_personal_info_dict.json**：
  - full_path_exact=0 normalized_path=0 leaf_exact=0 leaf_normalized=0 leaf_ancestor=0 ambiguous=0 unmatched=19415
  - unmatched leaf 示例：['MCCR生产实绩', 'MCCR计划', '——', '业务流程改善申报管理', '业务流程改善积分管理', '业务流程改善评分管理', '业务流程问题管理', '中厚板出入库管理', '中厚板库内管理', '中厚板生产实绩', '临时车辆管理', '产品规范管理', '产线生产物料', '产量指标计划', '人员统计']
- **standard:sensitive_personal_info_expanded.json**：
  - full_path_exact=0 normalized_path=0 leaf_exact=0 leaf_normalized=0 leaf_ancestor=0 ambiguous=0 unmatched=19415
  - unmatched leaf 示例：['MCCR生产实绩', 'MCCR计划', '——', '业务流程改善申报管理', '业务流程改善积分管理', '业务流程改善评分管理', '业务流程问题管理', '中厚板出入库管理', '中厚板库内管理', '中厚板生产实绩', '临时车辆管理', '产品规范管理', '产线生产物料', '产量指标计划', '人员统计']
- **standard:shanghai_fta_dict.json**：
  - full_path_exact=0 normalized_path=0 leaf_exact=66 leaf_normalized=66 leaf_ancestor=66 ambiguous=0 unmatched=19349
  - unmatched leaf 示例：['MCCR生产实绩', 'MCCR计划', '——', '业务流程改善申报管理', '业务流程改善积分管理', '业务流程改善评分管理', '业务流程问题管理', '中厚板出入库管理', '中厚板库内管理', '中厚板生产实绩', '临时车辆管理', '产品规范管理', '产线生产物料', '产量指标计划', '人员统计']

## 4. 结论

### 4.1 实际有哪些 dataset

`finance`, `infra`, `pers_info`, `shougang`（以存在 `all.json` 为准）

### 4.2 每个 dataset 对应哪个 corpus

- `finance` → `corpus:finance`（leaf 匹配 20/25）
- `infra` → `standard:guanji_dict.json`（leaf 匹配 4/4）
- `pers_info` → `standard:education_dict.json`（leaf 匹配 4/18）（部分覆盖：14/18 leaf 无任何语料定义，见 4.9 UNKNOWN）
- `shougang` → `standard:guanji_dict.json`（leaf 匹配 192/193）

### 4.3 每个 dataset 的分类深度

- `finance`：最深级别 {'level_4': 568}；完整路径样本 381；填充模式 level_1/level_2/level_3/level_4×381；level_1/level_2/level_4×187
- `infra`：最深级别 {'level_4': 64}；完整路径样本 64；填充模式 level_1/level_2/level_3/level_4×64
- `pers_info`：最深级别 {'level_4': 176}；完整路径样本 0；填充模式 level_4×176
- `shougang`：最深级别 {'level_4': 19415}；完整路径样本 19415；填充模式 level_1/level_2/level_3/level_4×19415

### 4.4 推荐 leaf level

- `finance` → `level_4`：语料 (corpus:finance / financial_standards) 的 category 对应 level_4 名；数据集 25 个 leaf，语料 220 个 leaf（冷启动：语料 ≫ 标注）
- `infra` → `level_4`：guanji_dict leaf 与 level_4 一一对应（4/4）
- `pers_info` → `level_4`：唯一被填充的级别（单级分类，深度 1）
- `shougang` → `level_4`：guanji_dict leaf 与 level_4 一一对应（192/193，排除 '——' 占位符）

### 4.5 匹配覆盖率

- `finance`：
  - corpus:finance: leaf_normalized 94.0%, leaf_exact 94.0%
  - standard:beijing_fta_dict.json: leaf_normalized 0.0%, leaf_exact 0.0%
  - standard:education_dict.json: leaf_normalized 32.2%, leaf_exact 32.2%
  - standard:financial_standards_dict.json: leaf_normalized 94.0%, leaf_exact 94.0%
  - standard:general_personal_info_dict.json: leaf_normalized 0.0%, leaf_exact 0.0%
  - standard:general_personal_info_expanded.json: leaf_normalized 0.0%, leaf_exact 0.0%
  - standard:guanji_dict.json: leaf_normalized 0.0%, leaf_exact 0.0%
  - standard:iov_standards_dict.json: leaf_normalized 0.0%, leaf_exact 0.0%
  - standard:iov_white_papers_dict.json: leaf_normalized 0.0%, leaf_exact 0.0%
  - standard:iov_white_papers_dict_before.json: leaf_normalized 0.0%, leaf_exact 0.0%
  - standard:sensitive_personal_info_dict.json: leaf_normalized 0.0%, leaf_exact 0.0%
  - standard:sensitive_personal_info_expanded.json: leaf_normalized 0.0%, leaf_exact 0.0%
  - standard:shanghai_fta_dict.json: leaf_normalized 0.0%, leaf_exact 0.0%
- `infra`：
  - corpus:finance: leaf_normalized 0.0%, leaf_exact 0.0%
  - standard:beijing_fta_dict.json: leaf_normalized 0.0%, leaf_exact 0.0%
  - standard:education_dict.json: leaf_normalized 0.0%, leaf_exact 0.0%
  - standard:financial_standards_dict.json: leaf_normalized 0.0%, leaf_exact 0.0%
  - standard:general_personal_info_dict.json: leaf_normalized 0.0%, leaf_exact 0.0%
  - standard:general_personal_info_expanded.json: leaf_normalized 0.0%, leaf_exact 0.0%
  - standard:guanji_dict.json: leaf_normalized 100.0%, leaf_exact 100.0%
  - standard:iov_standards_dict.json: leaf_normalized 0.0%, leaf_exact 0.0%
  - standard:iov_white_papers_dict.json: leaf_normalized 0.0%, leaf_exact 0.0%
  - standard:iov_white_papers_dict_before.json: leaf_normalized 0.0%, leaf_exact 0.0%
  - standard:sensitive_personal_info_dict.json: leaf_normalized 0.0%, leaf_exact 0.0%
  - standard:sensitive_personal_info_expanded.json: leaf_normalized 0.0%, leaf_exact 0.0%
  - standard:shanghai_fta_dict.json: leaf_normalized 0.0%, leaf_exact 0.0%
- `pers_info`：
  - corpus:finance: leaf_normalized 0.0%, leaf_exact 0.0%
  - standard:beijing_fta_dict.json: leaf_normalized 0.0%, leaf_exact 0.0%
  - standard:education_dict.json: leaf_normalized 19.9%, leaf_exact 19.9%
  - standard:financial_standards_dict.json: leaf_normalized 0.0%, leaf_exact 0.0%
  - standard:general_personal_info_dict.json: leaf_normalized 0.0%, leaf_exact 0.0%
  - standard:general_personal_info_expanded.json: leaf_normalized 0.0%, leaf_exact 0.0%
  - standard:guanji_dict.json: leaf_normalized 0.0%, leaf_exact 0.0%
  - standard:iov_standards_dict.json: leaf_normalized 0.0%, leaf_exact 0.0%
  - standard:iov_white_papers_dict.json: leaf_normalized 0.0%, leaf_exact 0.0%
  - standard:iov_white_papers_dict_before.json: leaf_normalized 0.0%, leaf_exact 0.0%
  - standard:sensitive_personal_info_dict.json: leaf_normalized 0.0%, leaf_exact 0.0%
  - standard:sensitive_personal_info_expanded.json: leaf_normalized 0.0%, leaf_exact 0.0%
  - standard:shanghai_fta_dict.json: leaf_normalized 0.0%, leaf_exact 0.0%
- `shougang`：
  - corpus:finance: leaf_normalized 0.0%, leaf_exact 0.0%
  - standard:beijing_fta_dict.json: leaf_normalized 0.0%, leaf_exact 0.0%
  - standard:education_dict.json: leaf_normalized 0.0%, leaf_exact 0.0%
  - standard:financial_standards_dict.json: leaf_normalized 0.0%, leaf_exact 0.0%
  - standard:general_personal_info_dict.json: leaf_normalized 0.0%, leaf_exact 0.0%
  - standard:general_personal_info_expanded.json: leaf_normalized 0.0%, leaf_exact 0.0%
  - standard:guanji_dict.json: leaf_normalized 94.7%, leaf_exact 94.7%
  - standard:iov_standards_dict.json: leaf_normalized 0.0%, leaf_exact 0.0%
  - standard:iov_white_papers_dict.json: leaf_normalized 0.0%, leaf_exact 0.0%
  - standard:iov_white_papers_dict_before.json: leaf_normalized 0.0%, leaf_exact 0.0%
  - standard:sensitive_personal_info_dict.json: leaf_normalized 0.0%, leaf_exact 0.0%
  - standard:sensitive_personal_info_expanded.json: leaf_normalized 0.0%, leaf_exact 0.0%
  - standard:shanghai_fta_dict.json: leaf_normalized 0.3%, leaf_exact 0.3%

### 4.6 ambiguous / unmatched 数量

- `finance`：
  - corpus:finance: ambiguous 0 leaf, unmatched 5 leaf
    - unmatched 示例：['交易清金额信息', '单位基本信息', '单位基本情况', '单位联系人信息', '基本信息（公开']
  - standard:beijing_fta_dict.json: ambiguous 0 leaf, unmatched 25 leaf
    - unmatched 示例：['个人基本概况信息', '个人联系信息', '交易基本信息', '交易对手信息', '交易清结算信息', '交易清金额信息', '交易记账信息', '交易金额信息', '传统鉴别信息', '信托运用信息', '关系标签信息', '单位基本信息', '单位基本情况', '单位基本概况', '单位联系人信息']
  - standard:education_dict.json: ambiguous 1 leaf, unmatched 24 leaf
    - unmatched 示例：['个人基本概况信息', '个人联系信息', '交易基本信息', '交易对手信息', '交易清结算信息', '交易清金额信息', '交易记账信息', '交易金额信息', '传统鉴别信息', '信托运用信息', '关系标签信息', '单位基本信息', '单位基本情况', '单位基本概况', '单位联系人信息']
  - standard:financial_standards_dict.json: ambiguous 3 leaf, unmatched 5 leaf
    - unmatched 示例：['交易清金额信息', '单位基本信息', '单位基本情况', '单位联系人信息', '基本信息（公开']
  - standard:general_personal_info_dict.json: ambiguous 0 leaf, unmatched 25 leaf
    - unmatched 示例：['个人基本概况信息', '个人联系信息', '交易基本信息', '交易对手信息', '交易清结算信息', '交易清金额信息', '交易记账信息', '交易金额信息', '传统鉴别信息', '信托运用信息', '关系标签信息', '单位基本信息', '单位基本情况', '单位基本概况', '单位联系人信息']
  - standard:general_personal_info_expanded.json: ambiguous 0 leaf, unmatched 25 leaf
    - unmatched 示例：['个人基本概况信息', '个人联系信息', '交易基本信息', '交易对手信息', '交易清结算信息', '交易清金额信息', '交易记账信息', '交易金额信息', '传统鉴别信息', '信托运用信息', '关系标签信息', '单位基本信息', '单位基本情况', '单位基本概况', '单位联系人信息']
  - standard:guanji_dict.json: ambiguous 0 leaf, unmatched 25 leaf
    - unmatched 示例：['个人基本概况信息', '个人联系信息', '交易基本信息', '交易对手信息', '交易清结算信息', '交易清金额信息', '交易记账信息', '交易金额信息', '传统鉴别信息', '信托运用信息', '关系标签信息', '单位基本信息', '单位基本情况', '单位基本概况', '单位联系人信息']
  - standard:iov_standards_dict.json: ambiguous 0 leaf, unmatched 25 leaf
    - unmatched 示例：['个人基本概况信息', '个人联系信息', '交易基本信息', '交易对手信息', '交易清结算信息', '交易清金额信息', '交易记账信息', '交易金额信息', '传统鉴别信息', '信托运用信息', '关系标签信息', '单位基本信息', '单位基本情况', '单位基本概况', '单位联系人信息']
  - standard:iov_white_papers_dict.json: ambiguous 0 leaf, unmatched 25 leaf
    - unmatched 示例：['个人基本概况信息', '个人联系信息', '交易基本信息', '交易对手信息', '交易清结算信息', '交易清金额信息', '交易记账信息', '交易金额信息', '传统鉴别信息', '信托运用信息', '关系标签信息', '单位基本信息', '单位基本情况', '单位基本概况', '单位联系人信息']
  - standard:iov_white_papers_dict_before.json: ambiguous 0 leaf, unmatched 25 leaf
    - unmatched 示例：['个人基本概况信息', '个人联系信息', '交易基本信息', '交易对手信息', '交易清结算信息', '交易清金额信息', '交易记账信息', '交易金额信息', '传统鉴别信息', '信托运用信息', '关系标签信息', '单位基本信息', '单位基本情况', '单位基本概况', '单位联系人信息']
  - standard:sensitive_personal_info_dict.json: ambiguous 0 leaf, unmatched 25 leaf
    - unmatched 示例：['个人基本概况信息', '个人联系信息', '交易基本信息', '交易对手信息', '交易清结算信息', '交易清金额信息', '交易记账信息', '交易金额信息', '传统鉴别信息', '信托运用信息', '关系标签信息', '单位基本信息', '单位基本情况', '单位基本概况', '单位联系人信息']
  - standard:sensitive_personal_info_expanded.json: ambiguous 0 leaf, unmatched 25 leaf
    - unmatched 示例：['个人基本概况信息', '个人联系信息', '交易基本信息', '交易对手信息', '交易清结算信息', '交易清金额信息', '交易记账信息', '交易金额信息', '传统鉴别信息', '信托运用信息', '关系标签信息', '单位基本信息', '单位基本情况', '单位基本概况', '单位联系人信息']
  - standard:shanghai_fta_dict.json: ambiguous 0 leaf, unmatched 25 leaf
    - unmatched 示例：['个人基本概况信息', '个人联系信息', '交易基本信息', '交易对手信息', '交易清结算信息', '交易清金额信息', '交易记账信息', '交易金额信息', '传统鉴别信息', '信托运用信息', '关系标签信息', '单位基本信息', '单位基本情况', '单位基本概况', '单位联系人信息']
- `infra`：
  - corpus:finance: ambiguous 0 leaf, unmatched 4 leaf
    - unmatched 示例：['厚板组板设计', '水质监测管理', '薄板合同产线专业化设计', '非常规委托检测管理']
  - standard:beijing_fta_dict.json: ambiguous 0 leaf, unmatched 4 leaf
    - unmatched 示例：['厚板组板设计', '水质监测管理', '薄板合同产线专业化设计', '非常规委托检测管理']
  - standard:education_dict.json: ambiguous 0 leaf, unmatched 4 leaf
    - unmatched 示例：['厚板组板设计', '水质监测管理', '薄板合同产线专业化设计', '非常规委托检测管理']
  - standard:financial_standards_dict.json: ambiguous 0 leaf, unmatched 4 leaf
    - unmatched 示例：['厚板组板设计', '水质监测管理', '薄板合同产线专业化设计', '非常规委托检测管理']
  - standard:general_personal_info_dict.json: ambiguous 0 leaf, unmatched 4 leaf
    - unmatched 示例：['厚板组板设计', '水质监测管理', '薄板合同产线专业化设计', '非常规委托检测管理']
  - standard:general_personal_info_expanded.json: ambiguous 0 leaf, unmatched 4 leaf
    - unmatched 示例：['厚板组板设计', '水质监测管理', '薄板合同产线专业化设计', '非常规委托检测管理']
  - standard:iov_standards_dict.json: ambiguous 0 leaf, unmatched 4 leaf
    - unmatched 示例：['厚板组板设计', '水质监测管理', '薄板合同产线专业化设计', '非常规委托检测管理']
  - standard:iov_white_papers_dict.json: ambiguous 0 leaf, unmatched 4 leaf
    - unmatched 示例：['厚板组板设计', '水质监测管理', '薄板合同产线专业化设计', '非常规委托检测管理']
  - standard:iov_white_papers_dict_before.json: ambiguous 0 leaf, unmatched 4 leaf
    - unmatched 示例：['厚板组板设计', '水质监测管理', '薄板合同产线专业化设计', '非常规委托检测管理']
  - standard:sensitive_personal_info_dict.json: ambiguous 0 leaf, unmatched 4 leaf
    - unmatched 示例：['厚板组板设计', '水质监测管理', '薄板合同产线专业化设计', '非常规委托检测管理']
  - standard:sensitive_personal_info_expanded.json: ambiguous 0 leaf, unmatched 4 leaf
    - unmatched 示例：['厚板组板设计', '水质监测管理', '薄板合同产线专业化设计', '非常规委托检测管理']
  - standard:shanghai_fta_dict.json: ambiguous 0 leaf, unmatched 4 leaf
    - unmatched 示例：['厚板组板设计', '水质监测管理', '薄板合同产线专业化设计', '非常规委托检测管理']
- `pers_info`：
  - corpus:finance: ambiguous 0 leaf, unmatched 18 leaf
    - unmatched 示例：['人力资源数据', '任课信息', '基本信息年级信息和班级信息', '学历学位信息', '学校概况基本信息', '学生个人基本信息', '学生敏感个人信息', '学科类科研管理', '学籍基本数据', '学籍管理信息', '教职工个人基本信息', '毕业就业信息', '离退休信息', '系统数据', '考核信息']
  - standard:beijing_fta_dict.json: ambiguous 0 leaf, unmatched 18 leaf
    - unmatched 示例：['人力资源数据', '任课信息', '基本信息年级信息和班级信息', '学历学位信息', '学校概况基本信息', '学生个人基本信息', '学生敏感个人信息', '学科类科研管理', '学籍基本数据', '学籍管理信息', '教职工个人基本信息', '毕业就业信息', '离退休信息', '系统数据', '考核信息']
  - standard:education_dict.json: ambiguous 0 leaf, unmatched 14 leaf
    - unmatched 示例：['人力资源数据', '基本信息年级信息和班级信息', '学校概况基本信息', '学生个人基本信息', '学生敏感个人信息', '学科类科研管理', '学籍基本数据', '学籍管理信息', '教职工个人基本信息', '毕业就业信息', '系统数据', '考试成绩', '职称信息', '课程信息']
  - standard:financial_standards_dict.json: ambiguous 0 leaf, unmatched 18 leaf
    - unmatched 示例：['人力资源数据', '任课信息', '基本信息年级信息和班级信息', '学历学位信息', '学校概况基本信息', '学生个人基本信息', '学生敏感个人信息', '学科类科研管理', '学籍基本数据', '学籍管理信息', '教职工个人基本信息', '毕业就业信息', '离退休信息', '系统数据', '考核信息']
  - standard:general_personal_info_dict.json: ambiguous 0 leaf, unmatched 18 leaf
    - unmatched 示例：['人力资源数据', '任课信息', '基本信息年级信息和班级信息', '学历学位信息', '学校概况基本信息', '学生个人基本信息', '学生敏感个人信息', '学科类科研管理', '学籍基本数据', '学籍管理信息', '教职工个人基本信息', '毕业就业信息', '离退休信息', '系统数据', '考核信息']
  - standard:general_personal_info_expanded.json: ambiguous 0 leaf, unmatched 18 leaf
    - unmatched 示例：['人力资源数据', '任课信息', '基本信息年级信息和班级信息', '学历学位信息', '学校概况基本信息', '学生个人基本信息', '学生敏感个人信息', '学科类科研管理', '学籍基本数据', '学籍管理信息', '教职工个人基本信息', '毕业就业信息', '离退休信息', '系统数据', '考核信息']
  - standard:guanji_dict.json: ambiguous 0 leaf, unmatched 18 leaf
    - unmatched 示例：['人力资源数据', '任课信息', '基本信息年级信息和班级信息', '学历学位信息', '学校概况基本信息', '学生个人基本信息', '学生敏感个人信息', '学科类科研管理', '学籍基本数据', '学籍管理信息', '教职工个人基本信息', '毕业就业信息', '离退休信息', '系统数据', '考核信息']
  - standard:iov_standards_dict.json: ambiguous 0 leaf, unmatched 18 leaf
    - unmatched 示例：['人力资源数据', '任课信息', '基本信息年级信息和班级信息', '学历学位信息', '学校概况基本信息', '学生个人基本信息', '学生敏感个人信息', '学科类科研管理', '学籍基本数据', '学籍管理信息', '教职工个人基本信息', '毕业就业信息', '离退休信息', '系统数据', '考核信息']
  - standard:iov_white_papers_dict.json: ambiguous 0 leaf, unmatched 18 leaf
    - unmatched 示例：['人力资源数据', '任课信息', '基本信息年级信息和班级信息', '学历学位信息', '学校概况基本信息', '学生个人基本信息', '学生敏感个人信息', '学科类科研管理', '学籍基本数据', '学籍管理信息', '教职工个人基本信息', '毕业就业信息', '离退休信息', '系统数据', '考核信息']
  - standard:iov_white_papers_dict_before.json: ambiguous 0 leaf, unmatched 18 leaf
    - unmatched 示例：['人力资源数据', '任课信息', '基本信息年级信息和班级信息', '学历学位信息', '学校概况基本信息', '学生个人基本信息', '学生敏感个人信息', '学科类科研管理', '学籍基本数据', '学籍管理信息', '教职工个人基本信息', '毕业就业信息', '离退休信息', '系统数据', '考核信息']
  - standard:sensitive_personal_info_dict.json: ambiguous 0 leaf, unmatched 18 leaf
    - unmatched 示例：['人力资源数据', '任课信息', '基本信息年级信息和班级信息', '学历学位信息', '学校概况基本信息', '学生个人基本信息', '学生敏感个人信息', '学科类科研管理', '学籍基本数据', '学籍管理信息', '教职工个人基本信息', '毕业就业信息', '离退休信息', '系统数据', '考核信息']
  - standard:sensitive_personal_info_expanded.json: ambiguous 0 leaf, unmatched 18 leaf
    - unmatched 示例：['人力资源数据', '任课信息', '基本信息年级信息和班级信息', '学历学位信息', '学校概况基本信息', '学生个人基本信息', '学生敏感个人信息', '学科类科研管理', '学籍基本数据', '学籍管理信息', '教职工个人基本信息', '毕业就业信息', '离退休信息', '系统数据', '考核信息']
  - standard:shanghai_fta_dict.json: ambiguous 0 leaf, unmatched 18 leaf
    - unmatched 示例：['人力资源数据', '任课信息', '基本信息年级信息和班级信息', '学历学位信息', '学校概况基本信息', '学生个人基本信息', '学生敏感个人信息', '学科类科研管理', '学籍基本数据', '学籍管理信息', '教职工个人基本信息', '毕业就业信息', '离退休信息', '系统数据', '考核信息']
- `shougang`：
  - corpus:finance: ambiguous 0 leaf, unmatched 193 leaf
    - unmatched 示例：['MCCR生产实绩', 'MCCR计划', '——', '业务流程改善申报管理', '业务流程改善积分管理', '业务流程改善评分管理', '业务流程问题管理', '中厚板出入库管理', '中厚板库内管理', '中厚板生产实绩', '临时车辆管理', '产品规范管理', '产线生产物料', '产量指标计划', '人员统计']
  - standard:beijing_fta_dict.json: ambiguous 0 leaf, unmatched 193 leaf
    - unmatched 示例：['MCCR生产实绩', 'MCCR计划', '——', '业务流程改善申报管理', '业务流程改善积分管理', '业务流程改善评分管理', '业务流程问题管理', '中厚板出入库管理', '中厚板库内管理', '中厚板生产实绩', '临时车辆管理', '产品规范管理', '产线生产物料', '产量指标计划', '人员统计']
  - standard:education_dict.json: ambiguous 0 leaf, unmatched 193 leaf
    - unmatched 示例：['MCCR生产实绩', 'MCCR计划', '——', '业务流程改善申报管理', '业务流程改善积分管理', '业务流程改善评分管理', '业务流程问题管理', '中厚板出入库管理', '中厚板库内管理', '中厚板生产实绩', '临时车辆管理', '产品规范管理', '产线生产物料', '产量指标计划', '人员统计']
  - standard:financial_standards_dict.json: ambiguous 0 leaf, unmatched 193 leaf
    - unmatched 示例：['MCCR生产实绩', 'MCCR计划', '——', '业务流程改善申报管理', '业务流程改善积分管理', '业务流程改善评分管理', '业务流程问题管理', '中厚板出入库管理', '中厚板库内管理', '中厚板生产实绩', '临时车辆管理', '产品规范管理', '产线生产物料', '产量指标计划', '人员统计']
  - standard:general_personal_info_dict.json: ambiguous 0 leaf, unmatched 193 leaf
    - unmatched 示例：['MCCR生产实绩', 'MCCR计划', '——', '业务流程改善申报管理', '业务流程改善积分管理', '业务流程改善评分管理', '业务流程问题管理', '中厚板出入库管理', '中厚板库内管理', '中厚板生产实绩', '临时车辆管理', '产品规范管理', '产线生产物料', '产量指标计划', '人员统计']
  - standard:general_personal_info_expanded.json: ambiguous 0 leaf, unmatched 193 leaf
    - unmatched 示例：['MCCR生产实绩', 'MCCR计划', '——', '业务流程改善申报管理', '业务流程改善积分管理', '业务流程改善评分管理', '业务流程问题管理', '中厚板出入库管理', '中厚板库内管理', '中厚板生产实绩', '临时车辆管理', '产品规范管理', '产线生产物料', '产量指标计划', '人员统计']
  - standard:guanji_dict.json: ambiguous 0 leaf, unmatched 1 leaf
    - unmatched 示例：['——']
  - standard:iov_standards_dict.json: ambiguous 0 leaf, unmatched 193 leaf
    - unmatched 示例：['MCCR生产实绩', 'MCCR计划', '——', '业务流程改善申报管理', '业务流程改善积分管理', '业务流程改善评分管理', '业务流程问题管理', '中厚板出入库管理', '中厚板库内管理', '中厚板生产实绩', '临时车辆管理', '产品规范管理', '产线生产物料', '产量指标计划', '人员统计']
  - standard:iov_white_papers_dict.json: ambiguous 0 leaf, unmatched 193 leaf
    - unmatched 示例：['MCCR生产实绩', 'MCCR计划', '——', '业务流程改善申报管理', '业务流程改善积分管理', '业务流程改善评分管理', '业务流程问题管理', '中厚板出入库管理', '中厚板库内管理', '中厚板生产实绩', '临时车辆管理', '产品规范管理', '产线生产物料', '产量指标计划', '人员统计']
  - standard:iov_white_papers_dict_before.json: ambiguous 0 leaf, unmatched 193 leaf
    - unmatched 示例：['MCCR生产实绩', 'MCCR计划', '——', '业务流程改善申报管理', '业务流程改善积分管理', '业务流程改善评分管理', '业务流程问题管理', '中厚板出入库管理', '中厚板库内管理', '中厚板生产实绩', '临时车辆管理', '产品规范管理', '产线生产物料', '产量指标计划', '人员统计']
  - standard:sensitive_personal_info_dict.json: ambiguous 0 leaf, unmatched 193 leaf
    - unmatched 示例：['MCCR生产实绩', 'MCCR计划', '——', '业务流程改善申报管理', '业务流程改善积分管理', '业务流程改善评分管理', '业务流程问题管理', '中厚板出入库管理', '中厚板库内管理', '中厚板生产实绩', '临时车辆管理', '产品规范管理', '产线生产物料', '产量指标计划', '人员统计']
  - standard:sensitive_personal_info_expanded.json: ambiguous 0 leaf, unmatched 193 leaf
    - unmatched 示例：['MCCR生产实绩', 'MCCR计划', '——', '业务流程改善申报管理', '业务流程改善积分管理', '业务流程改善评分管理', '业务流程问题管理', '中厚板出入库管理', '中厚板库内管理', '中厚板生产实绩', '临时车辆管理', '产品规范管理', '产线生产物料', '产量指标计划', '人员统计']
  - standard:shanghai_fta_dict.json: ambiguous 0 leaf, unmatched 192 leaf
    - unmatched 示例：['MCCR生产实绩', 'MCCR计划', '——', '业务流程改善申报管理', '业务流程改善积分管理', '业务流程改善评分管理', '业务流程问题管理', '中厚板出入库管理', '中厚板库内管理', '中厚板生产实绩', '临时车辆管理', '产品规范管理', '产线生产物料', '产量指标计划', '人员统计']

### 4.7 leaf name collision

- `finance`：{'level_2': 1, 'level_3': 1, 'level_4': 3}
- `infra`：无
- `pers_info`：无
- `shougang`：{'level_4': 1}

### 4.8 当前 schema 存在的问题

- **[high] finance**：level_3 is empty on 187 of 568 samples while level_4 is fully populated; 3-level paths are incomplete
- **[medium] finance/level_4**：truncated label '基本信息（公开' with unbalanced parenthesis
- **[medium] finance/level_4**：'交易清金额信息' vs '交易清结算信息': likely typo, needs human confirmation
- **[high] corpus:finance (data/finance/corpus.json)**：leaf-only corpus: category identity is the bare level_4 name; no path/code is kept, so any future leaf-name collision is unresolvable
- **[medium] corpus:finance**：6 labels contain stray internal spaces vs financial_standards (e.g. '其他类中间业务 信息'); exact-match coverage drops without whitespace normalization
- **[medium] standard:financial_standards_dict.json**：duplicate full categories 1 kinds / 4 extra instances; e.g. '业务-合约协议-基本信息' x5
- **[medium] standard:financial_standards_dict.json**：same leaf name under different parents (12 leaf kinds map to multiple full categories); leaf-only matching is ambiguous
- **[high] standard:financial_standards_dict.json vs finance dataset**：层级体系不一致：fs 全部 237 条为 3 段路径（L1-L2-leaf），而 finance 数据集为 4 级（L1-L2-L3-leaf）；fs 不携带 level_3。另有 7 条以 '监管' 为 L1 的类目（监管-数据报送/数据收取-*）不在 finance 数据集的任何 level 词表中（数据集 L1 仅 业务/客户/经营管理）
- **[info] corpus:finance**：corpus 的 '重复 category' 是同一 level_4 下的多条文档（12 个 label 共 29 条文档），不是 category 本身重复
- **[high] cross-corpus leaf collision**：shougang leaf '供应商管理数据'（66 样本）同时出现在 guanji_dict 与 shanghai_fta_dict 中；finance leaf '基本信息' 同时出现在 financial_standards （3 个父路径）与 education_dict（3 个 code）中 —— 跨语料同名 leaf 无法仅靠 leaf 字符串区分
- **[high] shougang/level_4**：'——' placeholder used as a label on 1022 samples (5.3%); it is not a real category and collides across many parent paths
- **[medium] data_level field (all datasets)**：data_level distribution does not match classification depth; semantics unknown
- **[high] pers_info**：single-level classification: only level_4 is populated (18 labels); no path exists
- **[high] pers_info alignment**：14/18 leaf labels match no standard in data/knowledge/standards_map/
- **[info] standard:guanji_dict.json**：codes A1-1-1 / B1-2: letter prefix maps 1:1 to level_1 (A=研发数据域, B=生产数据域, C=管理数据域, 192/192 proven); digit groups are ordinals with variable depth (2 or 3 groups) and cannot be mapped to level_2/level_3 from data
- **[medium] standard:iov_standards_dict.json**：category 名严重重复：33 条目仅 11 个唯一 name (22 个重复实例)，同一 name 对应多条 不同 class/ref 条目 —— leaf 名无法作为唯一标识
- **[medium] general_personal_info_expanded.json**：expanded 变体丢失 category 名称：value 只保留描述性文本 （原始 *_dict 的 category name 不在其中），无法与数据集 leaf 对齐
- **[medium] sensitive_personal_info_expanded.json**：expanded 变体丢失 category 名称：value 只保留描述性文本 （原始 *_dict 的 category name 不在其中），无法与数据集 leaf 对齐

### 4.9 UNKNOWN / NEEDS HUMAN CONFIRMATION

- UNKNOWN：finance level_4 '交易清金额信息' is likely a typo for '交易清结算信息', but both exist as distinct labels; correction needs human confirmation.
- UNKNOWN：finance level_4 '单位基本信息' / '单位基本情况' / '单位联系人信息' 在两个语料中均无定义 （corpus:finance 只有 '单位基本概况'）；是否为同一类别的命名不一致需要人工确认。
- UNKNOWN：data_level field semantics are inconsistent with classification depth (e.g. finance has L2 on 460/568 samples although every sample fills 4 levels); meaning of data_level needs human confirmation.
- UNKNOWN：pers_info has no corpus in the repo covering 14/18 leaf labels; the intended standard document is unknown (education_dict covers only 4).
- UNKNOWN：guanji code digit groups (e.g. '1-1' in A1-1-1, '2' in B1-2) cannot be mapped to level_2/level_3 names from data alone; the original guide is needed to confirm. Letter prefix maps to level_1 with 100% consistency (A=研发数据域, B=生产数据域, C=管理数据域, proven on 192/192 leaves).
- UNKNOWN：shougang '——' placeholder semantics (unclassified vs not-applicable) needs human confirmation.
- UNKNOWN：finance level_4 '基本信息（公开' is a truncated label with an unbalanced parenthesis; expected full form needs the source file.
- UNKNOWN：financial_standards duplicate full category '业务-合约协议-基本信息' x5: whether this is intentional (multiple class/ref rows) needs the source document.
- UNKNOWN：education_dict codes (A1-1 / A3-3 / A4-1 ...) 与 pers_info 无层级可对照 （pers_info 只有单级分类），code 到分类的映射无法从数据证明。
- UNKNOWN：finance↔education_dict 的 32.2% 覆盖全部来自 '基本信息' 一个 leaf 的跨域撞名，不代表 pers_info/education 是 finance 的语料；是否移除该误匹配需要人工确认。

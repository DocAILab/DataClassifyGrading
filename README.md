# DataClassifyGrading — 数据分类分级智能体

基于大模型对数据资产元数据做**四级分类（level_1..4）+ 敏感级别（L1–L4）标注**，面向后训练（RFT / RL）的智能体项目。

## 快速开始

### 1. 数据下载

数据来自 ModelScope：

```bash
export MODELSCOPE_TOKEN=xxx
git clone <ModelScope 数据集地址> data/   # 各领域 all/train/val/test.json + split_report.json
```

### 2. 数据管线（独立维护）

```bash
# 预处理 + 按表分组切分（0.8/0.1/0.1）
python -m script.preprocessing.cli prepare --dataset finance

# 指南语料 → 检索语料
python -m script.corpus.cli build --dataset finance --overwrite
```

详细 SOP 见 [docs/新数据集运行说明.md](docs/新数据集运行说明.md)。训练框架只消费
规范化的 `train/val/test.json`，不导入或改写预处理实现。

### 3. 知识库

`data/knowledge/standards_map/`：12 份多领域分类分级标准词典（车联网 YDT 3751 / 白皮书、关基、教育、金融、自贸区京沪、个人信息一般/敏感）

### 4. 可复用 VeRL 后训练框架

仓库保存算法无关的任务契约与评价逻辑，以及 VeRL 数据适配、验证和启动脚本；
不复制 VeRL 源码。当前已实现 SFT vertical slice，RL adapter 在 reward 与正式数据
契约冻结后再加入。
离线测试：

```bash
pip install -e ".[test]"
pytest -q
```

数据契约、导出命令和服务器 smoke-test 入口见
[`docs/SFT_BASELINE.md`](docs/SFT_BASELINE.md)；公共 seam 与 RL 接入条件见
[`docs/RL_FRAMEWORK.md`](docs/RL_FRAMEWORK.md)。

## 文档与产物结构

`docs/` 存放项目文档，`artifacts/generated/` 存放可由脚本重新生成的分析产物。

```text
docs/                  文档：设计说明、分析结论、阶段报告
├── design/            设计文档（data_contract / prompt_interface）
├── phase_reports/     阶段报告
├── data_alignment.md
├── prompt_length_analysis.md
├── SFT_BASELINE.md / RL_FRAMEWORK.md / 新数据集运行说明.md

artifacts/generated/   生成产物：统计结果、对齐报告等可复现输出
├── alignment/         对齐报告（data_alignment_report.json/.md）
└── prompt_stats/      统计产物
    ├── canonical_id_baseline/   旧 canonical-id 统计（baseline）
    └── choice_id/               choice-protocol 新统计
```

关键文档：

| 文档 | 内容 |
|---|---|
| [`docs/design/data_contract.md`](docs/design/data_contract.md) | canonical category_id 数据契约（registry/corpus/target、身份策略、训练入口规则） |
| [`docs/design/prompt_interface.md`](docs/design/prompt_interface.md) | prompt-facing choice protocol（choice id ↔ canonical id、共享解码层） |
| [`docs/prompt_length_analysis.md`](docs/prompt_length_analysis.md) | prompt token 分析（old∽new、Stage1 收益、worst-case 2689/32768） |
| [`docs/data_alignment.md`](docs/data_alignment.md) | dataset ↔ corpus/standard 对齐人工结论 |

`docs/reports/` 为本地 gitignored 调研/smoke scratch，不入库。

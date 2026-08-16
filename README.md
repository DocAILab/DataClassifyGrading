# DataClassifyGrading — 数据分类分级智能体

基于大模型对数据资产元数据做**四级分类（level_1..4）+ 敏感级别（L1–L4）标注**，面向后训练（RFT / RL）的智能体项目。

## 快速开始

### 1. 数据下载

数据来自 ModelScope：

```bash
export MODELSCOPE_TOKEN=xxx
git clone <ModelScope 数据集地址> data/   # 各领域 all/train/val/test.json + split_report.json
```

### 2. 数据管线（旧仓库复用）

```bash
# 预处理 + 按表分组切分（0.8/0.1/0.1）
python -m script.preprocessing.cli prepare --dataset finance

# 指南语料 → 检索语料
python -m script.corpus.cli build --dataset finance --overwrite
```

详细 SOP 见 [docs/新数据集运行说明.md](docs/新数据集运行说明.md)。

### 3. 知识库

`data/knowledge/standards_map/`：12 份多领域分类分级标准词典（车联网 YDT 3751 / 白皮书、关基、教育、金融、自贸区京沪、个人信息一般/敏感）

### 4. VeRL SFT 基座

仓库只保存 VeRL SFT 的数据适配、验证和启动脚本，不复制 VeRL 源码。
离线测试：

```bash
pip install -e ".[test]"
pytest -q
```

数据契约、导出命令和服务器 smoke-test 入口见
[`docs/SFT_BASELINE.md`](docs/SFT_BASELINE.md)。

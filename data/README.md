# data/ — 数据目录

## 层（语义）

```text
raw
  ↓ preprocessing（学长）
processed
  ↓ canonical resolution（script/canonical，冻结契约）
canonical
  ↓ SFT/RL export（script/verl/sft/export、script/verl/rl/export）
parquet（data/sft、data/rl）
```

- `raw/`：source of truth 的原始导入物（CSV/XLSX / 原始包）。**训练脚本不得直接修改**。
  当前仓库只保留占位（`data/raw/.gitkeep`）；真原始表格由学长侧
  `data/raw/<dataset>.{csv,xlsx}` 提供（preprocessing `prepare` 默认查找该路径）。
- `processed/<dataset>/`：预处理结果（`all.json` + `train.json` / `val.json` /
  `test.json` + `split_report.json`）。split 按 record id 隔离，是后续 canonical 的
  输入层，也是 `data/<dataset>` 归一化源层的正式位置。
- `canonical/<dataset>/`：canonical 契约层（`all.json` + `resolution_report.json`）。
  **后续所有算法（SFT/RL/评估）统一消费该层**；`data_level` 仅作 provenance。
- `sft/`、`rl/`：可再生训练派生产物（`*.parquet` + export_report.json），gitignored；
  源码变动后可删除并重新 export。
- `knowledge/standards_map/`：标准知识（12 份词典，入库），不属于训练数据生成产物。

## 下载

数据来自 ModelScope

```bash
export MODELSCOPE_TOKEN=xxx     # token 走环境变量，禁止写入脚本/文档/仓库
git clone <ModelScope 数据集地址> .   # 导入到 raw/（原始 CSV/XLSX）
```

## 各领域概况（2025-08-10 快照）

| 领域 | 业务域 | 记录数(processed all.json) | train/val/test |
|---|---|---|---|
| finance | 信托核心系统 | 568 | 439/69/58 |
| infra | 钢铁基建（=shougang 子集） | 64 | 40/15/9 |
| pers_info | 高校个人信息 | 176 | 140/18/18 |
| shougang | 首钢生产/管理/研发 | 19,415 | 15,347/2,028/2,040 |

## 入库内容

- `knowledge/standards_map/`：12 份多领域分类分级标准词典（公开标准知识，入库）
- 本文档：下载说明 + 分层语义

## 如何使用 `data/processed`（processed → canonical → parquet → 训练/评估）

> 这是把 `data/processed` 交付给他人后、从零生成训练数据的标准步骤。
> 只需要：仓库代码（含 `cfg/task/registry` + `cfg/task/corpus`，clone 自带）+ `data/processed`
> + Python 3.10+ 且装有 pyarrow。**不需要** `raw` / `knowledge` / 已有 `canonical`/`sft`/`rl`。

每步失败会清晰报错且不产生产物（fail-fast）；已产出时重跑需 `--overwrite`。

**1) canonical（processed → `data/canonical/<ds>/all.json` + resolution_report.json）**

```bash
python -m script.canonical.targets --overwrite --datasets finance infra pers_info shougang
# 或单数据集：--dataset pers_info
```

**2) SFT parquet（canonical → `data/sft/<ds>/{train,val,test}.parquet`）**

```bash
for ds in finance infra pers_info shougang; do
  python -m script.verl.sft.export \
    --canonical data/canonical/$ds/all.json \
    --split-dir data/processed/$ds \
    --output-dir data/sft/$ds \
    --registry cfg/task/registry/$ds.registry.json \
    --corpus cfg/task/corpus/$ds.corpus.json \
    --metadata-fields field_name field_description
done
```

**3) RL parquet（canonical → `data/rl/<ds>/{train,val,test}.parquet`，五字段）**

```bash
for ds in finance infra pers_info shougang; do
  python -m script.verl.rl.export \
    --canonical data/canonical/$ds/all.json \
    --split-dir data/processed/$ds \
    --output-dir data/rl/$ds \
    --dataset $ds \
    --registry cfg/task/registry/$ds.registry.json \
    --corpus cfg/task/corpus/$ds.corpus.json \
    --metadata-fields field_name field_description
done
```

**4) 校验**（契约 + token 预算；token 预算需要模型 tokenizer）

```bash
python -m script.verl.sft.validate --dataset-dir data/sft/pers_info \
  --registry cfg/task/registry/pers_info.registry.json --corpus cfg/task/corpus/pers_info.corpus.json \
  --metadata-fields field_name field_description
python -m script.verl.sft.check_token_budget --dataset-dir data/sft/pers_info \
  --model <hf-model> --max-length 512
```

**5) 训练 / 评估**（需要 verl 环境 + GPU + 模型）

```bash
# SFT baseline（7B LoRA）
DATASET=pers_info DATA_DIR=data/sft/pers_info MODEL_PATH=<hf-model> \
  bash script/verl/sft/run_baseline.sh
# RL smoke（GRPO）
DATASET=pers_info TRAIN_FILE=data/rl/pers_info/train.parquet VAL_FILE=data/rl/pers_info/val.parquet \
  MODEL_PATH=<hf-model> bash script/verl/rl/grpo_smoke.sh
# 评估（choice protocol）
python -m script.verl.sft.evaluate_baseline --model-path <merged> \
  --data data/sft/pers_info/test.parquet --registry cfg/task/registry/pers_info.registry.json \
  --report tmp/eval.json
python -m script.verl.sft.evaluate_true_e2e --model-path <merged> \
  --data data/sft/pers_info/test.parquet --registry cfg/task/registry/pers_info.registry.json \
  --corpus cfg/task/corpus/pers_info.corpus.json --report tmp/eval_e2e.json
```

**预期产物 / 校验**（与迁移前仓库逐字节一致的基准行数）

| dataset | canonical resolved | SFT/RL train | val | test |
|---|---|---|---|---|
| finance | 531 | 806 | 138 | 114 |
| infra | 64 | 80 | 30 | 18 |
| pers_info | 176 | 280 | 36 | 36 |
| shougang | 18393 | 29042 | 3852 | 3892 |

（shougang canonical 含 1022 条 `placeholder` 不入训；finance 有 34 missing_leaf / 3 path_mismatch 不入训。）
生成 SFT/RL parquet 为**纯 pyarrow 计算**（无 torch/GPU）；训练/评估阶段才需要 GPU 与模型。
新数据集上线另见 `docs/新数据集运行说明.md`（raw → processed 的完整流程）。

## 迁移说明（2026-08，data layout refactor）

- 原 `data/<dataset>/all.json + splits`（预处理归一化源层）→ `data/processed/<dataset>/`。
- 原 `data/<dataset>/canonical/*` → `data/canonical/<dataset>/`。
- 新增 `data/raw/`（原始物）、`data/legacy/`（历史遗留物）。
- 删除与 `all.json` 逐字节重复的 `data/shougang/all_shougang.json`。
- `data/<dataset>/corpus.json`（legacy finance corpus，仅 analysis 对照用）→
  `data/legacy/finance.corpus.json`；正式 corpus 唯一来源为 `cfg/task/corpus/`。
- 本次迁移只改目录结构与路径引用，样本 / label / split / canonical resolution /
  prompt / reward 均未改动。

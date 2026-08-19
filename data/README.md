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

## 迁移说明（2026-08，data layout refactor）

- 原 `data/<dataset>/all.json + splits`（预处理归一化源层）→ `data/processed/<dataset>/`。
- 原 `data/<dataset>/canonical/*` → `data/canonical/<dataset>/`。
- 新增 `data/raw/`（原始物）、`data/legacy/`（历史遗留物）。
- 删除与 `all.json` 逐字节重复的 `data/shougang/all_shougang.json`。
- `data/<dataset>/corpus.json`（legacy finance corpus，仅 analysis 对照用）→
  `data/legacy/finance.corpus.json`；正式 corpus 唯一来源为 `cfg/task/corpus/`。
- 本次迁移只改目录结构与路径引用，样本 / label / split / canonical resolution /
  prompt / reward 均未改动。

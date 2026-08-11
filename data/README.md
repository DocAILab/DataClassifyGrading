# data/ — 数据目录

## 下载

数据来自 ModelScope

```bash
export MODELSCOPE_TOKEN=xxx     # token 走环境变量，禁止写入脚本/文档/仓库
git clone <ModelScope 数据集地址> .
```


## 各领域概况（2025-08-10 快照）

| 领域 | 业务域 | 记录数 | train/val/test |
|---|---|---|---|
| finance | 信托核心系统 | 568 | 439/69/58 |
| infra | 钢铁基建（=shougang 子集） | 64 | 40/15/9 |
| pers_info | 高校个人信息 | 176 | 140/18/18 |
| shougang | 首钢生产/管理/研发 | 19,415 | 15,347/2,028/2,040 |

## 入库内容

- `knowledge/standards_map/`：12 份多领域分类分级标准词典（公开标准知识，入库）
- 本文档：下载说明

# 第一版 Planner-only SFT

按 `export_dataset.py` → `token_audit.py` → 最长来源trial → fresh formal顺序阅读。只监督真实Planner Program JSON与EOS；Skill返回和评估标签不进入loss。200来源、1epoch、13更新、LoRA r8/alpha16/dropout.05，语言层40目标，base/vision冻结。完整参数及资源见 [实验协议](../docs/EXPERIMENT_PROTOCOL.md)。

`train_student.py`负责计算，`completion_loss.py`对齐因果completion logits，`training_contract.py`核trial/GPU门，`pipeline.py`管理子进程与独占锁，`runtime_io.py`保存不可覆盖的manifest证据。源码使用实际PyTorch/Transformers Trainer/PEFT接口；这不是替换为其他训练框架或新训练。

运行CPU检查：`python -B -m unittest discover -s tests -v`。真实训练需要未公开的planner200、token审计、manifest和模型文件，不能把公开脱敏teacher轨迹直接喂入训练。当前整理没有重训或生成新checkpoint。

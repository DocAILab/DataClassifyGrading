# F66同框架配对评估

`runtime/`是金融主干唯一Program/Skill实现。`evaluate.py`对同一来源交替AB/BA；`backend.py`保证base组所有请求关闭adapter，sft组只有可信Planner system请求开启adapter，结束/异常均恢复关闭。Skill两组都用原基座。`replay.py`拒绝实际请求和记录序列不一致；`gate.py`核模型、adapter和真实阶段退出；`metrics.py`保留固定分母。

原版预算与生成参数见 [完整协议](../docs/EXPERIMENT_PROTOCOL.md)。原先P14目录不再作为nontraining入口。不要直接用ProgramConfig通用默认值代替正式study预算，也不要省略 `skill_pool="F66"` 后声称相同实验。

当前runner保留原 `smoke` / `val80` 参数，公开study只是合成fixture；原完整422结果在 `../results`，私有的补充运行协调器未在此发布。真实运行需自己的受控manifest，源码整理后SHA变化必须重新绑定；不能重写旧审计身份。

CPU检查：在此目录 `python -B -m unittest discover -s tests -v`；另在 `runtime/` 运行 `python -B -m pytest tests -q -p no:cacheprovider`。这些检查没有模型下载、API、GPU或新成绩。

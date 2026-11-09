# 原版与历史快照

本次只整理金融 Program，起点：[b6fde9b80802593ad8ab30bdfae66fc28483a849](https://github.com/DocAILab/DataClassifyGrading/tree/b6fde9b80802593ad8ab30bdfae66fc28483a849/research/finance_program_v1)。

当前树移除 `nontraining/` 下的独立P14实现、历史runner/下载器/旧源码审计及其测试，只保留指向正式F66的README。文件级原gitblob/SHA见 [removed-historical-files.json](removed-historical-files.json)。可从原提交完整取回，结果和失败不因目录整理而消失。原版公开树的P14依赖测试中3项已因旧source-manifest与当前download_model源码散列不符而失败；这次未改旧manifest伪造通过，而是移除不属F66主干的历史检查，新增实际F66依赖/回放检查。

`evaluation/runtime`、Teacher、第一版SFT、结果/轨迹保留；活动源码只做注释和格式整理，并核AST一致。所有 `results/`、`trajectories/`、Skill body、注册表/合成数据、公开协议字符串保持原字节。保留legacy模型/哈希工具和teacher录制重试助手，因为它们仍被主干实际导入。

本地2026-11-09 recovery01/02/03及其61CPU、smoke8/部分VAL58，M8/M9/M10与Skill SFT不是这份原版主干结果，不迁入、不改旧分母、不与79→171表混成同协议。其他金融RL、汽车、教育、仓库应用不在本次变更范围。

整理不修原版反馈覆盖或重复规划行为，也不声称新的GPU结果。若未来改这些行为，需另建协议版本和原版对照，保存新失败与成本。

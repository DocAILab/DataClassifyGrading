# Finance Program: non-training + first Planner SFT

This branch adds an isolated code-review snapshot of the historical non-training baseline and first teacher→Planner SFT experiment. It does not replace the repository application or publish later training experiments.

Start here:
- [中文汇报](REPORT.zh-CN.md)
- [一分钟汇报稿](BRIEF.zh-CN.md)
- [Reproduction and privacy](REPRODUCE.md)
- [Aggregate results](results/evidence-summary.json)
- [440 redacted trajectories and digests](results/trajectory-export.json)

Directories: nontraining/ (historical P14), teacher/ (**Bailian qwen3.8-2.4t-a95b** collection code, not a 9B teacher), sft/ (Qwen3.5-9B first200-source LoRA), evaluation/ (matched base/SFT F66 Val80), trajectories/ (lossy public views), tools/ (redaction), publication_tests/ (privacy regression).

**Matched result:** joint correctness 16/80→36/80; legal termination36/80→80/80. The historical P14 baseline21/80 is a separate interface, not the strict paired baseline.

Public trajectory views preserve attempts, actual Program/step outcomes, final classification and original trace SHA, but remove business metadata, prompts and semantic evidence. They are **not full replay inputs or SFT training data**. Failures are retained. Real weights, gold, credentials and lab endpoints are not included.

No launch occurs on clone/import. Do not run training against synthetic CPU fixtures. See the report for development-set limitations and exact publication scope.


# Teacher identity and code

Teacher is **Bailian qwen3.8-2.4t-a95b**, not the Qwen3.5-9B student. The teacher generated the actual Planner/Skill traces used for the student's first SFT cold start. Teacher weights were not fine-tuned in this experiment.

teacher_api.py and policy.py contain the executed SSE transport and explicit task protocol; retry_backend.py and the budget modules retain bounded transport-only retry and append-only safety accounting. collect.py retains the label-free metadata boundary. detached.py is the historical CPU collector/snapshot orchestration, not a public turnkey deployment.

Only code and a placeholder configuration are included. Historical controlled manifests, private study/labels, original ledger, replay fixtures and lab transport/credential loader are not public. A real detached run requires the corresponding controlled layout and a newly authorized manifest; cloning this branch does not launch API requests. TeacherBackend receives a key in memory from its caller; do not save it in config/argv/logs.

CPU tests (synthetic I/O only, no API):

```powershell
python -B -m pytest tests -q -p no:cacheprovider
```

The 200 teacher trajectory views are in ../trajectories/teacher200.redacted.jsonl. Their original trace digests were checked against the actual first-SFT targets. Public views are lossy and are not training or full replay inputs.

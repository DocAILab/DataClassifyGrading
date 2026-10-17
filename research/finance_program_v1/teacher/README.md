# Teacher trajectory collection

The teacher is **Bailian `qwen3.8-2.4t-a95b`**, not the Qwen3.5-9B student. The teacher generates Planner/Skill traces for the student's first SFT cold start; teacher weights are not fine-tuned by this code.

`teacher_api.py` and `policy.py` implement SSE transport and the task protocol. `retry_backend.py` and the budget modules provide bounded transport-only retry and append-only safety accounting. `collect.py` maintains the label-free metadata boundary. `detached.py` contains historical CPU collector/snapshot orchestration, not a public turnkey deployment.

Only code and a placeholder configuration are included. Controlled manifests, private study/labels, original trajectories, ledgers, replay fixtures and lab transport/credential loaders are not included. A real detached run requires the corresponding controlled layout and a newly authorized manifest. Cloning this branch does not launch API requests. `TeacherBackend` receives a key in memory from its caller; do not save it in configuration, command arguments or logs.

Run CPU tests from this directory (synthetic I/O only, no API):

```powershell
python -B -m pytest tests -q -p no:cacheprovider
```

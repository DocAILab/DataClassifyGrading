# Stage 1 BGE-M3 retrieval baseline

This experiment ranks every registry `classification.level_4` label from
`metadata.field_name` alone. It uses frozen BGE-M3 normalized dense `[CLS]`
embeddings and reports a character n-gram control on the identical validation
rows. The command has no split option and reads `val.json` by literal path.

Run the detached remote pipeline with:

```bash
bash src/method/retrieval/script/start_stage1_bge_m3.sh
```

The main report is written to
`/root/autodl-tmp/artifacts/shougang/stage1-bge-m3-v1/full/evaluation_report.json`.
Stage 1 Recall@5 is not an end-to-end two-stage accuracy.

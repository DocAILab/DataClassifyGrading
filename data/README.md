# Local data layout

Only `data/knowledge/**/*.json` is allowed in Git.

Production data must be stored outside the repository or under ignored local directories. The classification and training commands accept explicit paths; they do not require tracked datasets, registries, corpora, parquet files, reports, or model artifacts.

Before committing, verify staged paths with:

```bash
git diff --cached --name-only
```

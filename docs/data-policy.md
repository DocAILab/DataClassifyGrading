# Repository data policy

The only tracked data exception is `data/knowledge/**/*.json`.

All production or derived datasets, standards exports, canonical records, corpora, registries, training files, metrics, and reports are runtime-local. Test fixtures and configuration examples must be demonstrably synthetic and use anonymous identifiers.

The `.gitignore` rules are a convenience guard, not the acceptance check. Every release candidate must also scan the current tree and all rewritten refs for forbidden paths, extensions, and known content sentinels.

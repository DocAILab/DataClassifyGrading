# Vehicle NONT-D execution code

This is a publication copy of the final v8 runtime, not the identity of the historical immutable run. See ../REPRODUCE.md and ../results/code-publication.json.

The upstream F66 runtime is byte-identical to team commit 9473b2c2a330b7f48dc0c1831bda11e36d428379. Only this README and local asset-path defaults in api_backend.py/night_prepare.py differ from the final runtime. Model prompts, controller logic, budgets and scoring are unchanged.

Credentials are read only by explicit live operations, from VEHICLE_NONTD_CREDENTIAL_FILE or ~/.config/vehicle-nontd/credentials.md. The existing parser expects the dedicated heading 汽车 NONT-D SiliconFlow API and one API Key line. Never commit that file. VEHICLE_NONTD_TOKENIZER selects a separately authorized local reference tokenizer. No credentials, tokenizer assets, real study, gold or raw traces are included.

Python 3.11+; offline tests use the standard library. Actual Windows status/supervision also requires psutil; real reference counting requires tokenizers and jinja2. Public tests never request a real model. The retained local GPU backend was not used for the reported API result.

Run `python -B -m unittest discover -s tests -v` from this directory. New controlled runs require newly prepared manifests; do not insert this changed code into the old run or weaken identity checks.

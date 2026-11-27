"""Run the single M9 evaluation or train its Planner with GRPO."""
import argparse
import json
from pathlib import Path
import tempfile

from episode import cohort, install_native, run_episode
from grpo import references, train
from model import load_backend
from native_assets import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="mode", required=True)
    evaluate = subs.add_parser("evaluate", help="Run M9 greedily on VAL")
    evaluate.add_argument("--candidate", type=Path, help="RL checkpoint step-NNN/planner")
    evaluate.add_argument("--references", type=Path, help="Optional scoring labels")
    rl = subs.add_parser("train-rl", help="Update the Planner on TRAIN")
    rl.add_argument("--references", required=True, type=Path)
    rl.add_argument("--group-size", type=int, default=4)
    rl.add_argument("--batch-sources", type=int, default=8)
    for command in (evaluate, rl):
        for key in ("model", "planner", "skill", "data", "output"):
            command.add_argument("--" + key, required=True, type=Path)
        command.add_argument("--seed", type=int, default=2026110509)
    args = parser.parse_args()
    training = args.mode == "train-rl"
    split = "TRAIN" if training else "VAL"
    if training and (args.group_size < 2 or args.batch_sources < 1):
        parser.error("Need at least two samples per group and a positive batch size")
    if args.output.exists():
        parser.error("Use a new output directory for each run")
    public = cohort(args.data, split)
    refs = references(args.references, public, split) if args.references else None
    from transformers import set_seed
    set_seed(args.seed)
    with tempfile.TemporaryDirectory() as tmp:
        install_native(prepare(Path(tmp) / "dependencies", public["scopes"]))
        backend = load_backend(args, training=training)
        args.output.mkdir(parents=True, exist_ok=False)
        if training:
            summary = train(public, refs, backend, args.output, args.group_size, args.batch_sources)
        else:
            cases = public["inputs"]["cases"]
            valid = matches = 0
            with (args.output / "predictions.jsonl").open("x", encoding="utf-8") as stream:
                for case in cases:
                    result, _ = run_episode(public, case, backend)
                    valid += result["valid"]
                    if refs is not None:
                        matches += result["valid"] and result["prediction"] == refs[case["source_id"]]
                    stream.write(json.dumps({"source_id": case["source_id"], **result}, ensure_ascii=False) + "\n")
                    stream.flush()
            summary = {"sources": len(cases), "valid": valid, "joint_matches": matches if refs else None}
        summary.update(mode=args.mode, seed=args.seed, sft_planner_sha256=backend.parent,
                       skill_sha256=backend.fixed_skill)
        if getattr(args, "candidate", None):
            from model import sha
            summary["evaluated_planner_sha256"] = sha(args.candidate / "adapter_model.safetensors")
        (args.output / "completion.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary))


if __name__ == "__main__":
    main()

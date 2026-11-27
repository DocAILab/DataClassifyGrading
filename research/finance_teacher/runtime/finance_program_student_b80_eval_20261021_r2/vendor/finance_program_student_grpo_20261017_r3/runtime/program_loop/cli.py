"""Local scripted demo, or optional generation from an existing local model."""
import argparse
import json
from pathlib import Path
import sys

from . import ProgramConfig, build_program_controller

DEMO_TOOLS = ("INTERPRET_FIELD_NAME", "SELECT_CATEGORY_TOPK", "INFER_CATEGORY_FROM_TOPK", "INFER_GRADING_LEVEL", "FINALIZE_CLASSIFICATION")


def scripted_generate(messages):
    system = messages[0]["content"]
    if "Global Program Planner" in system:
        return json.dumps({"program": [{"tool": tool, "arguments": {}} for tool in DEMO_TOOLS]})
    if "selected Skill INTERPRET_FIELD_NAME." in system:
        return json.dumps({"summary": "合成语义", "data": {"field_semantics": "字段表示账户余额"}})
    if "selected Skill SELECT_CATEGORY_TOPK." in system:
        return json.dumps({"option_ids": ["C001", "C002", "C003", "C004", "C005"], "summary": "合成候选"})
    if "selected Skill INFER_CATEGORY_FROM_TOPK." in system:
        return json.dumps({"option_id": "C001", "summary": "合成选择"})
    raise ValueError("unsupported scripted model request")


def local_generate(path):
    path = Path(path).resolve()
    if not path.is_dir(): raise ValueError(f"local model path does not exist: {path}")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("optional local model runtime is unavailable: transformers is not installed") from exc
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
    model.eval()

    def generate(messages):
        import torch
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoded = tokenizer(prompt, return_tensors="pt")
        with torch.inference_mode():
            output = model.generate(**encoded, do_sample=False, max_new_tokens=1024)
        return tokenizer.decode(output[0, encoded["input_ids"].shape[1]:], skip_special_tokens=True)
    return generate


def main(argv=None):
    parser = argparse.ArgumentParser(description="Program loop: scripted model I/O + real executor; no accuracy claim")
    parser.add_argument("--output", type=Path, help="new JSON output path; refuses overwrite")
    parser.add_argument("--local-model", type=Path, help="existing local model directory; local_files_only")
    parser.add_argument("--registry", type=Path, help="frozen Registry JSON; optional for synthetic demo")
    parser.add_argument("--metadata", type=Path, help="label-free privately scoped metadata JSON")
    parser.add_argument("--source-id", default="synthetic-demo", help="private metadata lookup identity")
    parser.add_argument("--domain", default="合成金融")
    parser.add_argument("--field-name", default="balance")
    for key in ("programs", "program-steps", "skill-steps", "planner-attempts"):
        parser.add_argument("--max-" + key, type=int, default=getattr(ProgramConfig(), "max_" + key.replace("-", "_")))
    args = parser.parse_args(argv)
    try:
        if args.output and args.output.exists(): raise FileExistsError(f"output already exists: {args.output}")
        generate = local_generate(args.local_model) if args.local_model else scripted_generate
        registry = args.registry or {f"cat{i}": {"label": f"合成类别{i}", "description": f"合成定义{i}", "grading_level": "L2"} for i in range(1, 7)}
        metadata = args.metadata or [{"id": args.source_id, "domain": args.domain, "metadata": {"field_name": args.field_name}}]
        config = ProgramConfig(max_programs=args.max_programs, max_program_steps=args.max_program_steps, max_skill_steps=args.max_skill_steps, max_planner_attempts=args.max_planner_attempts)
        controller = build_program_controller(generate, registry, metadata, config)
        trace = controller.run({"domain": args.domain, "field_name": args.field_name, "source_id": args.source_id})
        trace.execution_mode = "local model I/O + real executor" if args.local_model else "scripted model I/O + real executor"
        if args.output: trace.write_json(args.output)
        print(json.dumps({"execution_mode": trace.execution_mode, "terminated": trace.terminated, "stop_reason": trace.stop_reason, "budget": trace.budget, "output": str(args.output) if args.output else None}, ensure_ascii=True))
        return 0 if trace.terminated else 1
    except Exception as exc:
        print(f"blocked: {exc}", file=sys.stderr)
        return 2

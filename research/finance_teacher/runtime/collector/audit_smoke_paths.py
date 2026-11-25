"""Audit executed Skill paths and classifier citations from an offline smoke score file."""
import argparse
from collections import Counter, defaultdict
from itertools import combinations
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_bytes())


SEARCHES = {"SEARCH_STANDARD", "SEARCH_STANDARD_BY_FIELD", "SEARCH_STANDARD_BY_CONTEXT",
            "SEARCH_STANDARD_BY_TABLE", "SEARCH_STANDARD_BY_DATABASE", "RESEARCH_STANDARD"}


def family(skill):
    if skill in SEARCHES:
        return "SELECT_CATEGORY_TOPK"
    return "INFER_CATEGORY_FROM_TOPK" if skill == "INFER_CATEGORY" else skill


def core_path(sequence):
    path = list(map(family, sequence))
    if path[:1] == ["GET_FIELD_METADATA"]:
        path.pop(0)
    if path[:1] == ["GET_TABLE_FIELDS"]:
        path.pop(0)
    for end in ("FINALIZE_CLASSIFICATION", "INFER_GRADING_LEVEL", "INFER_CATEGORY_FROM_TOPK"):
        if path[-1:] == [end]:
            path.pop()
    return path


def token_distance(left, right):
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j-1] + (a != b)))
        previous = current
    return previous[-1]


def classifier_citations(folder, trace):
    accepted = [a for a in trace["evidence_audits"] if a["status"] == "validated_citations"]
    if not accepted:
        return None
    audit = accepted[-1]
    for request in sorted(folder.glob("call-*/request.json")):
        req = read(request)
        if not req.get("stage", "").startswith("INFER_CATEGORY"):
            continue
        response = read(request.parent / "outcome.json")
        if response["provider_target_sha256"] != audit["provider_response_sha256"]:
            continue
        raw = json.loads(response["provider_raw_content"])
        support = raw["support"]
        if support != audit["support"]:
            raise ValueError(f"Support differs from replay in {request}")
        payload = json.loads(req["body"]["messages"][-1]["content"])
        facts = payload["evidence_boundary"]["facts"]
        citations = []
        for cite in support["citations"]:
            fid, quote = cite["fact_id"], cite["quote"]
            fact = facts.get(fid, {})
            citations.append({"id": fid, "quote": quote, "scope": fact.get("source_scope"),
                              "eligible": fact.get("decisive_eligible", False),
                              "verbatim": bool(quote) and quote in str(fact.get("value", ""))})
        return {"summary": raw.get("summary"), "sufficient": support["sufficient"], "citations": citations,
                "chosen_option": raw["option_id"],
                "candidate_labels": {c["option_id"]: c["label"] for c in payload["candidate_details"]}}
    raise ValueError(f"Missing archived classifier response in {folder}")


def inspect_row(directory, row):
    trace_file = directory / row["trace"]
    trace = read(trace_file)
    if trace["source_id"] != row["source_id"]:
        raise ValueError(f"Source mismatch in {trace_file}")
    inputs = read(trace_file.parents[2] / "program-inputs.json")
    meta = next(s["metadata"] for s in inputs["study"]["rows"] if s["id"] == row["source_id"])
    path, lookups = [], []
    observed = {}
    seen = set()
    for program in trace["programs"]:
        for step in program["steps"]:
            call = step["call"]
            tool = call["tool"]
            path.append(tool)
            if tool == "GET_FIELD_METADATA" and step["status"] == "success":
                observed.update(step["data"]["field_metadata"])
            if tool.startswith(("GET_", "ACQUIRE_")):
                args = call.get("arguments", {})
                key = tool, json.dumps(args, sort_keys=True)
                lookups.append({"skill": tool, "arguments": args, "status": step["status"],
                                "same_arguments_seen": key in seen, "new_raw_facts": len(step["new_raw_fact_ids"])})
                seen.add(key)
    if not path:
        raise ValueError(f"Empty executed path in {trace_file}")
    programs = [json.loads(a["raw_response"])["program"] for a in trace["planner_attempts"]
                if a["status"] == "accepted"]
    if programs and programs[0] == [{"tool": "GET_FIELD_METADATA", "arguments": {}}]:
        programs = programs[1:]
    support = classifier_citations(trace_file.parent, trace)
    if row["decision_status"] == "classified" and not support:
        raise ValueError(f"Missing classifier evidence in {trace_file}")
    return {"source_index": row["source_index"], "source_id": row["source_id"], "field": meta["field_description"],
            "table": observed.get("table_name"), "database": observed.get("database_name"),
            "decision": row["decision_status"],
            "eligible_for_sft_review": row["eligible_for_sft_review"], "raw_path": path,
            "family_path": list(map(family, path)), "core_path": core_path(path), "lookups": lookups,
            "accepted_programs_after_start": programs,
            "planner_statuses": dict(Counter(a["status"] for a in trace["planner_attempts"])),
            "blocked_classifications": sum(a["status"] == "locally_blocked" for a in trace["evidence_audits"]),
            "classifier_support": support}


def path_groups(rows, key):
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[key])].append(r["source_index"])
    ordered = sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))
    sizes = [len(ids) for _, ids in ordered]
    return {"unique": len(groups), "largest": max(sizes, default=0),
            "members_of_repeated_paths": sum(n for n in sizes if n > 1),
            "identical_pairs": sum(n * (n - 1) // 2 for n in sizes),
            "groups": [{"members": sorted(ids), "sequence": path} for path, ids in ordered]}


def summarize(rows):
    pairs = []
    for a, b in combinations(rows, 2):
        left, right = a["core_path"], b["core_path"]
        n = max(1, len(left), len(right))
        distance = token_distance(left, right)
        relation = "unknown" if not (a["table"] and b["table"]) else (
            "same" if (a["database"], a["table"]) == (b["database"], b["table"]) else "different")
        pairs.append({"members": [a["source_index"], b["source_index"]], "edits": distance,
                      "length": n, "similarity": round(1-distance/n, 4), "at_least_80pct": 5*distance <= n,
                      "tables": relation})
    scopes = {}
    for relation in ("all", "same", "different", "unknown"):
        subset = [p for p in pairs if relation == "all" or p["tables"] == relation]
        scopes[relation] = {"pairs": len(subset), "identical": sum(p["edits"] == 0 for p in subset),
                            "at_least_80pct": sum(p["at_least_80pct"] for p in subset)}
    lookups = [s for r in rows for s in r["lookups"]]
    supports = [r["classifier_support"] for r in rows if r["decision"] == "classified"]
    checked = [s["citations"] for s in supports]
    targets = defaultdict(list)
    for r in rows:
        for program in r["accepted_programs_after_start"]:
            targets[json.dumps(program, sort_keys=True, ensure_ascii=False)].append(r["source_index"])
    shared = [ids for ids in targets.values() if len(set(ids)) > 1]
    cited = [r for r in rows if r["decision"] == "classified"]
    citation_pairs = []
    for a, b in combinations(cited, 2):
        x, y = a["classifier_support"], b["classifier_support"]
        qx, qy = ({q["quote"] for q in s["citations"] if q["verbatim"]} for s in (x, y))
        citation_pairs.append({"members": [a["source_index"], b["source_index"]],
                               "identical_summary": bool(x["summary"]) and x["summary"] == y["summary"],
                               "shared_quotes": len(qx & qy), "distinct_quotes": len(qx | qy)})
    return {"sources": len(rows), "paths": {k: path_groups(rows, k) for k in ("raw_path", "family_path", "core_path")},
            "planner_programs": {"total": sum(map(len, targets.values())), "unique": len(targets),
                                 "instances_shared_across_fields": sum(map(len, shared)),
                                 "most_frequent_count": max(map(len, targets.values()), default=0)},
            "classifier_pairs": len(citation_pairs),
            "identical_summary_pairs": sum(p["identical_summary"] for p in citation_pairs),
            "quote_overlap_80pct_pairs": sum(p["distinct_quotes"] > 0 and
                                             5*p["shared_quotes"] >= 4*p["distinct_quotes"] for p in citation_pairs),
            "classifier_pair_details": citation_pairs,
            "core_similarity": scopes, "pair_details": pairs, "lookup_calls": len(lookups),
            "lookup_unavailable": sum(s["status"] == "unavailable" for s in lookups),
            "exact_repeat_lookups_without_new_fact_ids": sum(s["same_arguments_seen"] and not s["new_raw_facts"]
                                                              for s in lookups),
            "blocked_classifications": sum(r["blocked_classifications"] for r in rows),
            "planner_attempts": dict(sum((Counter(r["planner_statuses"]) for r in rows), Counter())),
            "classified": len(supports), "citations": sum(map(len, checked)),
            "classified_all_quotes_verbatim": sum(bool(c) and all(q["verbatim"] for q in c) for c in checked),
            "classified_with_eligible_quote": sum(any(q["eligible"] and q["verbatim"] for q in c) for c in checked),
            "classified_citing_field_description": sum(any(q["id"] == "metadata.field_description" and q["verbatim"]
                                                          for q in c) for c in checked),
            "classified_citing_table_fields": sum(any(q["id"].startswith("observations.table_fields.fields.")
                                                    and q["verbatim"] for q in c) for c in checked)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    records = [inspect_row(args.scores.parent, r) for r in read(args.scores)["rows"] if r.get("trace")]
    report = {"score_rows": len(read(args.scores)["rows"]), "all": summarize(records),
              "sft_review_subset": summarize([r for r in records if r["eligible_for_sft_review"]]), "rows": records}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

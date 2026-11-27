"""Verify the 422 student predictions against their raw evidence, then score offline."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import statistics
import tarfile

STUDY=Path(__file__).resolve().parents[1]
HERE=STUDY/"results/sft"
VAL=STUDY.parent/"val422"
ARMS=("student",)
HASHES={"student":"1d527f90b632a80809d6b0e152d62e04c773711c9ceb73ae8b441d7cb2f0bcd0",
        "fresh162":"6d78e440ab0c9749427e002d54fd94cca06dea77c31b63ae3628b48fdbec4e64",
        "skill":"0e52b9439b0016b0cad2903392030fca4b501019c7378a3a594b0fb6d1d4faf7"}


def read(path):
    return json.loads(path.read_bytes())


def shape(rows,key):
    usable=[tuple(row[key]) for row in rows if row[key]]
    paths=Counter(usable)
    if not paths:
        return dict(unique=0,top=[],count=0,without_path=len(rows))
    plan,count=paths.most_common(1)[0]
    return dict(unique=len(paths),top=list(plan),count=count,without_path=len(rows)-len(usable))


def retrieved_categories(trace):
    """Collect categories explicitly returned by successful standard searches."""
    categories=set()
    for program in trace["programs"]:
        for step in program["steps"]:
            if step["status"]=="success" and step["call"]["tool"].startswith("SEARCH_STANDARD"):
                for candidate in step["data"].get("standard_candidates",[]):
                    if candidate.get("category"):
                        categories.add(candidate["category"])
    return categories


def verify_archive(records):
    retrieved={}
    expected={f"runs/{arm}/{sid}/{name}":(arm,sid,name) for arm in ARMS for sid in records[arm]
              for name in ("trace.json","calls.json","result.json")}
    with tarfile.open(HERE/"全部轨迹.tar.gz","r|gz") as archive:
        for member in archive:
            if member.name not in expected:
                continue
            arm,sid,name=expected.pop(member.name)
            row=records[arm][sid]
            blob=archive.extractfile(member).read()
            if name=="result.json":
                value=json.loads(blob)
                assert all(row[k]==value[k] for k in ("source_id","planner","skill","valid","prediction","stop_reason","calls"))
            else:
                hash_key="trace_sha256" if name=="trace.json" else "calls_sha256"
                if hashlib.sha256(blob).hexdigest()!=row[hash_key]:
                    raise ValueError(f"Evidence checksum mismatch: {member.name}")
                value=json.loads(blob)
                if name=="trace.json":
                    steps=[s["call"]["tool"] for p in value["programs"] for s in p["steps"]]
                    assert value["source_id"]==sid and value["final_result"]==row["prediction"] and steps==row["path"]
                    retrieved[sid]=retrieved_categories(value)
                else:
                    assert value[0]["input_sha256"]==row["first_input_sha256"] and value[0]["stage"]=="Planner"
    if expected:
        raise ValueError(f"Archive is missing {len(expected)} evidence files")
    return retrieved


def report(o):
    n=o["n"]
    def fraction(value):
        return f"{value}/{n}（{100*value/n:.1f}%）"
    lines=["# 新版 Student Planner 的 Val422 全量评估","",
           "在 aiss-3 的 RTX 6000 Ada 上运行新训练的 Qwen3.5-9B Planner。分类器使用固定的 Scale1169，"
           "422 个字段都使用同一版 M9 贪心推理和查表工具。每条预测单独保存。","",
           "| 指标 | 结果 |","| --- | ---: |"]
    for k,title in (("joint","类别和等级同时正确"),("valid","合法终止"),("category","类别正确"),("grade","等级正确")):
        lines.append(f"| {title} | {fraction(o[k])} |")
    lines.extend([f"| 平均每字段模型调用 | {o['mean_calls']:.2f} |",
                  f"| 最常见的完整工具顺序 | {fraction(o['full_path']['count'])}，共 {o['full_path']['unique']} 种 |",
                  f"| 最常见的首次计划工具顺序 | {fraction(o['first_plan']['count'])}，共 {o['first_plan']['unique']} 种 |",
                  f"| 有多次规划尝试的字段 | {o['multiple_plans']} |","",
                  "正确率以全部 422 条为分母，未合法终止计作未命中。路径按工具名称和先后顺序分组。"
                  "检索 Top 5 的三次模型前向计作三次调用。","",
                  "## 错例和覆盖","","以下列出有合法预测时最常见的类别误判。","",
                  "| 参考类别 | 预测类别 | 字段数 | 示例字段 |","| --- | --- | ---: | --- |"])
    for e in o["confusions"][:5]:
        lines.append(f"| {e['reference'].split('/')[-1]} | {e['predicted'].split('/')[-1]} | {e['count']} | {e['example_field']} |")
    errors=o["wrong_category_retrieval"]
    lines.extend(["",f"在 {sum(errors.values())} 个合法终止且类别判错的字段中，"
                  f"{errors.get('hit',0)} 个曾在检索结果中出现参考类别，"
                  f"{errors.get('miss',0)} 个返回了其他候选但未列出参考类别，"
                  f"{errors.get('empty',0)} 个没有返回可核对的候选类别。"
                  "这里统计每条轨迹中所有成功的标准检索。","",
                  "| 参考类别 | 字段数 | 类别与等级均匹配 |","| --- | ---: | ---: |"])
    for key,b in sorted(o["by_category"].items(),key=lambda v:(-v[1]["n"],v[0])):
        lines.append(f"| {key.split('/')[-1]} | {b['n']} | {b['joint']} |")
    lines.extend(["","最常见的实际执行路径为 "+" → ".join(o["full_path"]["top"])+"。","",
                  "参考标签只在本机评分时按来源 ID 对齐。训练用的 200 个字段和这 422 个验证字段的来源 ID 无重叠。"
                  "本轮固定了模型权重、分类器、提示和评分口径。评分衡量学生接入当前 M9 入口后的表现。"
                  "训练、部署使用的提示格式有差异，核对的 201 个训练回合中，两端提示全文完全一致的有 0 个。"
                  "[训练与部署的提示格式核对](../../sft/input-audit.json)","",
                  "[逐字段预测与评分](逐字段结果.csv) · [详细指标](评分.json) · [全部原始轨迹与调用记录](全部轨迹.tar.gz) · [复算脚本](../../sft/score_full.py)"])
    (HERE/"README.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def main():
    snap=read(HERE/"snapshot.json")
    if not snap["finished"] or snap["review"]["student"] or snap["state"]["arms"]!=["student"]:
        raise ValueError("Student evaluation is not complete")
    source=VAL/"val422-references.json"
    if hashlib.sha256(source.read_bytes()).hexdigest()!="2df78f4d53402af486a9b354e494ff73d03fcd7cfcefe1cbba719ad182ccf0ef":
        raise ValueError("Reference checksum mismatch")
    refs=read(source)["references"]
    cases=read(VAL/"val422-inputs.json")["rows"]
    records={r["source_id"]:r for r in snap["records"]["student"]}
    if len(records)!=422 or len(snap["records"]["student"])!=422 or set(records)!=set(refs) or set(records)!={c["source_id"] for c in cases}:
        raise ValueError("Expected an exact match on all 422 unique source IDs")
    retrieved=verify_archive({"student":records})
    output=[]
    conf=Counter()
    examples={}
    by_category={}
    by_table={}
    wrong_retrieval=Counter()
    for c in cases:
        sid=c["source_id"]
        r=records[sid]
        assert (r["planner"],r["skill"])==(HASHES["student"],HASHES["skill"])
        pred=r["prediction"] or {}; ref=refs[sid]
        category=r["valid"] and pred.get("category")==ref["category"]
        grade=r["valid"] and pred.get("grading_level")==ref["grading_level"]
        joint=bool(category and grade)
        table=c["metadata"].get("table_name","")
        field=c["metadata"].get("field_description") or c["metadata"]["field_name"]
        output.append(dict(source_id=sid,field=field,field_name=c["metadata"]["field_name"],table=table,
            reference_category=ref["category"],reference_grade=ref["grading_level"],predicted_category=pred.get("category"),
            predicted_grade=pred.get("grading_level"),valid=r["valid"],category_match=category,grade_match=grade,joint_match=joint,
            stop=r["stop_reason"],calls=r["calls"],planner_attempts=len(r["attempts"]),path=" → ".join(r["path"]),
            retrieved_category_count=len(retrieved[sid]),reference_among_retrieved=ref["category"] in retrieved[sid]))
        if r["valid"] and not category:
            coverage="empty" if not retrieved[sid] else ("hit" if ref["category"] in retrieved[sid] else "miss")
            wrong_retrieval[coverage]+=1
            pair=(ref["category"],pred["category"]); conf[pair]+=1
            examples.setdefault(pair,dict(example_field=field,example_id=sid))
        for key,values in ((ref["category"],by_category),(table or "（表名缺失）",by_table)):
            group=values.setdefault(key,Counter());group["n"]+=1;group["joint"]+=joint
    rows=list(records.values());calls=[r["calls"] for r in rows]
    o=dict(n=len(rows),valid=sum(r["valid"] for r in rows),joint=sum(r["joint_match"] for r in output),
           category=sum(r["category_match"] for r in output),grade=sum(r["grade_match"] for r in output),
           full_path=shape(rows,"path"),first_plan=shape(rows,"first_tools"),multiple_plans=sum(len(r["attempts"])>1 for r in rows),
           total_calls=sum(calls),mean_calls=statistics.mean(calls),median_calls=statistics.median(calls),
           stops=Counter(r["stop_reason"] for r in rows),by_category=by_category,by_table=by_table,
           wrong_category_retrieval=wrong_retrieval,
           confusions=[dict(reference=k[0],predicted=k[1],count=count,**examples[k]) for k,count in conf.most_common()],
           planner_sha256=HASHES["student"],skill_sha256=HASHES["skill"])
    with (HERE/"逐字段结果.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        w=csv.DictWriter(stream,fieldnames=list(output[0]));w.writeheader();w.writerows(output)
    (HERE/"评分.json").write_text(json.dumps(o,ensure_ascii=False,indent=2),encoding="utf-8")
    report(o)
    print(json.dumps({k:o[k] for k in ("n","valid","joint","category","grade","full_path","first_plan","stops")},ensure_ascii=False))


if __name__=="__main__":
    main()

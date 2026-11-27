"""Verify rollout evidence and score the selected RL checkpoint on the locked Val422."""
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import tarfile

STUDY=Path(__file__).resolve().parents[1]
HERE=STUDY/"results/rl"
VAL=STUDY.parent/"val422"
BASE=STUDY/"results/sft"


def read(p):
    return json.loads(p.read_bytes())


def stats(rows, labels):
    valid=sum(r["valid"] for r in rows)
    paired=sum(r["valid"] and r["prediction"]==labels[r["source_id"]] for r in rows)
    category=sum(r["valid"] and r["prediction"]["category"]==labels[r["source_id"]]["category"] for r in rows)
    grade=sum(r["valid"] and r["prediction"]["grading_level"]==labels[r["source_id"]]["grading_level"] for r in rows)
    paths=Counter(tuple(r["path"]) for r in rows)
    return dict(total=len(rows),valid=valid,joint=paired,category=category,grade=grade,max_path=paths.most_common(1)[0][1],paths=len(paths),
                stops=dict(Counter(r["stop_reason"] for r in rows)),
                mean_calls=sum(r["calls"] for r in rows)/len(rows))


def report(o):
    final=o["final"]
    def fraction(k):
        return f"{final[k]}/{final['total']}（{100*final[k]/final['total']:.1f}%）"
    b=o["selected"]
    dev=o["dev"]
    delta=final["joint"]-198
    changes=o["changes"]
    reward_text="合法终止且类别、等级均匹配得 1 分。"
    if o.get("uses_skill_rewards"):
        reward_text+="检索排名和证据引用还可获得过程分，各批断点保存了每条轨迹的分数。"
    else:
        reward_text+="其余结果得 0 分。"
    monitoring="Windows 同步任务在连接恢复后补下载断点并验证文件哈希。"
    if (HERE/"守夜状态.json").is_file():
        monitoring+="服务器每半小时归档巡检状态。[守夜记录](守夜状态.json)"
    audit_file="pilot-reward-audit.json" if o.get("uses_skill_rewards") else "pilot-audit.json"
    w=HERE/"updates"/f"batch-{b:02d}"/"planner"
    lines=["# 新版 Student Planner 的 GRPO 后训练","",
           f"用训练集中的 128 个字段做在线 GRPO，在独立开发集选出第 {b} 批的 Planner。"
           f"该权重在 Val422 上类别与等级同时答对 {fraction('joint')}，比 SFT 的 198/422 净变化 {delta:+d} 条。","",
           "| Val422 指标 | SFT | RL |","| --- | ---: | ---: |",
           f"| 类别与等级同时命中 | 198/422（46.9%） | {fraction('joint')} |",
           f"| 类别正确 | 198/422（46.9%） | {fraction('category')} |",
           f"| 等级正确 | 240/422（56.9%） | {fraction('grade')} |",
           f"| 合法终止 | 419/422（99.3%） | {fraction('valid')} |",
           f"| 最常用完整工具顺序 | 345/422（81.8%） | {fraction('max_path')} |",
           f"| 每字段平均模型调用次数 | 7.21 | {final['mean_calls']:.2f} |","",
           f"逐字段配对后，{changes.get('错转对',0)} 个字段由错转对，{changes.get('对转错',0)} 个由对转错。","",
           f"和 SFT 相比，{o['paths_changed']}/422 个字段改变了完整工具顺序，"
           f"{o['predictions_changed']}/422 个字段改变了最终的类别、等级或是否给出结果。","",
           "## 数据与奖励","",
           "1,083 个正式 RL 训练字段中，有 402 个来源既未参与此轮 Planner SFT，也未参与 Scale1169 训练。"
           "开发字段来自五张整表，与本轮训练字段的表名无重叠。小试覆盖 32 个字段，每个字段采样四次。"
           f"其中 {o['pilot']['mixed']} 个字段有对有错。正式训练每批 8 个字段，每个字段用当时的策略重新采样四次，共 512 条轨迹。","",
           f"正式训练的 128 个字段中，{o['reward_groups']['mixed']} 个四次采样有对有错，能提供组内优劣信号。"
           f"其余 {o['reward_groups']['all_correct']} 个四次全对，{o['reward_groups']['all_wrong']} 个四次全错。","",
           "使用已评估的 Qwen3.5-9B 学生 Planner 初始化可训练策略和冻结的参考策略。分类 Skill 使用 Scale1169。"
           +reward_text+"推理只接收公开字段元数据和已取得的证据。"
           "训练使用 M9 入口，学习率为 5×10⁻⁶。","",
           "## 开发集选模","",
           "| 完成批次 | 开发集联合命中 |","| --- | ---: |"]
    for k,row in sorted(dev.items()):
        lines.append(f"| {int(k[-2:])} | {row['joint']}/{row['total']} |")
    lines += ["","在预先指定的第 4、8、16 批中，选择开发集联合命中最高、平分时最早的已更新模型。"
              "同一份 Val422 的参考标签只在本机最终评分时使用。","",
              "## 保存与复核","",
              f"完成 {o['boundaries']} 个批次，其中 {o['effective_updates']} 个实际改变参数。"
              f"每批原子保存权重、AdamW 状态、随机状态和续训位置。首次有效更新后实际退出训练进程，"
              f"后续有 {o['resumed']} 次从前一批检查点恢复的记录。"
              f"已核对全部 {o['rollouts_verified']} 条采样及评估结果与原始轨迹。","",
              monitoring+
              f"512 条正式训练轨迹中有 {o['training_token_limits']} 条耗尽单次生成的 token 配额且未输出结束符，"
              "按未完成计 0 分，保留已生成的 token。"
              f"另有 {o['oversized_search_training_episodes']} 条正式训练轨迹曾触发检索输入长度限制，执行器按原有工具失败反馈继续处理。"
              f"因整段上下文容量耗尽而结束的有 {o['training_stops'].get('context_token_limit',0)} 条，"
              f"因模型调用预算耗尽而结束的有 {o['training_stops'].get('episode_call_limit',0)} 条。"
              "服务故障单独处理。","",
              "各类别的配对结果可用于定位变化较大的字段。","","| 参考类别 | 字段数 | SFT 正确 | RL 正确 |",
              "| --- | ---: | ---: | ---: |"]
    for k,v in sorted(o["categories"].items(),key=lambda e:-e[1]["n"]):
        lines.append(f"| {k} | {v['n']} | {v['before']} | {v['after']} |")
    lines+=["",f"[选中的 Planner 权重]({w.relative_to(HERE).as_posix()}/adapter_model.safetensors) · "
            "[逐字段对照](Val422逐字段.csv) · [全部指标](评分.json) · [原始轨迹](https://github.com/DocAILab/DataClassifyGrading/releases/download/finance-teacher200-sft-grpo-20261127/rl-evidence.tar.gz) · "
            f"[小试复核]({audit_file}) · [断点与进度](progress.json)"]
    (HERE/"README.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def verify_training_rewards(update, ids, repeats, index, labels, counts):
    stage=f"train-{update['batch']:02d}"
    components=update.get("skill_reward_components")
    assert (components is not None)==("skill_reward.py" in update["contract"])
    if components is not None:
        assert len(components)==len(ids)*repeats
    expected=[]
    for field, sid in enumerate(ids):
        outcomes=[]
        totals=[]
        for sample in range(repeats):
            row=index[stage,f"{sid}_{sample}"]
            outcome=int(row["valid"] and row["prediction"]==labels[sid])
            outcomes.append(outcome)
            if components is None:
                totals.append(outcome)
                continue
            part=components[field*repeats+sample]
            assert (part["source_id"],part["sample"],part["outcome"])==(sid,sample,outcome)
            total=outcome+part["retrieval"]+part["evidence"]
            assert total==part["total"] and (row["valid"] or total==0)
            totals.append(total)
        expected.append(totals)
        group="all_correct" if all(outcomes) else "mixed" if any(outcomes) else "all_wrong"
        counts[group]+=1
    assert expected==update["rewards"]


def read_evidence(archive):
    index={}
    evidence={}
    oversized_search=Counter()
    with tarfile.open(archive,"r|gz") as tar:
        for m in tar:
            if not m.isfile():
                continue
            path=Path(m.name)
            if path.parts[0]!="runs":
                continue
            key=tuple(path.parts[1:3])
            if path.name=="result.json":
                assert key not in index
                index[key]=json.load(tar.extractfile(m))
            elif path.name=="evidence.json.gz":
                assert key not in evidence
                blob=tar.extractfile(m).read()
                content=json.loads(gzip.decompress(blob))
                tr=content["trace"]
                if tr.get("rank_input_limits"):
                    assert all(e["tokens"]>e["limit"]==8192 for e in tr["rank_input_limits"])
                    oversized_search[key[0]]+=1
                steps=[s["call"]["tool"] for p in tr["programs"] for s in p["steps"]]
                actions=content["actions"]
                if key[0]=="pilot" or key[0].startswith("train-"):
                    assert actions and all(a["prompt"] and len(a["tokens"])==len(a["behavior"])>0 for a in actions)
                else:
                    assert not actions
                evidence[key]=(hashlib.sha256(blob).hexdigest(),tr["source_id"],tr["final_result"],steps,tr["stop_reason"])
    assert set(index)==set(evidence)
    return index,evidence,oversized_search


def score_evidence(index,evidence,oversized_search):
    plan=read(HERE/"plan.json")
    progress=read(HERE/"progress.json")
    assert progress["complete"] and not progress["errors"]
    labels=read(STUDY/"rl/reward-labels.json")
    ref_file=VAL/"val422-references.json"
    assert hashlib.sha256(ref_file.read_bytes()).hexdigest()=="2df78f4d53402af486a9b354e494ff73d03fcd7cfcefe1cbba719ad182ccf0ef"
    gold=read(ref_file)["references"]
    records=defaultdict(list)
    allowed={"pilot":(plan["pilot"],4),"val":(list(gold),1)}
    for b in [0,*plan["dev_batches"]]:
        allowed[f"dev-{b:02d}"]=(plan["dev"],1)
    for b in range(1,len(plan["train"])//8+1):
        allowed[f"train-{b:02d}"]=(plan["train"][8*b-8:8*b],4)
    assert {(stage,sid+"_"+str(k)) for stage,(ids,rep) in allowed.items() for sid in ids for k in range(rep)}==set(index)
    weights={0:plan["parent"]}
    for u in progress["updates"]:
        prefix=HERE/"updates"/f"batch-{u['batch']:02d}"
        for file,key in ((prefix/"planner/adapter_model.safetensors","planner_sha256"),(prefix/"resume.pt","resume_sha256")):
            with file.open("rb") as f:
                assert hashlib.sha256(f.read()).hexdigest()==u[key]
        weights[u["batch"]]=u["planner_sha256"]
    for (stage,key),r in index.items():
        assert (r["sha256"],r["source_id"],r["prediction"],r["path"],r["stop_reason"])==evidence[stage,key]
        assert key==r["source_id"]+"_"+str(r["sample"]) and r["actor"]==weights[r["batch"]]
        records[stage].append(r)
    selected=progress["selected"]["batch"]
    assert all(r["batch"]==selected for r in records["val"])
    dev={stage:stats(rows,labels) for stage,rows in records.items() if stage.startswith("dev-")}
    for b in [0,*plan["dev_batches"]]:
        assert dev[f"dev-{b:02d}"]["joint"]==progress["selected"]["scores"][str(b)]
    candidates=[b for b in plan["dev_batches"] if any(u["updated"] and u["batch"]<=b for u in progress["updates"])]
    assert selected==max(candidates,key=lambda b:(dev[f"dev-{b:02d}"]["joint"],-b))
    reward_groups=dict(mixed=0,all_correct=0,all_wrong=0)
    for update in progress["updates"]:
        stage=f"train-{update['batch']:02d}"
        ids,repeats=allowed[stage]
        verify_training_rewards(update,ids,repeats,index,labels,reward_groups)
    assert sum(reward_groups.values())==len(plan["train"])
    pilot=defaultdict(list)
    for r in records["pilot"]:
        pilot[r["source_id"]].append(r)
    mixed=[sid for sid,rs in pilot.items() if 0<stats(rs,labels)["joint"]<4]
    assert len(mixed)==progress["pilot"]["mixed"]
    with (BASE/"逐字段结果.csv").open(encoding="utf-8-sig",newline="") as f:
        baseline={r["source_id"]:r for r in csv.DictReader(f)}
    assert set(baseline)==set(gold)
    changes=Counter()
    by_cat=defaultdict(Counter)
    result={r["source_id"]:r for r in records["val"]}
    detail=[]
    paths_changed=0
    predictions_changed=0
    for sid,ref in gold.items():
        old=baseline[sid]
        new=result[sid]
        before=old["joint_match"]=="True"
        after=bool(new["valid"] and new["prediction"]==ref)
        key=("两次均对" if before else "错转对") if after else ("对转错" if before else "两次均错")
        group=by_cat[ref["category"].split("/")[-1]]
        group.update(dict(n=1,before=int(before),after=int(after)))
        changes[key]+=1
        pred=new["prediction"] or {}
        new_path=" → ".join(new["path"])
        paths_changed+=old["path"]!=new_path
        predictions_changed+=(old["predicted_category"],old["predicted_grade"])!=(pred.get("category") or "",pred.get("grading_level") or "")
        detail.append(dict(source_id=sid,field=old["field"],reference_category=ref["category"],reference_grade=ref["grading_level"],
            sft_category=old["predicted_category"],sft_grade=old["predicted_grade"],rl_category=pred.get("category"),
            rl_grade=pred.get("grading_level"),sft_joint=before,rl_joint=after,change=key,stop=new["stop_reason"],
            calls=new["calls"],path=new_path))
    with (HERE/"Val422逐字段.csv").open("w",newline="",encoding="utf-8-sig") as f:
        writer=csv.DictWriter(f,fieldnames=list(detail[0]))
        writer.writeheader()
        writer.writerows(detail)
    final=stats(records["val"],gold)
    assert final["joint"]==198+changes["错转对"]-changes["对转错"]
    verification=dict(pilot=progress["pilot"],dev=dev,selected=selected,final=final,changes=changes,categories=by_cat,
        paths_changed=paths_changed,predictions_changed=predictions_changed,reward_groups=reward_groups,
        effective_updates=sum(u["updated"] for u in progress["updates"]),boundaries=len(progress["updates"]),
        interruption=progress["restart"],resumed=len(progress["restores"]),rollouts_verified=len(index),
        training_token_limits=sum(r["stop_reason"]=="generation_token_limit" for stage,rows in records.items() if stage.startswith("train-") for r in rows),
        training_stops=Counter(r["stop_reason"] for stage,rows in records.items() if stage.startswith("train-") for r in rows),
        oversized_search_training_episodes=sum(n for stage,n in oversized_search.items() if stage.startswith("train-")),
        oversized_search_by_stage=oversized_search,
        selected_sha256=weights[selected],baseline_sha256=weights[0],skill_sha256=plan["skill"])
    verification["uses_skill_rewards"]="skill_reward.py" in progress["updates"][0]["contract"]
    (HERE/"评分.json").write_text(json.dumps(verification,ensure_ascii=False,indent=2),encoding="utf-8")
    report(verification)
    print(json.dumps(verification,ensure_ascii=False))


def main(directory=None):
    global HERE
    if directory is not None:
        HERE=Path(directory)
    score_evidence(*read_evidence(HERE/"evidence.tar.gz"))


if __name__=="__main__":
    main()

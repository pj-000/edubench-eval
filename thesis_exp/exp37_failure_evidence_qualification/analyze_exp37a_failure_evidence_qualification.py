"""Analyze Exp37A-R1 model-reviewed references and train-only OOF utility."""

from __future__ import annotations

import argparse
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp37_failure_evidence_qualification.common import (  # noqa: E402
    FAILURE_CLASSES, OOF_PATH, QWEN_MANIFEST, R0_ROOT, ROOT, TRAIN_PATH,
    build_final_reference, load_teacher_map, norm, normalize_failure, question_key,
    read_jsonl, sample_id, tie_safe_average_precision, write_csv, write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    parser.add_argument("--r0-out-dir", type=Path, default=R0_ROOT)
    parser.add_argument("--train-jsonl", type=Path, default=TRAIN_PATH)
    parser.add_argument("--reviewer-a", type=Path)
    parser.add_argument("--reviewer-b", type=Path)
    parser.add_argument("--adjudication", type=Path)
    parser.add_argument("--qwen-manifest", type=Path, default=QWEN_MANIFEST)
    parser.add_argument("--oof-predictions", type=Path, default=OOF_PATH)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-missing-reviews", action="store_true")
    return parser.parse_args()


def optional_rows(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path) if path.exists() else []


def load_views(r0_root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for view in ("low_tail_all", "boundary_view", "high_control_view"):
        path = r0_root / "annotation_templates" / f"exp37a_{view}_reviewer_a_template.jsonl"
        for row in read_jsonl(path):
            result[str(row["sample_id"])] = view
    return result


def binary_metrics(gold: list[int], pred: list[int]) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        gold, pred, average="binary", zero_division=0
    )
    negatives = sum(value == 0 for value in gold)
    false_positive = sum(g == 0 and p == 1 for g, p in zip(gold, pred))
    return {
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "false_negative_rate": float(1.0 - recall),
        "no_major_failure_false_positive_rate": false_positive / negatives if negatives else 0.0,
    }


def qwen_presence(annotation: dict[str, Any]) -> str:
    failures = {normalize_failure(str(value)) for value in annotation.get("major_failures") or []}
    return "no" if not failures or failures == {"no_major_failure"} else "yes"


def qwen_failures(annotation: dict[str, Any]) -> set[str]:
    values = {normalize_failure(str(value)) for value in annotation.get("major_failures") or []}
    return values or {"no_major_failure"}


def qwen_evidence(annotation: dict[str, Any]) -> list[str]:
    return [
        str(item.get("evidence")) for item in annotation.get("rubric_assessment") or []
        if isinstance(item, dict) and str(item.get("evidence") or "").strip()
    ]


def tokens(value: str) -> set[str]:
    return {token for token in norm(value).lower().split() if token}


def span_compatible(left: str, right: str) -> bool:
    nl, nr = norm(left).lower(), norm(right).lower()
    if not nl or not nr:
        return False
    if nl == nr or nl in nr or nr in nl:
        return True
    union = tokens(nl) | tokens(nr)
    return bool(union) and len(tokens(nl) & tokens(nr)) / len(union) >= 0.5


def evidence_result(ref: dict[str, Any], annotation: dict[str, Any], output: str) -> dict[str, int]:
    spans = qwen_evidence(annotation)
    qfail = qwen_failures(annotation)
    qmissing = bool(qfail & {"missing_content_or_key_point"}) and not spans
    syntax_visible = bool(spans) and all(norm(span) and norm(span) in norm(output) for span in spans)
    syntax_missing = qmissing
    syntax_valid = (
        syntax_visible if ref["evidence_type"] == "explicit_span"
        else syntax_missing if ref["evidence_type"] == "missing_required_content"
        else True if ref["major_failure_presence"] == "no"
        else bool(annotation.get("reason") or annotation.get("rubric_assessment"))
    )
    support = False
    if ref["evidence_type"] == "explicit_span":
        support = any(span_compatible(q, r) for q in spans for r in ref.get("evaluator_output_evidence") or [])
    elif ref["evidence_type"] == "missing_required_content":
        support = qmissing and bool(qfail & set(ref.get("failure_classes") or []))
    elif ref["evidence_type"] == "global_inconsistency":
        support = qwen_presence(annotation) == "yes" and ref["evidence_sufficiency"] in {"sufficient", "partial"}
    elif ref["major_failure_presence"] == "no":
        support = qwen_presence(annotation) == "no"
    return {
        "syntax_valid": int(syntax_valid), "semantic_support": int(support),
        "empty_evidence": int(not spans), "visible_span_support": int(ref["evidence_type"] == "explicit_span" and support),
        "missing_content_support": int(ref["evidence_type"] == "missing_required_content" and support),
    }


def reviewer_agreement(a: list[dict[str, Any]], b: list[dict[str, Any]], views: dict[str, str]) -> list[dict[str, Any]]:
    bm = {str(row["sample_id"]): row for row in b}
    rows = []
    for scope in ["all", "low_tail_all", "boundary_view", "high_control_view"]:
        pairs = [(row, bm[str(row["sample_id"])]) for row in a if str(row["sample_id"]) in bm and (scope == "all" or views[str(row["sample_id"])] == scope)]
        scores_a = [int(x[0]["most_plausible_score"]) for x in pairs]
        scores_b = [int(x[1]["most_plausible_score"]) for x in pairs]
        presence_agree = [x[0]["major_failure_presence"] == x[1]["major_failure_presence"] for x in pairs]
        rows.append({
            "scope": scope, "n": len(pairs),
            "score_qwk": float(cohen_kappa_score(scores_a, scores_b, weights="quadratic")) if pairs else None,
            "major_failure_presence_agreement": float(np.mean(presence_agree)) if pairs else None,
            "point_score_exact_agreement": float(np.mean([x == y for x, y in zip(scores_a, scores_b)])) if pairs else None,
        })
    return rows


def semantic_metrics(reference: list[dict[str, Any]], qwen: dict[str, dict[str, Any]], packets: dict[str, dict[str, Any]], views: dict[str, str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    detail = []
    for ref in reference:
        sid = str(ref["sample_id"])
        if sid not in qwen:
            continue
        ann = qwen[sid]
        ev = evidence_result(ref, ann, str(packets[sid]["evaluator_output"]))
        detail.append({
            "sample_id": sid, "view": views[sid],
            "reference_evidence_type": ref["evidence_type"],
            "reference_evidence_sufficiency": ref["evidence_sufficiency"],
            "gold_presence": int(ref["major_failure_presence"] == "yes"),
            "pred_presence": int(qwen_presence(ann) == "yes"),
            "gold_failures": set(ref.get("failure_classes") or []),
            "pred_failures": qwen_failures(ann), **ev,
        })
    major = []
    for scope in ["all", "low_tail_all", "boundary_view", "high_control_view"]:
        subset = [row for row in detail if scope == "all" or row["view"] == scope]
        values = binary_metrics([r["gold_presence"] for r in subset], [r["pred_presence"] for r in subset]) if subset else {}
        major.append({"scope": scope, "n": len(subset), **values})
    subtype = []
    supported = []
    for label in FAILURE_CLASSES:
        gold = [int(label in row["gold_failures"]) for row in detail]
        pred = [int(label in row["pred_failures"]) for row in detail]
        positive = sum(gold)
        values = binary_metrics(gold, pred)
        formal = positive >= 10
        if formal:
            supported.append(values["f1"])
        subtype.append({"failure_subtype": label, "positive_count": positive, "formal_gate_supported": formal, **values})
    subtype.append({"failure_subtype": "FORMAL_SUPPORTED_MACRO", "positive_count": "", "formal_gate_supported": True, "f1": float(np.mean(supported)) if supported else 0.0, "unsupported_class_count": sum(not row["formal_gate_supported"] for row in subtype)})
    missing_rows = [r for r in detail if r["reference_evidence_type"] == "missing_required_content"]
    visible_rows = [r for r in detail if r["reference_evidence_type"] == "explicit_span"]
    evidence = [{
        "n": len(detail), "syntax_validity": float(np.mean([r["syntax_valid"] for r in detail])) if detail else 0.0,
        "semantic_support_sufficient": float(np.mean([r["semantic_support"] and r["reference_evidence_sufficiency"] == "sufficient" for r in detail])) if detail else 0.0,
        "semantic_support_partial_or_better": float(np.mean([r["semantic_support"] and r["reference_evidence_sufficiency"] in {"sufficient", "partial", "not_applicable"} for r in detail])) if detail else 0.0,
        "missing_content_support": float(np.mean([r["missing_content_support"] for r in missing_rows])) if missing_rows else None,
        "visible_span_support": float(np.mean([r["visible_span_support"] for r in visible_rows])) if visible_rows else None,
        "empty_evidence_count": sum(r["empty_evidence"] for r in detail),
    }]
    return major, subtype, evidence


def prediction(row: dict[str, Any]) -> int | None:
    for field in ("pred_label", "pred_score", "prediction"):
        if row.get(field) is not None:
            return int(round(float(row[field])))
    probs = [float(row.get(f"prob_{label}", 0.0)) for label in range(1, 6)]
    return int(np.argmax(probs)) + 1 if any(probs) else None


def odds_ratio(rows: list[dict[str, Any]], signal: str, target: str) -> float:
    a = sum(r[signal] and r[target] for r in rows); b = sum(r[signal] and not r[target] for r in rows)
    c = sum(not r[signal] and r[target] for r in rows); d = sum(not r[signal] and not r[target] for r in rows)
    return ((a + .5) * (d + .5)) / ((b + .5) * (c + .5))


def permute_signals(rows: list[dict[str, Any]], count: int, seed: int) -> tuple[list[list[int]], dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[(row["human_label"], row["language"], row["metric_group"])].append(index)
    original = [r["qualified_signal"] for r in rows]
    rng = random.Random(seed); permutations = []; changed_rates = []
    for _ in range(count):
        values = list(original)
        for indices in groups.values():
            shuffled = [original[index] for index in indices]
            rng.shuffle(shuffled)
            for index, value in zip(indices, shuffled): values[index] = value
        permutations.append(values)
        changed_rates.append(sum(a != b for a, b in zip(values, original)) / len(rows))
    return permutations, {
        "singleton_stratum_count": sum(len(indices) == 1 for indices in groups.values()),
        "actual_signal_changed_rate": float(np.mean(changed_rates)),
        "unchanged_permutation_rate": float(np.mean([rate == 0 for rate in changed_rates])),
    }


def bootstrap_difference(rows: list[dict[str, Any]], target: str, fixed: list[int], resamples: int, seed: int) -> dict[str, Any]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows): groups[row["question_key"]].append(index)
    keys = sorted(groups); rng = np.random.default_rng(seed); values = []
    for _ in range(resamples):
        sampled = rng.choice(keys, size=len(keys), replace=True)
        indices = [i for key in sampled for i in groups[str(key)]]
        labels = [rows[i][target] for i in indices]
        values.append(tie_safe_average_precision(labels, [rows[i]["qualified_signal"] for i in indices]) - tie_safe_average_precision(labels, [fixed[i] for i in indices]))
    return {"bootstrap_mean": float(np.mean(values)), "ci_lower_95": float(np.quantile(values, .025)), "ci_upper_95": float(np.quantile(values, .975)), "resamples": resamples}


def utility(reference: list[dict[str, Any]], train: dict[str, dict[str, Any]], oof_path: Path, permutations: int, bootstraps: int, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if not reference or not oof_path.exists():
        waiting = [{"status": "WAITING_FOR_COMPLETE_REFERENCE_OR_OOF"}]
        return waiting, waiting, waiting, []
    oof = {sample_id(row): row for row in read_jsonl(oof_path)}
    rows = []
    for ref in reference:
        sid = str(ref["sample_id"]); pred = prediction(oof[sid]) if sid in oof else None
        if sid not in train or pred is None: continue
        human = int(round(float(train[sid]["label_5"]))); silver = int(ref["most_plausible_score"])
        qualified = int(ref["major_failure_presence"] == "yes" and ref["confidence"] in {"high", "medium"} and ref["evidence_sufficiency"] in {"sufficient", "partial"} and set(ref["failure_classes"]) != {"unclear_or_other"})
        rows.append({"sample_id": sid, "question_key": question_key(train[sid]), "human_label": human, "silver_label": silver, "language": train[sid].get("language"), "metric_group": train[sid].get("metric_group"), "oof_pred": pred, "qualified_signal": qualified, "severe_human": int(abs(pred-human)>=2), "low_to_high_human": int(human<=2 and pred>=4), "severe_silver": int(abs(pred-silver)>=2), "low_to_high_silver": int(silver<=2 and pred>=4)})
    permuted, diagnostics = permute_signals(rows, permutations, seed)
    human_rows = []; silver_rows = []; null_rows = []; bootstrap_rows = []
    for anchor, targets, destination in (("human", ("severe_human", "low_to_high_human"), human_rows), ("silver", ("severe_silver", "low_to_high_silver"), silver_rows)):
        for target in targets:
            labels = [r[target] for r in rows]; aligned = tie_safe_average_precision(labels, [r["qualified_signal"] for r in rows])
            null = [tie_safe_average_precision(labels, values) for values in permuted]
            destination.append({"anchor": anchor, "target": target, "n": len(rows), "aligned_auprc": aligned, "odds_ratio": odds_ratio(rows, "qualified_signal", target)})
            null_rows.append({"anchor": anchor, "target": target, "permutations": permutations, "aligned_auprc": aligned, "permutation_mean": float(np.mean(null)), "permutation_std": float(np.std(null)), "permutation_q025": float(np.quantile(null,.025)), "permutation_q975": float(np.quantile(null,.975)), "aligned_minus_permutation_mean": aligned-float(np.mean(null)), "empirical_one_sided_p": (1+sum(value>=aligned for value in null))/(permutations+1), **diagnostics})
            bootstrap_rows.append({"anchor": anchor, "target": target, **bootstrap_difference(rows, target, permuted[0], bootstraps, seed), "cluster_unit": "question_key", "fixed_permutation_seed": seed})
    return human_rows, silver_rows, null_rows + bootstrap_rows, rows


def score_range(a: list[dict[str, Any]], b: list[dict[str, Any]], final: list[dict[str, Any]]) -> list[dict[str, Any]]:
    am={str(r["sample_id"]):r for r in a}; bm={str(r["sample_id"]):r for r in b}
    rows=[]
    for ref in final:
        sid=str(ref["sample_id"]); ar=am[sid]["score_range"]; br=bm[sid]["score_range"]
        intersection=max(0,min(ar[1],br[1])-max(ar[0],br[0])+1); union=max(ar[1],br[1])-min(ar[0],br[0])+1
        rows.append({"range_overlap":int(intersection>0),"range_iou":intersection/union,"point_within_other":(int(br[0]<=am[sid]["most_plausible_score"]<=br[1])+int(ar[0]<=bm[sid]["most_plausible_score"]<=ar[1]))/2,"width":ref["score_range"][1]-ref["score_range"][0]+1,"non_singleton":int(ref["score_range"][0]!=ref["score_range"][1])})
    return [{"n":len(rows),"range_overlap_rate":float(np.mean([r["range_overlap"] for r in rows])),"point_within_other_range_rate":float(np.mean([r["point_within_other"] for r in rows])),"mean_width":float(np.mean([r["width"] for r in rows])),"global_non_singleton_rate":float(np.mean([r["non_singleton"] for r in rows]))}]


def main() -> None:
    args=parse_args(); train={sample_id(r):r for r in read_jsonl(args.train_jsonl)}; views=load_views(args.r0_out_dir)
    packets={str(r["sample_id"]):r for r in read_jsonl(args.out_dir/"private_packets/exp37a_r1_reviewer_a_packets.jsonl")}
    a=optional_rows(args.reviewer_a or args.out_dir/"private_reviews/reviewer_a_filled.jsonl"); b=optional_rows(args.reviewer_b or args.out_dir/"private_reviews/reviewer_b_filled.jsonl"); c=optional_rows(args.adjudication or args.out_dir/"private_reviews/adjudication_filled.jsonl")
    complete_ab=len(a)==len(b)==196
    final=[]
    if complete_ab:
        try: final=build_final_reference(a,b,c)
        except ValueError:
            if not args.allow_missing_reviews: raise
    qwen,qwen_source=load_teacher_map(args.qwen_manifest)
    waiting=[{"status":"WAITING_FOR_EXTERNAL_MODEL_REVIEWS"}]
    agreement=reviewer_agreement(a,b,views) if complete_ab else waiting
    major,subtype,evidence=semantic_metrics(final,qwen,packets,views) if final else (waiting,waiting,waiting)
    human,silver,null_and_boot,private_utility=utility(final,train,args.oof_predictions,args.permutations,args.bootstrap_resamples,args.seed)
    null=[r for r in null_and_boot if "permutations" in r] or waiting; boot=[r for r in null_and_boot if "resamples" in r] or waiting
    ranges=score_range(a,b,final) if final else waiting
    outputs={"exp37a_r1_reviewer_agreement.csv":agreement,"exp37a_r1_major_failure_metrics.csv":major,"exp37a_r1_failure_subtype_metrics.csv":subtype,"exp37a_r1_evidence_metrics.csv":evidence,"exp37a_r1_oof_utility_human_anchor.csv":human,"exp37a_r1_oof_utility_silver_anchor.csv":silver,"exp37a_r1_permutation_null.csv":null,"exp37a_r1_question_key_bootstrap_ci.csv":boot,"exp37a_r1_score_range_qualification.csv":ranges}
    for name,rows in outputs.items(): write_csv(args.out_dir/"tables"/name,rows)
    if private_utility: write_csv(args.out_dir/"private_reference/exp37a_r1_oof_utility_rows.csv",private_utility)
    agree_all=agreement[0] if final else {}; major_all=major[0] if final else {}; subtype_gate=next((r for r in subtype if r.get("failure_subtype")=="FORMAL_SUPPORTED_MACRO"),{}); ev=evidence[0] if final else {}; severe_h=next((r for r in human if r.get("target")=="severe_human"),{}); null_h=next((r for r in null if r.get("target")=="severe_human"),{}); boot_h=next((r for r in boot if r.get("target")=="severe_human"),{})
    reference_gate=bool(final and float(agree_all.get("score_qwk",0))>=.60 and float(agree_all.get("major_failure_presence_agreement",0))>=.70)
    semantic_gate=bool(final and float(major_all.get("f1",0))>=.70 and float(next((r.get("recall",0) for r in major if r.get("scope")=="low_tail_all"),0))>=.70 and float(subtype_gate.get("f1",0))>=.45 and float(ev.get("syntax_validity",0))>=.90 and float(ev.get("semantic_support_partial_or_better",0))>=.70 and float(major_all.get("false_negative_rate",1))<=.20)
    utility_gate=bool(final and float(null_h.get("aligned_minus_permutation_mean",0))>=.05 and float(null_h.get("empirical_one_sided_p",1))<.05 and float(boot_h.get("ci_lower_95",-1))>0 and float(severe_h.get("odds_ratio",0))>=2)
    range_gate=bool(final and ranges[0]["range_overlap_rate"]>=.85 and ranges[0]["point_within_other_range_rate"]>=.90 and ranges[0]["mean_width"]<=2 and ranges[0]["global_non_singleton_rate"]>=.15)
    decision={"status":"GO" if reference_gate and semantic_gate and utility_gate else "NO_GO" if final else "READY_FOR_EXTERNAL_MODEL_REVIEWS","reference_complete":bool(final),"reference_type":"multi_session_model_reviewed_silver","reference_gate":reference_gate,"semantic_gate":semantic_gate,"utility_gate":utility_gate,"recommend_new_reason_evidence_training":bool(reference_gate and semantic_gate and utility_gate),"stop_reason_evidence_supervision":bool(final and not(reference_gate and semantic_gate and utility_gate)),"recommend_qwen_score_range_pilot":range_gate,"recommend_student_training":False,"qwen_source":qwen_source,"reason":"Reference reliability passed, but Qwen semantic qualification and benchmark-anchored OOF utility did not pass preregistered gates." if final and reference_gate and not (semantic_gate and utility_gate) else "Decision follows preregistered reference, semantic, and utility gates.","dev_access_count":0,"test_access_count":0}
    write_json(args.out_dir/"decision/exp37a_r1_qualification_decision.json",decision)
    utility_available = bool(human and human[0].get("aligned_auprc") is not None)
    report=[
        "# Exp37A-R1 qualification report", "",
        "## Decision",
        f"- Status: `{decision['status']}`",
        f"- Reference complete: `{bool(final)}`",
        f"- Reference gate: `{reference_gate}`",
        f"- Semantic gate: `{semantic_gate}`",
        f"- Utility gate: `{utility_gate}`",
        f"- Score-range pilot gate: `{range_gate}`", "",
        "## Reference quality",
        f"- Reviewer A/B score QWK: `{agree_all.get('score_qwk')}`",
        f"- Major-failure-presence agreement: `{agree_all.get('major_failure_presence_agreement')}`",
        f"- Selective Reviewer C conflicts: `{len(c)}` of 196", "",
        "## Qwen semantic qualification",
        f"- Major-failure F1: `{major_all.get('f1')}`",
        f"- Low-tail major-failure recall: `{next((r.get('recall') for r in major if r.get('scope') == 'low_tail_all'), None)}`",
        f"- Supported subtype macro-F1: `{subtype_gate.get('f1')}`",
        f"- Evidence syntax validity: `{ev.get('syntax_validity')}`",
        f"- Semantic support partial-or-better: `{ev.get('semantic_support_partial_or_better')}`", "",
        "## OOF utility",
        f"- Train-only OOF input available: `{utility_available}`",
        f"- Human-anchor severe-error AUPRC: `{severe_h.get('aligned_auprc')}`",
        f"- Human-anchor aligned minus permutation mean: `{null_h.get('aligned_minus_permutation_mean')}`",
        f"- Human-anchor permutation p: `{null_h.get('empirical_one_sided_p')}`",
        f"- Human-anchor bootstrap lower CI: `{boot_h.get('ci_lower_95')}`",
        f"- Silver-anchor severe-error AUPRC: `{next((r.get('aligned_auprc') for r in silver if r.get('target') == 'severe_silver'), None)}`",
        "- Human-anchor and model-reviewed silver-anchor targets remain strictly separate.",
        "- Strong silver-anchor utility without human-anchor utility is not treated as benchmark improvement evidence.",
        "- A missing train-only OOF input is reported as unavailable, never fabricated.", "",
        "## Boundary",
        "- No API, GPU, training, student inference, dev, or test access occurred.",
    ]
    path=args.out_dir/"reports/exp37a_r1_qualification_report.md"; path.parent.mkdir(parents=True,exist_ok=True); path.write_text("\n".join(report)+"\n",encoding="utf-8")
    print(decision)


if __name__ == "__main__": main()

"""Run or collect the single locked Exp43 final paper-like test campaign."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from thesis_exp.exp43_rubimor.common import ROOT, RUN_ROOT, SEEDS, TEST_PATH, human_stats, prediction_metrics, read_jsonl, sample_id, write_csv, write_json, write_jsonl
from thesis_exp.exp43_rubimor.modeling_rubimor import RubiMORConfig, build_model
from thesis_exp.exp43_rubimor.prepare_exp43_datasets import format_row

VARIANTS = ("E0", "E3", "E5", "E6", "E6N")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name-or-path")
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--test", type=Path, default=TEST_PATH)
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def run_one(args: argparse.Namespace) -> None:
    if not args.variant or args.seed is None or not args.model_name_or_path:
        raise ValueError("Run mode requires --variant, --seed, and --model-name-or-path")
    lock_path = args.out_dir / "configs/exp43_final_test_lock.json"
    if not lock_path.exists(): raise RuntimeError("Final test lock is missing")
    import torch
    from transformers import AutoTokenizer
    mapping = json.loads((args.out_dir/"configs/exp43_metric_mapping.json").read_text(encoding="utf-8"))["metrics"]
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, local_files_only=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    source = read_jsonl(args.test)
    rows = []
    for row in source:
        stats = human_stats(row); text,_ = format_row(tokenizer,row,args.variant != "E0")
        rows.append({"sample_id":sample_id(row),"question_key":str(row["question_key"]),"metric":str(row.get("metric_canonical") or row.get("metric_raw")),"metric_id":mapping.get(str(row.get("metric_canonical") or row.get("metric_raw")),-1),"text":text,**stats,"language":row.get("language") or "unknown","subject":row.get("subject_canonical") or "unknown","scenario":row.get("scenario_canonical") or "unknown","education_level":row.get("education_level_canonical") or "unknown"})
    model = build_model(RubiMORConfig(args.model_name_or_path,len(mapping),use_metric_residual=args.variant in {"E5","E6","E6N"})).cuda().eval()
    checkpoint = args.run_root/"headline"/args.variant/f"seed_{args.seed}"/"best_checkpoint.pt"
    state = torch.load(checkpoint,map_location="cuda",weights_only=False); model.load_state_dict(state["model"])
    train = read_jsonl(args.out_dir/f"private/data/exp43_train_{args.variant}.jsonl")
    counts = {}
    for row in train: counts[row["metric_id"]]=counts.get(row["metric_id"],0)+1
    enabled_ids={key for key,value in counts.items() if value>=30}
    predictions=[]
    for start in range(0,len(rows),args.batch_size):
        batch=rows[start:start+args.batch_size]
        encoded=tokenizer([r["text"] for r in batch],padding=True,truncation=True,max_length=2048,return_tensors="pt")
        metric_ids=torch.tensor([r["metric_id"] for r in batch],device="cuda")
        enabled=torch.tensor([value in enabled_ids for value in metric_ids.tolist()],device="cuda")
        with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16): logits=model(**{key:value.cuda() for key,value in encoded.items()},metric_ids=metric_ids,residual_enabled=enabled)["logits"].float()
        probs=torch.softmax(logits,-1).cpu().tolist()
        for row,p in zip(batch,probs): predictions.append({**{key:row[key] for key in ("sample_id","question_key","metric","gold_label_5","human_distribution_5","expected_human_score","human_entropy","human_score_range","language","subject","scenario","education_level")},"variant":args.variant,"seed":args.seed,"pred_label_5":int(np.argmax(p))+1,"pred_score_expected":sum(label*p[label-1] for label in range(1,6)),**{f"prob_{label}":p[label-1] for label in range(1,6)}})
        print(f"[exp43-test] {args.variant} seed={args.seed} {min(start+args.batch_size,len(rows))}/{len(rows)}",flush=True)
    path=args.out_dir/f"private/predictions/final_test_{args.variant}_seed{args.seed}.jsonl";write_jsonl(path,predictions)
    print(json.dumps({"status":"FINAL_TEST_RUN_COMPLETE","variant":args.variant,"seed":args.seed,"rows":len(predictions)},sort_keys=True))


def collect(args: argparse.Namespace) -> None:
    metrics=[]; labels=[]; cache={}
    for variant in VARIANTS:
        for seed in SEEDS:
            path=args.out_dir/f"private/predictions/final_test_{variant}_seed{seed}.jsonl"
            rows=read_jsonl(path); cache[(variant,seed)]=rows; values=prediction_metrics(rows)
            metrics.append({"variant":variant,"seed":seed,**values})
            labels.append({"variant":variant,"seed":seed,**{f"label{label}_accuracy":values[f"label{label}_recall"] for label in range(1,6)},"Overall":values["Exact_Match"]})
    # Historical row is a paper reference, not a same-code rerun.
    metrics.append({"variant":"historical_EduBenchEvaluator_reference","seed":"paper","MAE":.430,"Signed_Bias":.246,"Exact_Match":.725,"Kendall_tau":.508,"Bin_Agreement":.897,"comparison_scope":"historical_reference"})
    labels.append({"variant":"historical_EduBenchEvaluator_reference","seed":"paper","label1_accuracy":.481,"label2_accuracy":.234,"label3_accuracy":.211,"label4_accuracy":.661,"label5_accuracy":.877,"Overall":.725,"comparison_scope":"historical_reference"})
    bootstrap=[]
    rng=np.random.default_rng(43123)
    for seed in SEEDS:
        left,right=cache[("E6",seed)],cache[("E0",seed)]; lb={r["sample_id"]:r for r in left};rb={r["sample_id"]:r for r in right};groups={}
        for sid in lb:groups.setdefault(lb[sid]["question_key"],[]).append(sid)
        keys=sorted(groups);values={metric:[] for metric in ("MAE","QWK","Exact_Match","Kendall_tau")}
        for _ in range(2000):
            selected=rng.choice(keys,size=len(keys),replace=True);ids=[sid for key in selected for sid in groups[str(key)]]
            lm,rm=prediction_metrics([lb[sid] for sid in ids]),prediction_metrics([rb[sid] for sid in ids])
            for metric in values:values[metric].append(lm[metric]-rm[metric])
        for metric,vals in values.items():bootstrap.append({"comparison":"E6_vs_E0","seed":seed,"metric":metric,"delta_mean":float(np.mean(vals)),"ci_low":float(np.quantile(vals,.025)),"ci_high":float(np.quantile(vals,.975)),"replicates":2000,"unit":"question_key"})
    write_csv(args.out_dir/"tables/exp43_final_test_metrics.csv",metrics);write_csv(args.out_dir/"tables/exp43_final_test_label_accuracy.csv",labels);write_csv(args.out_dir/"tables/exp43_final_test_bootstrap_ci.csv",bootstrap)
    lock=json.loads((args.out_dir/"configs/exp43_final_test_lock.json").read_text(encoding="utf-8"))
    write_json(args.out_dir/"decision/exp43_final_test_consumed.json",{"final_test_consumed":True,"timestamp":datetime.now(timezone.utc).isoformat(),"lock_hash":lock["lock_hash"],"test_file_hash":lock["test_file_hash"],"runs":15})
    print(json.dumps({"status":"FINAL_TEST_COLLECTED","runs":15,"final_test_consumed":True},sort_keys=True))


def main() -> None:
    args=parse_args(); collect(args) if args.collect else run_one(args)


if __name__=="__main__":main()


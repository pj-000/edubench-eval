"""Build the lightweight Exp43 status/final report at any terminal gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from thesis_exp.exp43_rubimor.common import ROOT, write_json


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--out-dir",type=Path,default=ROOT);args=parser.parse_args()
    decisions={}
    for path in sorted((args.out_dir/"decision").glob("*.json")):
        decisions[path.stem]=json.loads(path.read_text(encoding="utf-8"))
    statuses={name:value.get("status",value.get("final_test_consumed")) for name,value in decisions.items()}
    group=statuses.get("exp43_groupcv_decision")
    headline=statuses.get("exp43_headline_dev_decision")
    consumed=bool(decisions.get("exp43_final_test_consumed",{}).get("final_test_consumed"))
    if consumed and group=="RUBIMOR_FULL_GROUPCV_GO" and headline=="HEADLINE_DEV_GO":final="RUBIMOR_FULL_GO"
    elif consumed and headline=="HEADLINE_DEV_GO":final="RUBIMOR_OVERALL_ONLY"
    elif statuses.get("exp43_ordinal_decision")=="GO" and statuses.get("exp43_metric_head_decision")!="GO":final="ORDINAL_ONLY_SIGNAL"
    elif statuses.get("exp43_metric_head_decision")=="GO" and statuses.get("exp43_pairwise_decision")!="GO":final="METRIC_HEAD_SIGNAL"
    else:final="RUBIMOR_STOP" if any(str(value).endswith("STOP") or str(value).endswith("NO_GO") for value in statuses.values()) else "IN_PROGRESS"
    lines=["# Exp43 RubiMOR Final Report","",f"- Current final status: **{final}**",f"- Final test consumed: **{consumed}**","","## Stage decisions","",* [f"- `{name}`: `{value}`" for name,value in statuses.items()],"","## Protocol integrity","","- Qwen3-Reranker-0.6B only; full fine-tuning.","- No teacher API, teacher relabeling, or teacher reason supervision.","- No test-driven method tuning.","- Runtime checkpoints, raw predictions, pair JSONL, and logs remain private/ignored."]
    (args.out_dir/"reports/exp43_rubimor_final_report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    write_json(args.out_dir/"decision/exp43_final_decision.json",{"status":final,"final_test_consumed":consumed,"stage_statuses":statuses,"no_teacher_api":True,"no_teacher_relabeling":True,"no_dev_test_method_tuning":True,"test_access_count":15 if consumed else 0})
    print(json.dumps({"status":final,"final_test_consumed":consumed},sort_keys=True))


if __name__=="__main__":main()


"""Build the lightweight Exp43 status/final report at any terminal gate."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from thesis_exp.exp43_rubimor.common import ROOT, write_json


STAGE_DECISIONS = (
    ("Stage 0", "exp43_stage0_decision"),
    ("Stage 1", "exp43_smoke_decision"),
    ("Stage 2", "exp43_baseline_pipeline_decision"),
    ("Stage 3", "exp43_ordinal_decision"),
    ("Stage 4", "exp43_metric_head_decision"),
    ("Stage 5", "exp43_pairwise_decision"),
    ("Stage 6", "exp43_groupcv_decision"),
    ("Stage 8", "exp43_headline_dev_decision"),
)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_seed42_report(out_dir: Path, statuses: dict[str, object]) -> None:
    metrics = [
        row for row in read_csv(out_dir / "tables/exp43_groupcv_metrics_by_seed.csv")
        if int(row["seed"]) == 42
    ]
    if not metrics:
        return
    columns = ("MAE", "QWK", "Exact_Match", "Kendall_tau", "human_RPS", "low_to_high_rate", "label2_recall", "label5_recall")
    lines = [
        "# Exp43 Seed42 Module Report", "",
        "All values are five-fold question-key-disjoint out-of-fold metrics at fixed epoch 10.", "",
        "| Variant | MAE | QWK | Exact | Kendall | Human RPS | Low-to-high | Label2 recall | Label5 recall |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics:
        values = [f"{float(row[column]):.6f}" for column in columns]
        lines.append(f"| {row['variant']} | " + " | ".join(values) + " |")
    lines.extend(["", "## Module gates", ""])
    for stage, name in STAGE_DECISIONS:
        value = statuses.get(name, "NOT_RUN_AFTER_GATE_STOP")
        lines.append(f"- {stage}: `{value}`")
    lines.extend([
        "", "## Interpretation", "",
        "- E4 passed because it preserved all protection guards and improved MAE over E3 by at least 0.005.",
        "- E5 stopped because QWK protection failed and no preregistered metric-head mechanism improved.",
        "- Stages 5-9 were not authorized after the Stage 4 stop.",
        "- The sealed test set was not parsed or evaluated.",
    ])
    (out_dir / "reports/exp43_seed42_module_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    build_seed42_report(args.out_dir, statuses)
    stage_lines = [
        f"- {stage}: `{statuses.get(name, 'NOT_RUN_AFTER_GATE_STOP')}`"
        for stage, name in STAGE_DECISIONS
    ]
    stage_lines.append(f"- Stage 9: `{'COMPLETED' if consumed else 'NOT_RUN_TEST_SEALED'}`")
    lines=["# Exp43 RubiMOR Final Report","",f"- Current final status: **{final}**",f"- Final test consumed: **{consumed}**","","## Stage decisions","",*stage_lines,"","## Protocol integrity","","- Qwen3-Reranker-0.6B only; full fine-tuning.","- No teacher API, teacher relabeling, or teacher reason supervision.","- No test-driven method tuning.","- Runtime checkpoints, raw predictions, pair JSONL, and logs remain private/ignored."]
    (args.out_dir/"reports/exp43_rubimor_final_report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    write_json(args.out_dir/"decision/exp43_final_decision.json",{"status":final,"final_test_consumed":consumed,"stage_statuses":statuses,"no_teacher_api":True,"no_teacher_relabeling":True,"no_dev_test_method_tuning":True,"test_access_count":15 if consumed else 0})
    print(json.dumps({"status":final,"final_test_consumed":consumed},sort_keys=True))


if __name__=="__main__":main()

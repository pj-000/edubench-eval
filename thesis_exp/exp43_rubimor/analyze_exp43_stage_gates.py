"""Apply preregistered Exp43 stage gates without threshold changes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from thesis_exp.exp43_rubimor.common import ROOT, read_csv, write_json


def number(row: dict[str, Any], key: str) -> float:
    return float(row[key])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "stage2", "stage3", "stage4", "stage5", "stage6"), required=True)
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    return parser.parse_args()


def available(root: Path) -> dict[tuple[str, int], dict[str, str]]:
    return {(row["variant"], int(row["seed"])): row for row in read_csv(root / "tables/exp43_groupcv_metrics_by_seed.csv")}


def guard(current: dict, base: dict, *, mae=.01, qwk=.015, exact=.01) -> dict[str, bool]:
    return {
        "mae_guard": number(current, "MAE") <= number(base, "MAE") + mae,
        "qwk_guard": number(current, "QWK") >= number(base, "QWK") - qwk,
        "exact_guard": number(current, "Exact_Match") >= number(base, "Exact_Match") - exact,
        "label5_guard": number(current, "label5_recall") >= number(base, "label5_recall") - .03,
        "high_to_low_guard": number(current, "high_to_low_rate") <= number(base, "high_to_low_rate") + .02,
    }


def pair_accuracy(root: Path, variant: str, seed: int) -> float:
    rows = [row for row in read_csv(root / "tables/exp43_groupcv_pair_metrics.csv") if row["variant"] == variant and int(row["seed"]) == seed and row.get("heldout_pairs")]
    denominator = sum(int(row["heldout_pairs"]) for row in rows)
    return sum(int(row["heldout_pairs"]) * float(row["pair_accuracy"]) for row in rows) / denominator if denominator else float("nan")


def metric_summary(root: Path, variant: str, seed: int) -> tuple[float, float]:
    rows = [row for row in read_csv(root / "tables/exp43_groupcv_stratified_metrics.csv") if row["variant"] == variant and int(row["seed"]) == seed and row["stratum_type"] == "metric"]
    maes = sorted(float(row["MAE"]) for row in rows)
    return sum(maes) / len(maes), sum(maes[-max(1, math.ceil(len(maes) / 4)):]) / max(1, math.ceil(len(maes) / 4))


def main() -> None:
    args = parse_args(); root = args.out_dir
    if args.stage == "smoke":
        summaries = []
        for variant in ("E0", "E4", "E5", "E6", "E6N"):
            path = Path("thesis_exp/runs/exp43_rubimor/smoke") / variant / "seed_42" / "fold_0/run_summary.json"
            if not path.exists():
                summaries.append({"variant": variant, "valid": False, "reason": "missing"}); continue
            row = json.loads(path.read_text(encoding="utf-8"))
            valid = row.get("status") == "COMPLETED" and row.get("smoke_save_reload") == "PASS" and row.get("nan_count") == 0 and row.get("oom_count") == 0 and row.get("test_access_count") == 0
            if variant == "E6N": valid = valid and .48 <= float(row.get("pair_flip_rate", -1)) <= .52
            summaries.append({"variant": variant, "valid": valid})
        status = "GO" if all(row["valid"] for row in summaries) else "SMOKE_NO_GO"
        write_json(root / "decision/exp43_smoke_decision.json", {"status": status, "runs": summaries, "test_access_count": 0})
        print(json.dumps({"status": status, "runs": summaries}, sort_keys=True)); return
    rows = available(root); seed = 42
    checks: dict[str, bool] = {}; detail: dict[str, Any] = {}
    if args.stage == "stage2":
        e0, e3 = rows[("E0", seed)], rows[("E3", seed)]
        checks = {"mae_guard": number(e3, "MAE") <= number(e0, "MAE") + .02, "qwk_guard": number(e3, "QWK") >= number(e0, "QWK") - .03, "exact_guard": number(e3, "Exact_Match") >= number(e0, "Exact_Match") - .02}
        status, path = ("GO" if all(checks.values()) else "BASE_PIPELINE_STOP"), root / "decision/exp43_baseline_pipeline_decision.json"
    elif args.stage == "stage3":
        base, current = rows[("E3", seed)], rows[("E4", seed)]; checks = guard(current, base)
        mechanisms = {"rps": number(base, "human_RPS") - number(current, "human_RPS") >= .001, "expected_mae": number(base, "expected_score_MAE") - number(current, "expected_score_MAE") >= .003, "mae": number(base, "MAE") - number(current, "MAE") >= .005, "qwk": number(current, "QWK") - number(base, "QWK") >= .01, "kendall": number(current, "Kendall_tau") - number(base, "Kendall_tau") >= .0075}
        checks["mechanism_improvement"] = any(mechanisms.values()); detail["mechanisms"] = mechanisms
        status, path = ("GO" if all(checks.values()) else "ORDINAL_MODULE_STOP"), root / "decision/exp43_ordinal_decision.json"
    elif args.stage == "stage4":
        base, current = rows[("E4", seed)], rows[("E5", seed)]; checks = guard(current, base)
        base_macro, base_worst = metric_summary(root, "E4", seed); current_macro, current_worst = metric_summary(root, "E5", seed)
        mechanisms = {"macro_metric_mae": base_macro-current_macro >= .003, "worst_quartile_metric_mae": base_worst-current_worst >= .005, "overall_mae": number(base,"MAE")-number(current,"MAE") >= .003, "overall_qwk": number(current,"QWK")-number(base,"QWK") >= .005}
        checks["mechanism_improvement"] = any(mechanisms.values())
        residuals = [row for row in read_csv(root / "tables/exp43_metric_residual_norms.csv") if row["variant"] == "E5" and int(row["seed"]) == seed]
        total = sum(float(row["effective_residual_norm"]) for row in residuals)
        max_share = max((float(row["effective_residual_norm"]) / total for row in residuals), default=0.0)
        checks["residual_concentration_guard"] = max_share <= .40; detail.update({"mechanisms": mechanisms, "max_residual_norm_share": max_share})
        status, path = ("GO" if all(checks.values()) else "METRIC_HEAD_STOP"), root / "decision/exp43_metric_head_decision.json"
    elif args.stage == "stage5":
        base, current, noise = rows[("E5", seed)], rows[("E6", seed)], rows[("E6N", seed)]; checks = guard(current, base)
        base_pair, current_pair, noise_pair = pair_accuracy(root,"E5",seed), pair_accuracy(root,"E6",seed), pair_accuracy(root,"E6N",seed)
        mechanisms = {"heldout_pair_accuracy": current_pair-base_pair >= .03, "kendall": number(current,"Kendall_tau")-number(base,"Kendall_tau") >= .005, "low_to_high_relative": (number(base,"low_to_high_rate")-number(current,"low_to_high_rate"))/max(number(base,"low_to_high_rate"),1e-12) >= .05, "mae": number(base,"MAE")-number(current,"MAE") >= .003, "qwk": number(current,"QWK")-number(base,"QWK") >= .005}
        noise_controls = {"mae": number(noise,"MAE")-number(current,"MAE") >= .003, "qwk": number(current,"QWK")-number(noise,"QWK") >= .005, "pair_accuracy": current_pair-noise_pair >= .03}
        checks["mechanism_improvement"] = any(mechanisms.values()); checks["beats_E6N"] = any(noise_controls.values()); detail.update({"mechanisms": mechanisms, "noise_control": noise_controls, "pair_accuracy": {"E5": base_pair, "E6": current_pair, "E6N": noise_pair}})
        status, path = ("GO" if all(checks.values()) else "PAIRWISE_MODULE_STOP"), root / "decision/exp43_pairwise_decision.json"
    else:
        crossed = {(row["comparison"], row["metric"]): row for row in read_csv(root / "tables/exp43_groupcv_crossed_bootstrap_ci.csv")}
        def favorable_ci(comparison: str, metric: str) -> bool:
            row = crossed[(comparison, metric)]
            return float(row["ci_high"]) < 0 if metric == "MAE" else float(row["ci_low"]) > 0
        per_seed = [row for row in read_csv(root / "tables/exp43_groupcv_pairwise_differences.csv") if row["comparison"] == "E6_vs_E0"]
        e0_mean = next(row for row in read_csv(root / "tables/exp43_groupcv_multiseed_summary.csv") if row["variant"] == "E0")
        e6_mean = next(row for row in read_csv(root / "tables/exp43_groupcv_multiseed_summary.csv") if row["variant"] == "E6")
        primary = {"MAE": float(e0_mean["MAE_mean"])-float(e6_mean["MAE_mean"]) >= .015 and favorable_ci("E6_vs_E0","MAE"), "QWK": float(e6_mean["QWK_mean"])-float(e0_mean["QWK_mean"]) >= .020 and favorable_ci("E6_vs_E0","QWK"), "Kendall": float(e6_mean["Kendall_tau_mean"])-float(e0_mean["Kendall_tau_mean"]) >= .015 and favorable_ci("E6_vs_E0","Kendall_tau")}
        incremental = {"MAE": float(crossed[("E6_vs_E3","MAE")]["delta_mean"]) <= -.005, "QWK": float(crossed[("E6_vs_E3","QWK")]["delta_mean"]) >= .010, "Kendall": float(crossed[("E6_vs_E3","Kendall_tau")]["delta_mean"]) >= .0075}
        noise = {"MAE": float(crossed[("E6_vs_E6N","MAE")]["delta_mean"]) <= -.003, "QWK": float(crossed[("E6_vs_E6N","QWK")]["delta_mean"]) >= .005}
        protection = {"exact": float(e6_mean["Exact_Match_mean"]) >= float(e0_mean["Exact_Match_mean"])-.005, "bin": float(e6_mean["Bin_Agreement_mean"]) >= float(e0_mean["Bin_Agreement_mean"])-.005, "bias": float(e6_mean["abs_Signed_Bias_mean"]) <= float(e0_mean["abs_Signed_Bias_mean"])+.01, "label5": float(e6_mean["label5_recall_mean"]) >= float(e0_mean["label5_recall_mean"])-.02, "high_to_low": float(e6_mean["high_to_low_rate_mean"]) <= float(e0_mean["high_to_low_rate_mean"])+.01}
        seed_favorable = sum(float(row["delta_MAE"]) <= 0 or float(row["delta_QWK"]) >= 0 or float(row["delta_Kendall_tau"]) >= 0 for row in per_seed) >= 2
        low_tail = ((float(e6_mean["label1_recall_mean"])+float(e6_mean["label2_recall_mean"]))-(float(e0_mean["label1_recall_mean"])+float(e0_mean["label2_recall_mean"]))) / 2 >= .05 and (float(e0_mean["low_to_high_rate_mean"])-float(e6_mean["low_to_high_rate_mean"]))/max(float(e0_mean["low_to_high_rate_mean"]),1e-12) >= .10 and float(e0_mean["Signed_Bias_mean"])-float(e6_mean["Signed_Bias_mean"]) <= .10
        checks = {"primary": any(primary.values()), "incremental": any(incremental.values()), "beats_E6N": any(noise.values()), "protection": all(protection.values()), "seed_stability": seed_favorable}
        full = all(checks.values()) and low_tail; overall = all(checks.values())
        status = "RUBIMOR_FULL_GROUPCV_GO" if full else ("RUBIMOR_OVERALL_GROUPCV_GO" if overall else "RUBIMOR_GROUPCV_STOP")
        path = root / "decision/exp43_groupcv_decision.json"; detail.update({"primary": primary, "incremental": incremental, "noise_control": noise, "protection": protection, "low_tail_full_claim": low_tail})
    write_json(path, {"stage": args.stage, "status": status, "checks": checks, **detail, "test_access_count": 0})
    print(json.dumps({"stage": args.stage, "status": status, "checks": checks}, sort_keys=True))


if __name__ == "__main__":
    main()


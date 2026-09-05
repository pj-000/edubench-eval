"""CPU-only, paired cluster analysis of frozen Exp54 predictions.

No original collector, inference protocol, checkpoint, or result is modified.
See STATISTICAL_CLOSURE_PROTOCOL.md for the estimand and evidence boundaries.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = Path("thesis_exp/outputs/exp54_rar_sft/rar_v2")
SEEDS = (42, 43, 44)
REPLICATES = 10_000
RNG_SEED = 20260905
MIN_VALID = 9500
PRIMARY = ("MAE", "L2H_rate")
ALIASES = {"Kendall": "Kendall_tau_b", "Recall_2": "Label_2_Recall",
           "Recall_5": "Label_5_Recall"}
FAMILIES = {
    "SFT_DEV": ("dev", (("SFT_VS_SCORE", "S0", "R3"),
                           ("CONSISTENT_VS_ALL", "R1", "R3"),
                           ("ALIGNED_VS_SHUFFLED", "R2", "R3"))),
    "PREFERENCE_DEV": ("dev", (("DEV_FIELD", "R3", "P1_FIELD_DPO"),
         ("DEV_OFFSET", "P1_FIELD_DPO", "P2_SORC_SCORE"),
         ("DEV_JOINT", "P2_SORC_SCORE", "P3_JOINT_SORC"),
         ("DEV_JOINT_VS_SFT", "R3", "P3_JOINT_SORC"))),
    "FORMAL_TEST": ("test", (("H1_FIELD_DPO", "P0_R3_SFT", "P1_FIELD_DPO"),
         ("H2_ORDINAL_OFFSET", "P1_FIELD_DPO", "P2_SORC_SCORE"),
         ("H3_RATIONALE_BLOCK", "P2_SORC_SCORE", "P3_JOINT_SORC"))),
    "MECHANISM_TEST": ("test", (("C1_BLOCK_BALANCED_SFT", "R3_TOKENAVG", "P0_R3_SFT"),
         ("C2_FIELD_LOCAL_DPO", "P1_FULLSEQ", "P1_FIELD_DPO"),
         ("C3_ACTUAL_ERROR_NEGATIVES", "P1_SYN_LR5E6", "P1_FIELD_DPO"))),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def json_bytes(path: Path) -> tuple[Any, str]:
    payload = path.read_bytes()
    return json.loads(payload), hashlib.sha256(payload).hexdigest()


def jsonl_bytes(path: Path) -> tuple[list[dict], str]:
    payload = path.read_bytes()
    return [json.loads(x) for x in payload.splitlines() if x.strip()], hashlib.sha256(payload).hexdigest()


def finite_ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    shape = np.broadcast_shapes(np.shape(a), np.shape(b))
    return np.divide(a, b, out=np.full(shape, np.nan), where=np.asarray(b) != 0)


def confusion_metrics(c: np.ndarray) -> dict[str, np.ndarray]:
    """Vectorized metrics on *full* 5x5 confusion matrices, gold rows/pred columns."""
    c = np.asarray(c, dtype=np.float64)
    require(c.shape[-2:] == (5, 5) and np.all(c >= 0), "invalid confusion matrices")
    g, p = c.sum(axis=-1), c.sum(axis=-2)
    n = g.sum(axis=-1)
    d = np.arange(5)[None, :] - np.arange(5)[:, None]
    diag = np.diagonal(c, axis1=-2, axis2=-1)
    bias = finite_ratio((c * d).sum(axis=(-2, -1)), n)
    expected = finite_ratio(g[..., :, None] * p[..., None, :], n[..., None, None])
    qwk = 1 - finite_ratio((c * d**2).sum(axis=(-2, -1)),
                          (expected * d**2).sum(axis=(-2, -1)))
    concordant = np.zeros_like(n)
    discordant = np.zeros_like(n)
    for i in range(5):
        for j in range(5):
            concordant += c[..., i, j] * c[..., i+1:, j+1:].sum(axis=(-2, -1))
            discordant += c[..., i, j] * c[..., i+1:, :j].sum(axis=(-2, -1))
    all_pairs = n * (n - 1) / 2
    untied_gold = all_pairs - (g * (g - 1) / 2).sum(axis=-1)
    untied_pred = all_pairs - (p * (p - 1) / 2).sum(axis=-1)
    result = {
        "Exact": finite_ratio(diag.sum(axis=-1), n),
        "MAE": finite_ratio((c * np.abs(d)).sum(axis=(-2, -1)), n),
        "Signed_Bias": bias, "absolute_Signed_Bias": np.abs(bias),
        "Kendall_tau_b": finite_ratio(concordant-discordant, np.sqrt(untied_gold*untied_pred)),
        "QWK": qwk,
        "L2H_rate": finite_ratio(c[..., :2, 3:].sum(axis=(-2, -1)), g[..., :2].sum(axis=-1)),
        "H2L_rate": finite_ratio(c[..., 3:, :2].sum(axis=(-2, -1)), g[..., 3:].sum(axis=-1)),
    }
    for i in range(5):
        result[f"Label_{i+1}_Recall"] = finite_ratio(diag[..., i], g[..., i])
    return result


def cluster_codes(rows: list[dict], unit: str) -> np.ndarray:
    keys = []
    for r in rows:
        require(all(isinstance(r.get(k), str) and r[k] for k in
                    ("record_id", "question_key", "answer_key")), "missing cluster identity")
        if unit == "question":
            key = (r["question_key"],)
        elif unit == "qa":
            key = (r["question_key"], r["answer_key"])
        elif unit == "record":
            key = (r["record_id"],)
        else:
            raise ValueError("unknown resampling unit")
        keys.append(key)
    index = {key: i for i, key in enumerate(sorted(set(keys)))}
    return np.asarray([index[k] for k in keys], dtype=np.int64)


def validate_predictions(rows: list[dict], source: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    require(len(rows) == len(source), "prediction/source row count mismatch")
    ids = [r["record_id"] for r in source]
    require(len(set(ids)) == len(ids), "duplicate source record")
    scores, forced = [], []
    for i, (r, s) in enumerate(zip(rows, source, strict=True)):
        require(r["record_id"] == s["record_id"], "prediction/source identity or order mismatch")
        require(type(r["row_position"]) is int and r["row_position"] == i, "row position mismatch")
        require(type(s["label_5"]) is int and 1 <= s["label_5"] <= 5, "invalid source label")
        require(type(r["label_5"]) is int and r["label_5"] == s["label_5"], "label mismatch")
        require(r["metric_id"] == s["metric_id"] and r["language"] == s["language"], "stratum mismatch")
        require(r["parse_success"] is True, "parse failure: refusing silent exclusion")
        score = r["prediction"]["score"]
        require(type(score) is int and 1 <= score <= 5, "invalid predicted score")
        require(type(r["forced_completion"]) is bool, "missing/nonboolean forced-completion status")
        scores.append(score)
        forced.append(r["forced_completion"])
    return np.asarray(scores), np.asarray(forced, dtype=np.int64)


def cluster_tensors(gold: np.ndarray, predictions: np.ndarray,
                    forced: np.ndarray, codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """predictions/forced [models, records]; return [clusters, models, ...]."""
    k, m = int(codes.max()) + 1, len(predictions)
    c = np.zeros((k, m, 5, 5), dtype=np.float64)
    f = np.zeros((k, m), dtype=np.float64)
    for j, pred in enumerate(predictions):
        np.add.at(c[:, j], (codes, gold - 1, pred - 1), 1)
        np.add.at(f[:, j], codes, forced[j])
    return c, f


def bootstrap_metrics(c: np.ndarray, f: np.ndarray, *, replicates: int = REPLICATES,
                      seed: int = RNG_SEED) -> dict[str, np.ndarray]:
    k, m = c.shape[:2]
    rng = np.random.Generator(np.random.PCG64(seed))
    chunks: dict[str, list] = {}
    for start in range(0, replicates, 100):
        weights = rng.multinomial(k, np.full(k, 1/k), size=min(100, replicates-start))
        samples = (weights @ c.reshape(k, -1)).reshape(-1, m, 5, 5)
        values = confusion_metrics(samples)
        n = samples.sum(axis=(-2, -1))
        values["forced_completion_rate"] = finite_ratio(weights @ f, n)
        values["strict_parse_rate"] = np.ones_like(n)
        for name, array in values.items():
            chunks.setdefault(name, []).append(array)
    return {name: np.concatenate(arrays) for name, arrays in chunks.items()}


def interval(values: np.ndarray, *, minimum: int = MIN_VALID) -> dict:
    valid = values[np.isfinite(values)]
    sufficient = len(valid) >= minimum
    return {"ci95_low": float(np.quantile(valid, .025)) if sufficient else None,
            "ci95_high": float(np.quantile(valid, .975)) if sufficient else None,
            "valid_replicates": len(valid), "undefined_replicates": len(values)-len(valid),
            "minimum_valid_met": sufficient}


def centered_p(values: np.ndarray, point: float, *, minimum: int = MIN_VALID) -> float | None:
    valid = values[np.isfinite(values)]
    if len(valid) < minimum or not np.isfinite(point):
        return None
    return float((1 + np.count_nonzero(np.abs(valid-point) >= abs(point))) / (1+len(valid)))


def holm(p_values: list[float | None]) -> list[float]:
    values = np.asarray([1.0 if x is None else x for x in p_values])
    require(np.all((values >= 0) & (values <= 1)), "invalid p value")
    order = np.argsort(values, kind="stable")
    adjusted = np.minimum(1, np.maximum.accumulate(values[order] * np.arange(len(values), 0, -1)))
    result = np.empty(len(values))
    result[order] = adjusted
    return result.tolist()


def compare_historical(metrics: dict[str, float], historical: dict) -> int:
    compared = 0
    for name, value in historical.items():
        current_name = ALIASES.get(name, name)
        if current_name in metrics and value not in ("", None):
            require(np.isclose(metrics[current_name], float(value), atol=1e-10, rtol=0),
                    f"historical point mismatch: {name}")
            compared += 1
    require(compared >= 5, "insufficient historical metric cross-checks")
    return compared


class Inputs:
    def __init__(self, private: Path):
        self.private = private
        self.hashes: dict[str, str] = {}

    def read(self, path: Path, *, lines: bool = False) -> Any:
        value, digest = jsonl_bytes(path) if lines else json_bytes(path)
        try:
            name = "private/" + path.relative_to(self.private).as_posix()
        except ValueError:
            name = path.relative_to(ROOT).as_posix()
        self.hashes[name] = digest
        return value

    def digest(self, path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def csv(self, path: Path) -> list[dict]:
        payload = path.read_bytes()
        self.hashes[path.relative_to(ROOT).as_posix()] = hashlib.sha256(payload).hexdigest()
        return list(csv.DictReader(payload.decode().splitlines()))


def load_study(inputs: Inputs, split: str) -> tuple[list, list, np.ndarray, np.ndarray, list]:
    source_path = ROOT / "thesis_exp/data/splits/paper_like_triple_seed42" / f"{split}.jsonl"
    source = inputs.read(source_path, lines=True)
    expected_hash = {"dev": "a18d6a27b9a524d4592a359658ae70c9348fe88e43c962971ba95f62d2b6cdf0",
                     "test": "9bc6ad99dbc3856e292b073f1ce0e883ac980574b5703533f449c34600a5d7af"}[split]
    require(inputs.digest(source_path) == expected_hash, "locked split hash mismatch")
    require(len(source) == {"dev": 664, "test": 2218}[split], "locked split size mismatch")
    cluster_codes(source, "question")
    if split == "dev":
        sft_lock = inputs.read(ROOT/PUBLIC/"protocol/sft_dev_checkpoint_selection_frozen_lock.json")
        bindings = {(r["arm"], r["seed"]): r for r in sft_lock["selected_dev_result_bindings"]}
        require(len(bindings) == 12 and all(r["selected_epoch"] == 3 for r in bindings.values()), "selected SFT lock mismatch")
        old = inputs.csv(ROOT/PUBLIC/"preference_lr5e6_followup/dev_summary/per_seed_metrics.csv")
        historical = {(r["arm"], int(r["seed"])): r for r in old}
        specs = [(a, "dev_runs_vllm", a.lower(), True) for a in ("S0", "R1", "R2", "R3")]
        specs += [(a, "preference_dev", a.lower(), False) for a in ("P1_FIELD_DPO", "P2_SORC_SCORE", "P3_JOINT_SORC")]
    else:
        bindings = {}
        old = inputs.csv(ROOT/PUBLIC/"sorc_dpo_one_time_test_v1/final_results/per_seed_metrics.csv")
        historical = {(r["arm"], int(r["seed"])): r for r in old}
        lock = inputs.read(ROOT/PUBLIC/"mechanism_control_test_v1/public_result_lock.json")
        for name in ("per_seed_metrics", "final_results", "paired_bootstrap", "multiseed_summary"):
            extension = "csv" if name in ("per_seed_metrics", "multiseed_summary") else "json"
            path = inputs.private/"mechanism_summary"/f"{name}.{extension}"
            require(inputs.digest(path) == lock["private_result_hashes"][f"{name}_sha256"], "frozen mechanism result hash mismatch")
        path = inputs.private/"mechanism_summary/per_seed_metrics.csv"
        payload = path.read_bytes()
        inputs.hashes["private/mechanism_summary/per_seed_metrics.csv"] = hashlib.sha256(payload).hexdigest()
        for r in csv.DictReader(payload.decode().splitlines()):
            historical[(r["arm"], int(r["seed"]))] = r
        specs = [(a, "test_runs", a.lower(), False) for a in ("P0_R3_SFT", "P1_FIELD_DPO", "P2_SORC_SCORE", "P3_JOINT_SORC")]
        specs += [(a, "mechanism_runs", a.lower(), False) for a in ("R3_TOKENAVG", "P1_FULLSEQ", "P1_SYN_LR5E6")]
    keys, predictions, forced, provenance = [], [], [], []
    for arm, root, dirname, sft in specs:
        for seed in SEEDS:
            run = inputs.private/root/dirname/(f"seed{seed}" if sft else f"seed_{seed}")
            if sft:
                run /= "epoch3"
            path = run/"predictions.jsonl"
            rows = inputs.read(path, lines=True)
            protocol = inputs.read(run/"protocol.json")
            require(protocol["arm"] == arm and protocol["seed"] == seed, "run metadata mismatch")
            if sft:
                binding = bindings[(arm, seed)]
                for name, filename in (("predictions_jsonl", "predictions.jsonl"), ("protocol", "protocol.json"), ("metrics", "metrics.json")):
                    require(inputs.digest(run/filename) == binding[name]["sha256"], "frozen selected SFT artifact mismatch")
                old_metrics = inputs.read(run/"metrics.json")
                old_row = {**old_metrics["score"], **old_metrics["execution"]}
                anchor = "pre_existing_public_SFT_prediction_hash"
            else:
                old_row = historical[(arm, seed)]
                if split == "test":
                    receipt = inputs.read(run/"completion_receipt.json")
                    require(receipt["arm"] == arm and receipt["seed"] == seed and receipt["rows"] == len(source), "receipt metadata mismatch")
                    require(receipt["predictions_sha256"] == inputs.digest(path), "receipt prediction hash mismatch")
                    require(receipt["protocol_sha256"] == inputs.digest(run/"protocol.json"), "receipt protocol hash mismatch")
                    require(protocol["test_sha256"] == expected_hash, "protocol test source mismatch")
                    anchor = "archived_completion_receipt_plus_historical_point_reproduction"
                else:
                    require(protocol["dev_sha256"] == expected_hash, "protocol dev source mismatch")
                    anchor = "historical_point_reproduction_no_individual_public_prediction_hash_checked"
            pred, f = validate_predictions(rows, source)
            gold = np.asarray([s["label_5"] for s in source])
            c = np.bincount((gold-1)*5 + pred-1, minlength=25).reshape(5, 5)
            metrics = {k: float(v) for k, v in confusion_metrics(c).items()}
            metrics.update(strict_parse_rate=1.0, forced_completion_rate=float(f.mean()))
            matched = compare_historical(metrics, old_row)
            keys.append((arm, seed)); predictions.append(pred); forced.append(f)
            provenance.append({"split": split, "arm": arm, "seed": seed,
                               "provenance_level": anchor, "historical_metrics_matched": matched})
    return source, keys, np.asarray(predictions), np.asarray(forced), provenance


def analyze(source: list, keys: list, predictions: np.ndarray, forced: np.ndarray,
            split: str) -> tuple[list, list, list, dict]:
    gold = np.asarray([r["label_5"] for r in source])
    c, f = cluster_tensors(gold, predictions, forced, cluster_codes(source, "record"))
    full = c.sum(axis=0)
    point = confusion_metrics(full)
    point["forced_completion_rate"] = forced.mean(axis=1)
    point["strict_parse_rate"] = np.ones(len(keys))
    per_seed = []
    for i, (arm, seed) in enumerate(keys):
        row = {"split": split, "arm": arm, "seed": seed, "n": len(gold),
               **{k: float(v[i]) for k, v in point.items()},
               "L2H_count": int(full[i, :2, 3:].sum()), "low_n": int((gold <= 2).sum()),
               "H2L_count": int(full[i, 3:, :2].sum()), "high_n": int((gold >= 4).sum()),
               "forced_completion_count": int(forced[i].sum()),
               "confusion_matrix_gold_rows_pred_columns": full[i].astype(int).tolist()}
        for label in range(1, 6):
            row[f"Label_{label}_n"] = int((gold == label).sum())
            row[f"Label_{label}_correct_count"] = int(full[i, label-1, label-1])
        per_seed.append(row)
    arm_indices = {arm: [keys.index((arm, s)) for s in SEEDS] for arm, _ in keys}
    arm_results, contrasts, support = [], [], {}
    for unit in ("question", "qa", "record"):
        codes = cluster_codes(source, unit)
        c, f = cluster_tensors(gold, predictions, forced, codes)
        sizes = np.bincount(codes)
        support[unit] = {"clusters": len(sizes), "rows": len(gold), "min_cluster_rows": int(sizes.min()),
                         "max_cluster_rows": int(sizes.max()), "median_cluster_rows": float(np.median(sizes)),
                         "low_score_clusters": len(set(codes[gold <= 2].tolist())),
                         "class_support": {str(i): {"rows": int((gold == i).sum()),
                                  "clusters": len(set(codes[gold == i].tolist()))} for i in range(1, 6)}}
        boot = bootstrap_metrics(c, f)
        for arm, ix in arm_indices.items():
            for metric in point:
                # np.mean deliberately propagates undefined seed metrics.
                sample = boot[metric][:, ix].mean(axis=1)
                arm_results.append({"split": split, "unit": unit, "arm": arm, "metric": metric,
                                    "mean": float(point[metric][ix].mean()),
                                    "seed_sample_sd": float(point[metric][ix].std(ddof=1)),
                                    **interval(sample)})
        for family, (family_split, comparisons) in FAMILIES.items():
            if split != family_split:
                continue
            rows = []
            for contrast, baseline, treatment in comparisons:
                b, t = arm_indices[baseline], arm_indices[treatment]
                for metric in point:
                    seed_deltas = point[metric][t] - point[metric][b]
                    estimate = float(seed_deltas.mean())
                    sample = (boot[metric][:, t] - boot[metric][:, b]).mean(axis=1)
                    r = {"split": split, "unit": unit, "family": family, "contrast": contrast,
                         "baseline": baseline, "treatment": treatment, "metric": metric,
                         "delta_treatment_minus_baseline": estimate,
                         "paired_seed_sample_sd": float(seed_deltas.std(ddof=1)),
                         **{f"delta_seed_{s}": float(seed_deltas[j]) for j, s in enumerate(SEEDS)},
                         **interval(sample), "p_centered_bootstrap_approx": centered_p(sample, estimate),
                         "endpoint_role": "primary" if metric in PRIMARY else "exploratory",
                         "family_holm_p": None, "pooled_test_holm_sensitivity_p": None}
                    rows.append(r)
            primary = [r for r in rows if r["metric"] in PRIMARY]
            for r, p in zip(primary, holm([r["p_centered_bootstrap_approx"] for r in primary]), strict=True):
                r["family_holm_p"] = p
            contrasts.extend(rows)
        if split == "test":
            pooled = [r for r in contrasts if r["unit"] == unit and r["metric"] in PRIMARY]
            for r, p in zip(pooled, holm([r["p_centered_bootstrap_approx"] for r in pooled]), strict=True):
                r["pooled_test_holm_sensitivity_p"] = p
        print(f"{split}/{unit}: {len(sizes)} clusters; {REPLICATES} paired replicates complete", flush=True)
    return per_seed, arm_results, contrasts, support


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(clean(value), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+"\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in clean(row).items()})


def historical_comparisons(inputs: Inputs, contrasts: list[dict]) -> list[dict]:
    old = {
        "SFT_DEV": inputs.read(ROOT/PUBLIC/"dev_summary_vllm/paired_bootstrap.json"),
        "PREFERENCE_DEV": inputs.read(ROOT/PUBLIC/"preference_lr5e6_followup/dev_summary/paired_bootstrap.json"),
        "FORMAL_TEST": inputs.read(ROOT/PUBLIC/"sorc_dpo_one_time_test_v1/final_results/paired_bootstrap.json"),
        "MECHANISM_TEST": inputs.read(inputs.private/"mechanism_summary/paired_bootstrap.json"),
    }
    result = []
    reverse_aliases = {v: k for k, v in ALIASES.items()}
    for r in contrasts:
        if r["unit"] != "question":
            continue
        family, metric = r["family"], r["metric"]
        entry = None
        if isinstance(old[family], list):
            name = f'{r["treatment"]}_minus_{r["baseline"]}'
            comparison = next((x for x in old[family] if x["comparison"] == name), None)
            if comparison:
                entry = comparison["deltas"].get(reverse_aliases.get(metric, metric))
            sign = 1
            method = "seed_plus_record_hierarchical_2000"
        else:
            entry = old[family][r["contrast"]].get(metric)
            sign = -1 if metric in ("MAE", "L2H_rate", "absolute_Signed_Bias", "H2L_rate") else 1
            method = "fixed_seed_record_10000_sign_tail_p_approximation"
        if entry is None:
            continue
        low, high = entry["ci95_low"], entry["ci95_high"]
        if sign < 0:
            low, high = -high, -low
        result.append({"family": family, "contrast": r["contrast"], "metric": metric,
                       "delta_treatment_minus_baseline": r["delta_treatment_minus_baseline"],
                       "historical_bootstrap": method,
                       "historical_ci95_low": low, "historical_ci95_high": high,
                       "historical_holm_p": entry.get("holm_adjusted_p"),
                       "question_ci95_low": r["ci95_low"], "question_ci95_high": r["ci95_high"],
                       "question_holm_p": r["family_holm_p"]})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output_dir.exists(), "use a new output directory; preserve earlier reports")
    inputs = Inputs(args.private_input_dir.resolve())
    # Validate all 42 runs before any bootstrap; do not publish partial output.
    studies = {s: load_study(inputs, s) for s in ("dev", "test")}
    print("42 frozen prediction files validated; historical point metrics reproduced", flush=True)
    all_seeds, all_arms, all_contrasts, support, provenance = [], [], [], {}, []
    for split, (source, keys, predictions, forced, prov) in studies.items():
        per_seed, arms, contrasts, clusters = analyze(source, keys, predictions, forced, split)
        all_seeds.extend(per_seed); all_arms.extend(arms); all_contrasts.extend(contrasts)
        support[split] = clusters; provenance.extend(prov)
    old_new = historical_comparisons(inputs, all_contrasts)
    report = {"status": "FROZEN_PREDICTION_CLUSTER_RECOMPUTATION_COMPLETE",
              "estimand": "record_weighted_mean_of_three_seed_specific_metrics",
              "uncertainty": "conditional_on_fixed_trained_seeds_42_43_44",
              "replicates": REPLICATES, "rng": "PCG64", "rng_seed": RNG_SEED,
              "numpy_version": np.__version__, "python_version": platform.python_version(),
              "new_training": False, "new_inference": False, "gpu_used": False,
              "existing_dev_predictions_accessed": True, "existing_test_predictions_accessed": True,
              "raw_probabilities_not_available_in_selected_inputs": True,
              "families": FAMILIES, "cluster_support": support, "source_validation": provenance,
              "input_sha256": inputs.hashes, "per_seed": all_seeds,
              "arm_uncertainty": all_arms, "paired_contrasts": all_contrasts,
              "historical_comparisons": old_new}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir/"cluster_results.json", report)
    write_csv(args.output_dir/"per_seed_metrics.csv", all_seeds)
    write_csv(args.output_dir/"arm_uncertainty.csv", all_arms)
    write_csv(args.output_dir/"paired_contrasts.csv", all_contrasts)
    write_csv(args.output_dir/"historical_comparison.csv", old_new)
    source_files = [Path(__file__), Path(__file__).with_name("STATISTICAL_CLOSURE_PROTOCOL.md"),
                    ROOT/"thesis_exp/tests/test_exp54_statistical_closure.py"]
    lock = {"analysis_source_sha256": {p.relative_to(ROOT).as_posix(): inputs.digest(p) for p in source_files},
            "result_sha256": {p.name: inputs.digest(p) for p in sorted(args.output_dir.iterdir())},
            "upstream_revision": "a75f79d300fe9f7634247443950a268c6c836bfc", "original_reports_modified": False,
            "new_model_execution_allowed": False, "scientific_scope": "statistical_correction_not_new_experiment"}
    write_json(args.output_dir/"analysis_lock.json", lock)


if __name__ == "__main__":
    main()

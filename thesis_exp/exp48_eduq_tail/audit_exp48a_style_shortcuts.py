"""Family-grouped style-only shortcut probe for intended score classes."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .common import OUT, PRIVATE, ensure_output_layout, normalize_text, read_jsonl, write_csv, write_json


def features(text: str, language: str) -> list[float]:
    value = normalize_text(text)
    token_list = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", value)
    unique = set(token_list)
    chars = max(1, len(value))
    punctuation = sum(value.count(mark) for mark in ".,!?;:，。！？；：")
    sentences = len(re.findall(r"[.!?。！？]", value)) or 1
    return [
        float(len(value)), float(len(token_list)), float(sentences), punctuation / chars,
        len(unique) / max(1, len(token_list)),
        sum(len(token) for token in token_list) / max(1, len(token_list)),
        float(language == "zh"), float(language == "en"),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--families", type=Path, default=PRIVATE / "generated_families/exp48a_generated_families.jsonl")
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args()
    ensure_output_layout(args.out_dir)
    families = read_jsonl(args.families)
    rows = []
    for family in families:
        for answer in family["answers"]:
            rows.append({"family_id": family["family_id"], "score": int(answer["intended_score"]), "features": features(answer["text"], family["language"])})
    x = np.asarray([row["features"] for row in rows], dtype=float)
    y = np.asarray([row["score"] for row in rows], dtype=int)
    groups = np.asarray([row["family_id"] for row in rows])
    predictions = np.zeros_like(y)
    fold_rows = []
    splitter = GroupKFold(n_splits=5)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(x, y, groups), 1):
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=48))
        model.fit(x[train_idx], y[train_idx])
        predictions[test_idx] = model.predict(x[test_idx])
        fold_rows.append({"fold": fold, "families": len(set(groups[test_idx])), "rows": len(test_idx), "macro_f1": f1_score(y[test_idx], predictions[test_idx], average="macro", labels=[2, 3, 5], zero_division=0)})
    overall = float(f1_score(y, predictions, average="macro", labels=[2, 3, 5], zero_division=0))
    output = fold_rows + [{"fold": "overall", "families": len(set(groups)), "rows": len(rows), "macro_f1": overall}]
    write_csv(args.out_dir / "tables/exp48a_style_probe_metrics.csv", output)
    write_json(args.out_dir / "private/generated_families/exp48a_style_probe_predictions.json", [{"family_id": row["family_id"], "gold": int(gold), "prediction": int(pred)} for row, gold, pred in zip(rows, y, predictions)])
    print(json.dumps({"style_only_macro_f1": overall, "style_shortcut_pass": overall <= 0.45}, sort_keys=True))


if __name__ == "__main__":
    main()

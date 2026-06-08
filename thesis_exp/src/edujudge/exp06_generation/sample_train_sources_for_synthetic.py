"""Create a train-only source sampling plan for Exp6 synthetic generation."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from thesis_exp.src.edujudge.exp06_generation.common import load_split, planned_error_types, planned_target_labels, source_record_id, write_table
from thesis_exp.src.edujudge.utils.text_norm import stringify, truncate_text


FIELDS = [
    "source_record_id",
    "source_question_key",
    "source_triple_key",
    "metric_canonical",
    "language",
    "scenario_canonical",
    "subject_canonical",
    "education_level_canonical",
    "current_label_5",
    "rubric_text",
    "selected_for_generation",
    "planned_error_types",
    "planned_target_labels",
    "notes",
]


def selection_budget(low_count: int) -> int:
    if low_count <= 1:
        return 4
    if low_count <= 3:
        return 3
    return 2


def build_sampling_plan() -> list[dict[str, Any]]:
    train = load_split("train")
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    low_counts = Counter((row["metric_canonical"], row["language"]) for row in train if int(row.get("label_5", 0)) <= 2)
    for row in train:
        groups[(row["metric_canonical"], row["language"])].append(row)

    selected_ids: set[str] = set()
    for combo, rows in groups.items():
        low_count = low_counts.get(combo, 0)
        budget = selection_budget(low_count)
        # Prefer non-low train anchors with complete rubric/question so the source
        # is stable while the generated answer carries the synthetic low label.
        ranked = sorted(
            rows,
            key=lambda row: (
                int(row.get("label_5", 0)) <= 2,
                not bool(row.get("rubric")),
                -int(row.get("label_5", 0)),
                stringify(row.get("record_id")),
            ),
        )
        for row in ranked[:budget]:
            selected_ids.add(stringify(row.get("record_id")))

    out = []
    for row in sorted(train, key=lambda item: (item["metric_canonical"], item["language"], stringify(item.get("record_id")))):
        metric = row.get("metric_canonical", "")
        language = row.get("language", "")
        low_count = low_counts.get((metric, language), 0)
        selected = stringify(row.get("record_id")) in selected_ids
        out.append(
            {
                "source_record_id": source_record_id(row),
                "source_question_key": row.get("question_key", ""),
                "source_triple_key": row.get("triple_key", ""),
                "metric_canonical": metric,
                "language": language,
                "scenario_canonical": row.get("scenario_canonical", ""),
                "subject_canonical": row.get("subject_canonical", ""),
                "education_level_canonical": row.get("education_level_canonical", ""),
                "current_label_5": row.get("label_5", ""),
                "rubric_text": truncate_text(row.get("rubric", ""), 500),
                "selected_for_generation": selected,
                "planned_error_types": planned_error_types(metric),
                "planned_target_labels": planned_target_labels(),
                "notes": f"train-only source; metric/language train low-count={low_count}; budget={selection_budget(low_count)}",
            }
        )
    return out


def main() -> None:
    rows = build_sampling_plan()
    write_table("train_source_sampling_plan.csv", rows, FIELDS)
    selected = sum(1 for row in rows if row["selected_for_generation"])
    print(f"Wrote train_source_sampling_plan.csv with {len(rows)} rows; selected={selected}")


if __name__ == "__main__":
    main()

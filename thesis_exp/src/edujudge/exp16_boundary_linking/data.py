"""Data loading and tokenization helpers for Exp16A boundary linking."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from thesis_exp.src.edujudge.exp16_boundary_linking import EXP16_DEFAULT_DATA_DIR
from thesis_exp.src.edujudge.utils.io import read_jsonl


LABELS = [1, 2, 3, 4, 5]
DEFAULT_QUALITY_FIELDS = ("metadata", "question", "answer", "metric", "rubric")
VARIANT_BOUNDARY_FIELDS = {
    "global": (),
    "metric_rubric": ("metric", "rubric"),
    "qmr": ("question", "metric", "rubric"),
    "qmr_meta": ("metadata", "question", "metric", "rubric"),
}


def _clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def row_id(row: dict[str, Any]) -> str:
    return _clean(row.get("record_id") or row.get("id") or row.get("sample_id"))


def question_key(row: dict[str, Any]) -> str:
    explicit = _clean(row.get("question_key") or row.get("source_question_key"))
    if explicit:
        return explicit
    question = _clean(row.get("question"))
    return hashlib.sha1(question.encode("utf-8")).hexdigest() if question else row_id(row)


def label_5(row: dict[str, Any]) -> int:
    value = row.get("label_5", row.get("label"))
    if value is None:
        raise ValueError(f"Missing label_5 for row {row_id(row)}")
    label = int(value)
    if label not in LABELS:
        raise ValueError(f"label_5 must be in 1..5, got {label} for row {row_id(row)}")
    return label


def metric_name(row: dict[str, Any]) -> str:
    return _clean(row.get("metric_canonical") or row.get("metric") or row.get("metric_abbr"))


def rubric_text(row: dict[str, Any]) -> str:
    return _clean(row.get("rubric_text") or row.get("rubric") or row.get("rubric_canonical"))


def metadata_text(row: dict[str, Any]) -> str:
    parts = [
        ("Scenario", row.get("scenario_canonical") or row.get("scenario")),
        ("Subject", row.get("subject_canonical") or row.get("subject")),
        ("Education Level", row.get("education_level_canonical") or row.get("education_level")),
        ("Language", row.get("language")),
        ("Metric Group", row.get("metric_group")),
    ]
    lines = [f"{name}: {_clean(value)}" for name, value in parts if _clean(value)]
    return "\n".join(lines)


def field_text(row: dict[str, Any], field: str) -> str:
    field = field.strip().lower()
    if field == "metadata":
        return metadata_text(row)
    if field == "question":
        return f"Question:\n{_clean(row.get('question'))}"
    if field == "answer":
        return f"Answer:\n{_clean(row.get('answer'))}"
    if field == "metric":
        return f"Evaluation Dimension:\n{metric_name(row)}"
    if field == "rubric":
        return f"Rubric:\n{rubric_text(row)}"
    raise ValueError(f"Unsupported text field: {field}")


def parse_boundary_fields(variant: str, boundary_fields: str | None) -> tuple[str, ...]:
    if boundary_fields:
        fields = tuple(part.strip().lower() for part in boundary_fields.split(",") if part.strip())
    else:
        fields = VARIANT_BOUNDARY_FIELDS.get(variant)
        if fields is None:
            raise ValueError(f"Unsupported boundary variant: {variant}")
    if "answer" in fields:
        raise ValueError("boundary_fields must not include answer")
    return fields


def build_text(row: dict[str, Any], fields: tuple[str, ...]) -> str:
    chunks = [field_text(row, field) for field in fields]
    chunks = [chunk for chunk in chunks if chunk.strip()]
    return "\n\n".join(chunks).strip()


def ordinal_target(label: int) -> list[float]:
    return [1.0 if label > threshold else 0.0 for threshold in [1, 2, 3, 4]]


def make_sample(row: dict[str, Any], variant: str = "qmr_meta", boundary_fields: str | None = None) -> dict[str, Any]:
    label = label_5(row)
    b_fields = parse_boundary_fields(variant, boundary_fields)
    quality_text = build_text(row, DEFAULT_QUALITY_FIELDS)
    boundary_text = "Global ordinal boundary context." if variant == "global" and not b_fields else build_text(row, b_fields)
    answer = _clean(row.get("answer"))
    if answer and answer in boundary_text:
        raise ValueError(f"Boundary text leaked answer for row {row_id(row)}")
    return {
        "sample_id": row_id(row),
        "question_key": question_key(row),
        "metric": metric_name(row),
        "metric_abbr": _clean(row.get("metric_abbr")),
        "rubric_text": rubric_text(row),
        "label": label,
        "label_5": label,
        "human_mean_5": row.get("human_mean_5"),
        "quality_text": quality_text,
        "boundary_text": boundary_text,
        "answer": answer,
        "scenario_canonical": _clean(row.get("scenario_canonical") or row.get("scenario")),
        "subject_canonical": _clean(row.get("subject_canonical") or row.get("subject")),
        "education_level_canonical": _clean(row.get("education_level_canonical") or row.get("education_level")),
        "language": _clean(row.get("language")),
        "target": ordinal_target(label),
    }


def read_split(path_or_dir: Path, split: str | None = None) -> list[dict[str, Any]]:
    path = Path(path_or_dir)
    if split is not None:
        path = path / f"{split}.jsonl"
    return read_jsonl(path)


def load_samples(
    path: Path,
    variant: str = "qmr_meta",
    boundary_fields: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    if limit:
        rows = rows[:limit]
    return [make_sample(row, variant=variant, boundary_fields=boundary_fields) for row in rows]


class BoundaryLinkingDataset(torch.utils.data.Dataset):
    def __init__(self, samples: list[dict[str, Any]]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.samples[idx]


class BoundaryLinkingCollator:
    def __init__(self, tokenizer: Any, max_length_quality: int = 2048, max_length_boundary: int = 768) -> None:
        self.tokenizer = tokenizer
        self.max_length_quality = int(max_length_quality)
        self.max_length_boundary = int(max_length_boundary)

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        quality = self.tokenizer(
            [row["quality_text"] for row in rows],
            padding=True,
            truncation=True,
            max_length=self.max_length_quality,
            return_tensors="pt",
        )
        boundary = self.tokenizer(
            [row["boundary_text"] for row in rows],
            padding=True,
            truncation=True,
            max_length=self.max_length_boundary,
            return_tensors="pt",
        )
        labels = torch.tensor([int(row["label"]) for row in rows], dtype=torch.long)
        targets = torch.tensor([row["target"] for row in rows], dtype=torch.float32)
        return {
            "quality_input_ids": quality["input_ids"],
            "quality_attention_mask": quality["attention_mask"],
            "boundary_input_ids": boundary["input_ids"],
            "boundary_attention_mask": boundary["attention_mask"],
            "labels": labels,
            "targets": targets,
            "samples": rows,
        }


class SimpleBoundaryTokenizer:
    """Tiny deterministic tokenizer for CPU sanity checks without downloads."""

    pad_token_id = 0
    cls_token_id = 101
    sep_token_id = 102

    def __init__(self, vocab_size: int = 30522) -> None:
        self.vocab_size = int(vocab_size)

    def _ids(self, text: str, max_length: int) -> list[int]:
        pieces = text.replace("\n", " ").split()
        ids = [self.cls_token_id]
        for token in pieces[: max(0, max_length - 2)]:
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            ids.append(103 + (int(digest[:8], 16) % max(1, self.vocab_size - 103)))
        ids.append(self.sep_token_id)
        return ids[:max_length]

    def __call__(
        self,
        texts: list[str],
        padding: bool = True,
        truncation: bool = True,
        max_length: int = 128,
        return_tensors: str = "pt",
    ) -> dict[str, torch.Tensor]:
        del padding, truncation, return_tensors
        encoded = [self._ids(text, max_length=max_length) for text in texts]
        width = max(len(ids) for ids in encoded) if encoded else 1
        input_ids = []
        masks = []
        for ids in encoded:
            pad = [self.pad_token_id] * (width - len(ids))
            input_ids.append(ids + pad)
            masks.append([1] * len(ids) + [0] * len(pad))
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
        }


def default_data_paths() -> dict[str, Path]:
    return {split: EXP16_DEFAULT_DATA_DIR / f"{split}.jsonl" for split in ["train", "dev", "test"]}

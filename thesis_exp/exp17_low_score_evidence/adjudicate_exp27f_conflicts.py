"""Build Exp27F conflict-adjudication pilot outputs.

Exp27F is an offline, train-only adjudication pilot over the Exp27E top-40
conflict queue. It does not call APIs, does not train, and does not read
dev/test labels. The adjudications below are intentionally explicit so that a
human or GPT reviewer can audit every case and revise individual records.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import jsonschema


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.src.edujudge.utils.io import write_csv, write_json, write_jsonl, write_text  # noqa: E402


DEFAULT_EXP27E_DIR = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp27e_provider_bias_conflict_analysis_seed42"
)
DEFAULT_OUT_DIR = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp27f_conflict_adjudication_pilot_seed42"
)
DEFAULT_REASON_ROOT = Path("5-grades")
SCHEMA_PATH = Path("thesis_exp/exp17_low_score_evidence/schemas/exp27e_conflict_adjudication_schema.json")
PROVIDERS = ("qwen", "deepseek")
REASON_FILES = (
    "5_human_1.jsonl",
    "5_human_2.jsonl",
    "5_human_3.jsonl",
    "5_merge_human_metric_en.jsonl",
    "5_merge_human_metric_zh.jsonl",
)


def a(
    sid: str,
    score_range: tuple[int, int],
    score: int,
    original: bool,
    qwen: bool,
    deepseek: bool,
    evidence: str,
    bucket: str,
    risk: str,
    use: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "sample_id": sid,
        "adjudicated_score_range": list(score_range),
        "most_plausible_score": score,
        "original_label_plausible": original,
        "qwen_plausible": qwen,
        "deepseek_plausible": deepseek,
        "rubric_evidence": evidence,
        "adjudicated_failure_bucket": bucket,
        "adjudicated_derived_risk": risk,
        "recommended_training_use": use,
        "adjudication_reason": reason,
    }


ADJUDICATIONS: dict[str, dict[str, Any]] = {
    "89f8c6ed8c1b947a44510dcc18b38a1d97571453": a(
        "89f8c6ed8c1b947a44510dcc18b38a1d97571453",
        (2, 3),
        2,
        True,
        True,
        False,
        "The answer selects self-determination for a territorial-dispute question and adds caveats, but this is not a clearly safe factual answer for state territorial disputes.",
        "hidden_or_missing_failure",
        "high",
        "review_only",
        "Original low score and Qwen strict score are plausible because the answer may overstate self-determination. DeepSeek 5 is too lenient for the factual-accuracy rubric.",
    ),
    "c0d4f41ccf2066449887e86887cb7b8903269b2a": a(
        "c0d4f41ccf2066449887e86887cb7b8903269b2a",
        (2, 3),
        2,
        True,
        True,
        False,
        "The response contains inconsistent mathematical claims about which functions are diffeomorphisms and contradicts parts of the student's selection.",
        "visible_failure",
        "high",
        "review_only",
        "The case depends on mathematical answer-key interpretation. The low score is defensible; a 5 is not defensible under relevance/error-identification rubrics.",
    ),
    "f33e86ba6eaa5281c801dea1ed174135a7d4c0da": a(
        "f33e86ba6eaa5281c801dea1ed174135a7d4c0da",
        (4, 5),
        5,
        False,
        True,
        False,
        "The response correctly identifies that hyperpolarization is part of an action potential and that calcium influx is the excluded option.",
        "no_failure",
        "low",
        "high_weight",
        "For factual accuracy, the explanation is substantively correct despite a nonnumeric score field. The original 2 and DeepSeek 1 appear overly punitive.",
    ),
    "7b2343c0fac6f79059007d116f746842fe6a6637": a(
        "7b2343c0fac6f79059007d116f746842fe6a6637",
        (1, 2),
        2,
        True,
        False,
        True,
        "The response assigns Score=1 while saying the student's true/false answer is correct, creating a direct score-reason contradiction.",
        "visible_failure",
        "high",
        "low_weight",
        "The low label captures the scoring contradiction. Qwen's high score overlooks that the final score itself is wrong.",
    ),
    "b51fe8038ded98685f2c18b27990ce26056dfc17": a(
        "b51fe8038ded98685f2c18b27990ce26056dfc17",
        (2, 3),
        2,
        True,
        True,
        False,
        "The answer gives a generic Porter framework explanation but does not integrate a concrete industry context or scenario-specific evidence.",
        "hidden_or_missing_failure",
        "high",
        "low_weight",
        "For scenario integration, the answer is fluent but generic. DeepSeek 4 is too generous for the rubric dimension being judged.",
    ),
    "bd384e01426757c4dd86e755a75d5ef86fe6e59f": a(
        "bd384e01426757c4dd86e755a75d5ef86fe6e59f",
        (1, 2),
        2,
        True,
        False,
        True,
        "The score says 1 even though the explanation says the student's answer is correct, so the scoring decision contradicts the rationale.",
        "visible_failure",
        "high",
        "low_weight",
        "The original low label and DeepSeek strict score align with the internal inconsistency. Qwen 4 misses the score-channel error.",
    ),
    "bdfb1901aed7bfddbd4bcd0898854e7ecfd4513c": a(
        "bdfb1901aed7bfddbd4bcd0898854e7ecfd4513c",
        (2, 3),
        2,
        True,
        True,
        False,
        "The answer frames self-determination as a territorial-dispute principle, which is legally debatable and not sufficiently justified for a PhD-level factual item.",
        "hidden_or_missing_failure",
        "high",
        "review_only",
        "The original low score is plausible, but the legal answer-key ambiguity makes this better as a review-only adjudication sample.",
    ),
    "a5b76b94f849ef3c81364bfd2a49a16777a2d7fa": a(
        "a5b76b94f849ef3c81364bfd2a49a16777a2d7fa",
        (3, 4),
        4,
        False,
        True,
        True,
        "The reasoning explains the components of Porter's Five Forces and how they shape competition, with no obvious hidden failure.",
        "no_failure",
        "low",
        "low_weight",
        "The original 2 appears too harsh for reasoning rigor. Both teacher scores are more plausible, though the answer may still be somewhat generic.",
    ),
    "0578273c54e1a724084529ebbc385f1bdd1076ea": a(
        "0578273c54e1a724084529ebbc385f1bdd1076ea",
        (5, 5),
        5,
        False,
        True,
        True,
        "The response selects that z^2+1 has no singularities, which is correct for an entire polynomial.",
        "no_failure",
        "low",
        "high_weight",
        "The original label 1 is very likely label noise. Both teachers correctly identify the answer as high quality.",
    ),
    "5238e123a16a826d9dd25c0454890d96db10099a": a(
        "5238e123a16a826d9dd25c0454890d96db10099a",
        (5, 5),
        5,
        False,
        True,
        True,
        "The response gives the correct complex-analysis option: the polynomial has no singularities.",
        "no_failure",
        "low",
        "high_weight",
        "This is a near-duplicate of the previous case and strongly suggests original low-label noise.",
    ),
    "65890fb994cef054dc75e0e6dd4c1b3e2d91805d": a(
        "65890fb994cef054dc75e0e6dd4c1b3e2d91805d",
        (2, 4),
        3,
        True,
        True,
        True,
        "The student definition of demand elasticity is correct but brief; the response's score and feedback are reasonable but not clearly excellent.",
        "no_failure",
        "medium",
        "low_weight",
        "A 2, 3, or 4 can be defended depending on completeness expectations, so this should not be used as a high-confidence training target.",
    ),
    "d47f50d9f862f2b3b38b2bce2d879ce46d379e2b": a(
        "d47f50d9f862f2b3b38b2bce2d879ce46d379e2b",
        (4, 5),
        4,
        False,
        True,
        True,
        "The teaching-material response includes objectives, key points, difficulties, and activity design aligned with constructivist learning theory.",
        "no_failure",
        "low",
        "low_weight",
        "The original 2 is not well supported by the rubric. Use low weight because the response is long and may contain some generic content.",
    ),
    "46133cfb383c1c10e94bab324dabdce1277ab939": a(
        "46133cfb383c1c10e94bab324dabdce1277ab939",
        (3, 4),
        4,
        True,
        False,
        True,
        "The output follows the requested JSON-like scoring task but contains mathematical and formatting imperfections around the diffeomorphism options.",
        "visible_failure",
        "medium",
        "review_only",
        "This case mixes instruction-following, answer-key ambiguity, and score-format issues, so it is unsuitable as a clean training target.",
    ),
    "c42d9020842ba63f6ee5de91d4d4b21f97f246ac": a(
        "c42d9020842ba63f6ee5de91d4d4b21f97f246ac",
        (5, 5),
        5,
        True,
        False,
        True,
        "The answer correctly identifies cooperative learning as emphasizing active student participation and gives coherent feedback.",
        "no_failure",
        "low",
        "high_weight",
        "The original 5 and DeepSeek 5 are plausible. Qwen 2 is overly strict for this clearly correct pedagogy item.",
    ),
    "b9f1c6853708c1217088b33dabe3b2ce1f70233e": a(
        "b9f1c6853708c1217088b33dabe3b2ce1f70233e",
        (3, 4),
        3,
        True,
        True,
        False,
        "The dynamic-efficiency plan has relevant modeling steps but is abstract and notation-heavy, so clarity and learner accessibility are limited.",
        "visible_failure",
        "medium",
        "low_weight",
        "The original 4 and Qwen 3 are defensible. DeepSeek 2 is too harsh, but this is not a clean high-confidence case.",
    ),
    "712cd1dd80a4537e4bee8ff3fa1e3580d4575d0a": a(
        "712cd1dd80a4537e4bee8ff3fa1e3580d4575d0a",
        (2, 3),
        2,
        False,
        True,
        True,
        "The response is mostly exposition about plate tectonics rather than showing higher-order reasoning for mountain formation.",
        "hidden_or_missing_failure",
        "high",
        "review_only",
        "The original 5 looks too generous for higher-order thinking. The case is useful diagnostically but not yet a clean training target.",
    ),
    "b01129b756e85be3f61bf81d006ca0e61ef3e530": a(
        "b01129b756e85be3f61bf81d006ca0e61ef3e530",
        (3, 5),
        4,
        True,
        True,
        True,
        "The response identifies anxiety sources and gives relevant support strategies; the quality is good though not necessarily perfect.",
        "no_failure",
        "low",
        "low_weight",
        "All three labels are within a defensible range, so this is not a high-value conflict for training.",
    ),
    "b5f54f2d11493dd41a74e9e2274d26451a958dcc": a(
        "b5f54f2d11493dd41a74e9e2274d26451a958dcc",
        (1, 3),
        2,
        True,
        True,
        True,
        "The response states the correct option but then tells the student the wrong answer is correct, creating a contradiction.",
        "visible_failure",
        "medium",
        "low_weight",
        "Because the rubric dimension is content relevance rather than final-score correctness, a wide low-to-mid range is more appropriate.",
    ),
    "e226030ea01e090a606049052bd6c187688f7a7a": a(
        "e226030ea01e090a606049052bd6c187688f7a7a",
        (4, 5),
        5,
        True,
        True,
        False,
        "The crop-success reasoning correctly identifies climate as crucial and compares the options in a rubric-grounded way.",
        "no_failure",
        "low",
        "high_weight",
        "The original 5 and Qwen 5 are plausible. DeepSeek 3 appears unnecessarily strict.",
    ),
    "eebb6c2d0851b385f14d5f66295db77afd02c747": a(
        "eebb6c2d0851b385f14d5f66295db77afd02c747",
        (5, 5),
        5,
        True,
        True,
        False,
        "The response lists the same correct options as the student and provides supportive feedback.",
        "no_failure",
        "low",
        "high_weight",
        "The score 3 conflicts with the response's own correctness details; original and Qwen high labels are more reliable.",
    ),
    "022806ec04a3b92cfcb17f3ba4951ae817c1f854": a(
        "022806ec04a3b92cfcb17f3ba4951ae817c1f854",
        (3, 4),
        3,
        True,
        False,
        True,
        "The response content is basically correct, but the score field uses 80 rather than the required 1-5 scale.",
        "visible_failure",
        "medium",
        "low_weight",
        "The out-of-scale score prevents this from being a clean high score; original 3 and DeepSeek 3 are more plausible than Qwen 2.",
    ),
    "18ae31b0930f4812f14b417a1b631e06c74e7410": a(
        "18ae31b0930f4812f14b417a1b631e06c74e7410",
        (4, 5),
        4,
        False,
        True,
        True,
        "The teaching material covers objectives, methods, key difficulties, and activities for crop-growth model parameter optimization.",
        "no_failure",
        "low",
        "low_weight",
        "The original 3 is probably too low, while teacher high scores are more consistent with the rubric.",
    ),
    "6195cd84a8bf06d5a3608b3bb6561111e242ace6": a(
        "6195cd84a8bf06d5a3608b3bb6561111e242ace6",
        (3, 4),
        4,
        True,
        True,
        True,
        "The reasoning points toward leaves and explains photosynthesis without simply outputting the answer.",
        "no_failure",
        "low",
        "low_weight",
        "A 3 or 4 is plausible depending on whether indirect reasoning is enough. The conflict is mild.",
    ),
    "ec9a7870c5ca6a68b1ba47d38ed2852f9d6a317e": a(
        "ec9a7870c5ca6a68b1ba47d38ed2852f9d6a317e",
        (3, 4),
        4,
        True,
        True,
        True,
        "The anxiety-support response is empathetic and specific, though some advice remains generic.",
        "no_failure",
        "low",
        "low_weight",
        "The original 3 and teacher 4 are both defensible, so keep as low-weight rather than using as a strong target.",
    ),
    "1d5f77fabc222d72e4fb87f25eb3e5c41e786cf4": a(
        "1d5f77fabc222d72e4fb87f25eb3e5c41e786cf4",
        (1, 2),
        2,
        True,
        True,
        True,
        "The response calls a correct cooperative-learning answer only partially correct and weakens the feedback.",
        "visible_failure",
        "high",
        "low_weight",
        "The answer is visibly inconsistent with the expected pedagogy answer, so a low score is plausible.",
    ),
    "4a94159b2cfae1127c979628b87af3087401a413": a(
        "4a94159b2cfae1127c979628b87af3087401a413",
        (4, 5),
        4,
        False,
        True,
        True,
        "The response addresses severe anxiety with concrete stressors, emotional analysis, and supportive recommendations.",
        "no_failure",
        "low",
        "low_weight",
        "The original 3 is likely conservative; both teachers' high scores are plausible.",
    ),
    "6c2ca811835d3ab2cc459573f13c0ae3caf83824": a(
        "6c2ca811835d3ab2cc459573f13c0ae3caf83824",
        (4, 5),
        4,
        False,
        True,
        True,
        "The medical learning-path response is structured, personalized, and aligned with the student's weak points.",
        "no_failure",
        "low",
        "low_weight",
        "Original 3 appears conservative. The high teacher scores are plausible, but not strong enough for high-weight due to generic elements.",
    ),
    "a6a61e243a1f0c09ac86c3557bd39377eff935bc": a(
        "a6a61e243a1f0c09ac86c3557bd39377eff935bc",
        (4, 5),
        4,
        False,
        True,
        True,
        "The response provides a coherent staged medical learning plan that uses the student profile and weak-point metadata.",
        "no_failure",
        "low",
        "low_weight",
        "The original 3 likely underestimates the answer, while both teacher scores are plausible.",
    ),
    "b93239ad93a7561a06f1b07c82c82f44bf81667e": a(
        "b93239ad93a7561a06f1b07c82c82f44bf81667e",
        (4, 5),
        4,
        False,
        True,
        True,
        "The response generates a relevant sports-education multiple-choice question and provides plausible options.",
        "no_failure",
        "low",
        "low_weight",
        "The original 3 is conservative. Teachers agree on a high score, but the output is still simple enough to keep low-weight.",
    ),
    "cd63948c28174674b09c1bed2f6fa389c8ec4a34": a(
        "cd63948c28174674b09c1bed2f6fa389c8ec4a34",
        (4, 5),
        4,
        False,
        True,
        True,
        "The public-policy learning plan is structured around the profile, goals, interests, learning style, and challenges.",
        "no_failure",
        "low",
        "low_weight",
        "The original 3 is likely too strict; both teachers' high scores are more compatible with the rubric.",
    ),
    "ff657c059d6b1fdddbeae35add3c566377c0dd86": a(
        "ff657c059d6b1fdddbeae35add3c566377c0dd86",
        (4, 5),
        5,
        True,
        True,
        True,
        "The dynamic-efficiency response gives a detailed model structure and optimization path.",
        "no_failure",
        "low",
        "high_weight",
        "All labels are high and the answer is substantively strong; teacher 4s are compatible with original 5.",
    ),
    "7a3d744f872bfd9b86744f132f9dfc6a3c5d8e9f": a(
        "7a3d744f872bfd9b86744f132f9dfc6a3c5d8e9f",
        (2, 3),
        3,
        True,
        True,
        True,
        "The response claims one key medieval factor was missed; if the answer key includes urbanization, the score 3 is justified.",
        "visible_failure",
        "medium",
        "low_weight",
        "The answer-key dependency means both 2 and 3 are plausible, so this should remain low-weight.",
    ),
    "107a460ba82cd473b9a08a5d43257a7f1d336874": a(
        "107a460ba82cd473b9a08a5d43257a7f1d336874",
        (1, 2),
        2,
        True,
        True,
        True,
        "The output gives an out-of-range Score=8 and an internally odd subscore despite describing Nash equilibrium reasonably.",
        "visible_failure",
        "high",
        "low_weight",
        "The low score is due to score-format/structure failure rather than absence of subject knowledge.",
    ),
    "1947f424625d896777f4dbddef16ab9afb7730aa": a(
        "1947f424625d896777f4dbddef16ab9afb7730aa",
        (1, 2),
        2,
        True,
        True,
        True,
        "The response marks a correct liver-regeneration answer as Score=1 while explaining that the answer is correct.",
        "visible_failure",
        "high",
        "low_weight",
        "This is a clear score-reason contradiction; low labels are plausible despite correct factual text.",
    ),
    "3a5e33210fd977f5c8bc6df5dc98a0402f4663ac": a(
        "3a5e33210fd977f5c8bc6df5dc98a0402f4663ac",
        (3, 4),
        4,
        True,
        True,
        True,
        "The response gives relevant emotional analysis and advice for mild anxiety, with minor generic phrasing.",
        "no_failure",
        "low",
        "low_weight",
        "Original 4 and Qwen 4 are plausible; DeepSeek 3 is within the plausible range.",
    ),
    "41f6f52edd1473f3cc3cde8e6923d1e09fa429d5": a(
        "41f6f52edd1473f3cc3cde8e6923d1e09fa429d5",
        (4, 5),
        4,
        True,
        True,
        True,
        "The personalized learning content matches the student's social-justice interests and hands-on learning preference.",
        "no_failure",
        "low",
        "high_weight",
        "All plausible scores are high and no hidden failure is visible.",
    ),
    "5eca8796afa57332b77efb229a632e49b196cbed": a(
        "5eca8796afa57332b77efb229a632e49b196cbed",
        (3, 4),
        4,
        True,
        True,
        False,
        "The response correctly states the Andes are in South America, but uses an out-of-scale Score=0.",
        "visible_failure",
        "medium",
        "low_weight",
        "Substantive feedback is correct, yet score-format failure prevents treating a 5 as fully plausible.",
    ),
    "c13167075148a265b283ce6bd0a309d539f243f7": a(
        "c13167075148a265b283ce6bd0a309d539f243f7",
        (3, 4),
        4,
        True,
        True,
        False,
        "The civil-rights analysis is relevant and fairly detailed, but the response uses an 85-point score rather than the required 1-5 scale.",
        "visible_failure",
        "medium",
        "low_weight",
        "The original 3 is plausible because of format/scale problems; Qwen 4 is plausible for content; DeepSeek 5 is too lenient.",
    ),
    "e927013b12cac98ecddc4843316f9befc695ed6d": a(
        "e927013b12cac98ecddc4843316f9befc695ed6d",
        (4, 5),
        5,
        True,
        True,
        True,
        "The mild-anxiety response is well structured, empathetic, and specific to exam-preparation pressure.",
        "no_failure",
        "low",
        "high_weight",
        "All labels are high and the case is suitable as a high-score protection control.",
    ),
    "4468842156d5b938a609185df023c4a860497e75": a(
        "4468842156d5b938a609185df023c4a860497e75",
        (3, 5),
        4,
        True,
        True,
        True,
        "The cooperative-learning feedback is substantively correct, but the response uses an 80-point score scale.",
        "visible_failure",
        "medium",
        "low_weight",
        "Content supports a high score, while the scale mismatch supports discounting the case.",
    ),
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def text_key(question: Any, answer: Any) -> str:
    text = normalize_text(question) + "\n---\n" + normalize_text(answer)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def score_region(score: int | None) -> str:
    if score is None:
        return "unknown"
    if score <= 2:
        return "low"
    if score == 3:
        return "mid"
    return "high"


def load_top40(exp27e_dir: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    queue_path = exp27e_dir / "annotation" / "exp27e_gpt55_human_adjudication_queue.csv"
    packets_path = exp27e_dir / "annotation" / "exp27e_gpt55_human_adjudication_packets.jsonl"
    queue = [row for row in read_csv_rows(queue_path) if row.get("top40_for_manual_review", "").lower() == "true"]
    packets = {str(row.get("sample_id")): row for row in read_jsonl(packets_path)}
    missing_packets = sorted(row["sample_id"] for row in queue if row.get("sample_id") not in packets)
    if missing_packets:
        raise ValueError(f"Exp27E queue has missing packets: {missing_packets[:5]}")
    queue_ids = {row["sample_id"] for row in queue}
    adjudication_ids = set(ADJUDICATIONS)
    if queue_ids != adjudication_ids:
        raise ValueError(
            "Top40/adjudication id mismatch: "
            f"missing={sorted(queue_ids - adjudication_ids)[:5]} extra={sorted(adjudication_ids - queue_ids)[:5]}"
        )
    queue.sort(key=lambda row: (int(row.get("adjudication_priority") or 999), row.get("sample_id", "")))
    return queue, packets


def load_reason_index(reason_root: Path) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for name in REASON_FILES:
        path = reason_root / name
        for row in read_jsonl(path):
            key = text_key(row.get("question"), row.get("response"))
            index[key].append(
                {
                    "reason_file": name,
                    "principle": row.get("principle", ""),
                    "reason_score": row.get("score", ""),
                    "reason_snippet": normalize_text(row.get("reason"))[:420],
                    "model": row.get("model", ""),
                }
            )
    return index


def recover_reason_snippets(packets: dict[str, dict[str, Any]], reason_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    index = load_reason_index(reason_root)
    rows: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    for sid, packet in sorted(packets.items()):
        if sid not in ADJUDICATIONS:
            continue
        key = text_key(packet.get("question"), packet.get("answer"))
        matches = index.get(key, [])
        summary.append(
            {
                "sample_id": sid,
                "matched_reason_count": len(matches),
                "matched_principles": sorted({normalize_text(m.get("principle")) for m in matches if m.get("principle")}),
                "matched_scores": sorted({str(m.get("reason_score")) for m in matches if m.get("reason_score") != ""}),
            }
        )
        for match in matches[:3]:
            rows.append({"sample_id": sid, **match})
    return summary, rows


def attach_context(queue: list[dict[str, str]], packets: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    by_queue = {row["sample_id"]: row for row in queue}
    out: list[dict[str, Any]] = []
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)
    for sid, base in ADJUDICATIONS.items():
        row = dict(base)
        row["conflict_type"] = by_queue[sid].get("conflict_type", "")
        errors = sorted(validator.iter_errors(row), key=lambda e: e.path)
        if errors:
            raise ValueError(f"schema validation failed for {sid}: {errors[0].message}")
        packet = packets[sid]
        summary = packet.get("compact_teacher_disagreement_summary", {})
        row["original_train_score"] = packet.get("original_train_score")
        row["qwen_score"] = summary.get("qwen", {}).get("score")
        row["deepseek_score"] = summary.get("deepseek", {}).get("score")
        row["language"] = packet.get("language", "")
        row["metric"] = packet.get("metric", "")
        row["subject"] = packet.get("subject", "")
        row["question_key"] = packet.get("question_key", "")
        out.append(row)
    out.sort(key=lambda row: (int(by_queue[row["sample_id"]].get("adjudication_priority") or 999), row["sample_id"]))
    return out


def provider_distance(provider: str, row: dict[str, Any]) -> int | None:
    value = row.get(f"{provider}_score")
    if value is None or value == "":
        return None
    return abs(int(value) - int(row["most_plausible_score"]))


def summarize(rows: list[dict[str, Any]], reason_summary: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    n = len(rows)
    summary_rows = [
        {"metric": "top40_adjudicated_count", "value": n},
        {"metric": "original_plausible_rate", "value": mean(1 if r["original_label_plausible"] else 0 for r in rows)},
        {"metric": "qwen_plausible_rate", "value": mean(1 if r["qwen_plausible"] else 0 for r in rows)},
        {"metric": "deepseek_plausible_rate", "value": mean(1 if r["deepseek_plausible"] else 0 for r in rows)},
        {"metric": "original_implausible_count", "value": sum(not r["original_label_plausible"] for r in rows)},
        {"metric": "high_weight_count", "value": sum(r["recommended_training_use"] == "high_weight" for r in rows)},
        {"metric": "low_weight_count", "value": sum(r["recommended_training_use"] == "low_weight" for r in rows)},
        {"metric": "review_only_count", "value": sum(r["recommended_training_use"] == "review_only" for r in rows)},
        {"metric": "exclude_count", "value": sum(r["recommended_training_use"] == "exclude" for r in rows)},
        {"metric": "human_reason_recovered_sample_count", "value": sum(int(r["matched_reason_count"]) > 0 for r in reason_summary)},
    ]

    by_type: dict[str, dict[str, Any]] = {}
    for row in rows:
        for ctype in str(row["conflict_type"]).split(";"):
            if not ctype:
                continue
            bucket = by_type.setdefault(
                ctype,
                {
                    "conflict_type": ctype,
                    "count": 0,
                    "original_plausible_count": 0,
                    "qwen_plausible_count": 0,
                    "deepseek_plausible_count": 0,
                    "high_weight_count": 0,
                    "low_weight_count": 0,
                    "review_only_count": 0,
                    "exclude_count": 0,
                },
            )
            bucket["count"] += 1
            bucket["original_plausible_count"] += int(row["original_label_plausible"])
            bucket["qwen_plausible_count"] += int(row["qwen_plausible"])
            bucket["deepseek_plausible_count"] += int(row["deepseek_plausible"])
            bucket[f"{row['recommended_training_use']}_count"] += 1
    type_rows = sorted(by_type.values(), key=lambda r: (-r["count"], r["conflict_type"]))

    reliability_rows: list[dict[str, Any]] = []
    for provider in ("original", "qwen", "deepseek"):
        if provider == "original":
            plausible_key = "original_label_plausible"
            score_key = "original_train_score"
        else:
            plausible_key = f"{provider}_plausible"
            score_key = f"{provider}_score"
        distances = [
            abs(int(row[score_key]) - int(row["most_plausible_score"]))
            for row in rows
            if row.get(score_key) not in {None, ""}
        ]
        reliability_rows.append(
            {
                "source": provider,
                "n": len(distances),
                "plausible_count": sum(row[plausible_key] for row in rows),
                "plausible_rate": mean(1 if row[plausible_key] else 0 for row in rows),
                "mae_to_adjudicated_score": mean(distances) if distances else "",
                "exact_to_adjudicated_score": mean(1 if d == 0 else 0 for d in distances) if distances else "",
                "adjacent_to_adjudicated_score": mean(1 if d <= 1 else 0 for d in distances) if distances else "",
                "low_human_teacher_high_error_count": sum(
                    int(row["original_train_score"]) <= 2
                    and row.get(score_key) not in {None, ""}
                    and int(row[score_key]) >= 4
                    and not row[plausible_key]
                    for row in rows
                )
                if provider != "original"
                else "",
                "high_human_teacher_low_error_count": sum(
                    int(row["original_train_score"]) >= 4
                    and row.get(score_key) not in {None, ""}
                    and int(row[score_key]) <= 2
                    and not row[plausible_key]
                    for row in rows
                )
                if provider != "original"
                else "",
            }
        )

    original_rows: list[dict[str, Any]] = []
    by_region: Counter[str] = Counter()
    implausible_by_region: Counter[str] = Counter()
    for row in rows:
        region = score_region(int(row["original_train_score"]))
        by_region[region] += 1
        implausible_by_region[region] += int(not row["original_label_plausible"])
    for region in ("low", "mid", "high"):
        total = by_region[region]
        bad = implausible_by_region[region]
        original_rows.append(
            {
                "original_label_region": region,
                "top40_count": total,
                "implausible_original_count": bad,
                "implausible_original_rate": bad / total if total else "",
            }
        )

    decision = {
        "adjudication_scope": "Exp27E top40 conflict queue only",
        "adjudication_is_final_gold": False,
        "top40_count": n,
        "original_implausible_rate": sum(not r["original_label_plausible"] for r in rows) / n if n else 0.0,
        "qwen_plausible_rate": sum(r["qwen_plausible"] for r in rows) / n if n else 0.0,
        "deepseek_plausible_rate": sum(r["deepseek_plausible"] for r in rows) / n if n else 0.0,
        "recommended_primary_teacher_for_361": "qwen",
        "recommend_use_dual_teacher_for_361": True,
        "recommend_selective_second_teacher": True,
        "recommend_adjudicate_provider_conflicts_before_training": True,
        "proceed_to_361": True,
        "proceed_to_full_3326": False,
        "recommended_361_strategy": "Use one blind teacher plus a second teacher on conflict-prone/risk-prone cases, then adjudicate high-disagreement cases before producing train labels.",
        "paper_claim_allowed": "The original labels and teacher labels both contain nontrivial conflict; a teacher-audited protocol with selective adjudication is justified.",
        "paper_claim_not_allowed": "Do not claim that Codex top40 adjudication is final gold annotation.",
    }
    return summary_rows, type_rows, reliability_rows, original_rows, decision


def make_report(
    rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    reliability_rows: list[dict[str, Any]],
    reason_summary: list[dict[str, Any]],
    decision: dict[str, Any],
) -> str:
    value = {row["metric"]: row["value"] for row in summary_rows}
    recovered = sum(int(r["matched_reason_count"]) > 0 for r in reason_summary)
    n = len(rows)
    high_weight = value.get("high_weight_count", 0)
    low_weight = value.get("low_weight_count", 0)
    review = value.get("review_only_count", 0)
    provider_bits = "\n".join(
        f"- {r['source']}: plausible={r['plausible_count']}/{r['n']}, "
        f"MAE_to_adjudicated={float(r['mae_to_adjudicated_score']):.3f}"
        for r in reliability_rows
        if r["source"] in {"original", "qwen", "deepseek"}
    )
    return f"""# Exp27F Conflict Adjudication Pilot

Exp27F adjudicates the Exp27E top-40 conflict queue offline. It does not call
teacher APIs, does not train a model, and does not read dev/test labels. The
output is a pilot adjudication artifact for review, not final gold annotation.

## What This Step Does

- Reads the Exp27E top-40 provider/human conflict queue.
- Applies explicit case-level adjudications with the Exp27E schema.
- Checks whether original labels, Qwen labels, and DeepSeek labels are plausible.
- Attempts strict recovery of human reason snippets from `5-grades`.
- Produces a decision on whether 361-case teacher auditing should proceed.

## Main Counts

- Top40 adjudicated samples: {n}
- Original label implausible count: {value.get('original_implausible_count')}
- High-weight usable count: {high_weight}
- Low-weight usable count: {low_weight}
- Review-only count: {review}
- Exclude count: {value.get('exclude_count')}
- Samples with strictly recovered human reason snippets: {recovered}/{n}

## Provider Reliability After Adjudication

{provider_bits}

## Interpretation

The top40 queue confirms that the original labels are not always trustworthy:
some low labels are likely wrong high-quality answers, while some teacher high
scores miss score-format or hidden rubric failures. Qwen is generally more
conservative on these conflicts, while DeepSeek is often more lenient; neither
provider is reliable enough to be used alone without a conflict policy.

The recovered human reasons are useful when they match exactly, but coverage is
not guaranteed because the processed split lost direct reason identifiers. The
pipeline therefore treats recovered snippets as audit evidence only, not as a
mandatory label source.

## Decision

- Proceed to 361-case teacher audit: {decision['proceed_to_361']}
- Proceed directly to full 3326 train relabeling: {decision['proceed_to_full_3326']}
- Recommended primary teacher for 361: {decision['recommended_primary_teacher_for_361']}
- Use dual teacher selectively: {decision['recommend_selective_second_teacher']}

The next step should be a controlled 361-case expansion, not full-train
relabeling. Use dual teacher or second-teacher review on conflict-prone cases,
then adjudicate high-disagreement samples before using them for SFT/DPO data.

## Paper Claim Boundary

Allowed claim: {decision['paper_claim_allowed']}

Not allowed claim: {decision['paper_claim_not_allowed']}
"""


def next_prompt() -> str:
    return """你现在在 pj-000/edubench-eval 仓库中继续 Exp27G：teacher-audited 361-case expansion。

背景：
Exp27F 已完成 Exp27E top40 冲突仲裁 pilot。结论是：原始人类标签和 teacher 标签都存在冲突，不能直接扩大到全量 3326；但可以进入 361-case controlled expansion。Exp27F 不是最终 gold，只是用于确定扩展协议。

目标：
只在 train split 中构造 361 条 teacher-audited annotation packets，并运行 API 标注或生成可运行脚本。不要读取 dev/test labels，不训练模型。

要求：
1. 输入：
   - thesis_exp/data/splits/question_seed42/train.jsonl
   - Exp27D/Exp27E/Exp27F 的轻量结果
2. 输出目录：
   - thesis_exp/exp17_low_score_evidence/outputs/exp27g_teacher_audited_361_seed42/
3. 需要输出：
   - packets/exp27g_361_teacher_packets.jsonl
   - tables/exp27g_sampling_distribution.csv
   - tables/exp27g_leakage_audit.csv
   - reports/exp27g_prepare_report.md
   - decision/exp27g_prepare_decision.json
4. 抽样策略：
   - 覆盖 train 中低分、隐藏失败、teacher-human 冲突、高分保护控制样本。
   - 不允许 dev/test sample_id 或 question_key 泄漏。
   - 对 conflict-prone/risk-prone 样本保留 second-teacher 标注标记。
5. 如果运行 API：
   - 不提交 raw API outputs/logs。
   - 只提交 parsed lightweight CSV/MD/JSON。
6. 验证：
   - python -m py_compile 新增脚本
   - 运行 prepare 脚本
   - 检查 leakage_audit 全 0

最终回复请汇报：
- 构造了多少 361 packets；
- 各风险桶/分数段/语言/metric 分布；
- 是否有泄漏；
- 是否建议开始真实 API 标注；
- 下一步 Codex 命令。"""


def run(args: argparse.Namespace) -> dict[str, Any]:
    queue, packets = load_top40(args.exp27e_dir)
    rows = attach_context(queue, packets)
    top_packets = {sid: packets[sid] for sid in ADJUDICATIONS}
    reason_summary, reason_snippets = recover_reason_snippets(top_packets, args.reason_root)
    summary_rows, type_rows, reliability_rows, original_rows, decision = summarize(rows, reason_summary)

    out = args.out_dir
    write_jsonl(out / "annotation" / "exp27f_top40_adjudications.jsonl", rows)
    write_csv(out / "tables" / "exp27f_adjudication_summary.csv", summary_rows)
    write_csv(out / "tables" / "exp27f_conflict_resolution_by_type.csv", type_rows)
    write_csv(out / "tables" / "exp27f_provider_reliability_after_adjudication.csv", reliability_rows)
    write_csv(out / "tables" / "exp27f_original_human_label_quality.csv", original_rows)
    write_csv(out / "tables" / "exp27f_human_reason_recovery_summary.csv", reason_summary)
    write_csv(out / "annotation" / "exp27f_human_reason_snippets.csv", reason_snippets)
    write_json(out / "decision" / "exp27f_conflict_adjudication_decision.json", decision)
    write_text(out / "reports" / "exp27f_conflict_adjudication_report.md", make_report(rows, summary_rows, reliability_rows, reason_summary, decision))
    write_text(out / "prompts" / "exp27f_next_step_prompt_for_codex.md", next_prompt())

    return {
        "out_dir": str(out),
        "top40_count": len(rows),
        "human_reason_recovered_sample_count": sum(int(r["matched_reason_count"]) > 0 for r in reason_summary),
        "decision": decision,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Exp27F conflict-adjudication pilot outputs.")
    parser.add_argument("--exp27e-dir", type=Path, default=DEFAULT_EXP27E_DIR)
    parser.add_argument("--reason-root", type=Path, default=DEFAULT_REASON_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

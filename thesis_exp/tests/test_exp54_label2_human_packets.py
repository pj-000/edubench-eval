from __future__ import annotations

from thesis_exp.exp54_rar_sft.build_label2_human_audit_packets import (
    RESPONSE_FIELDS,
    _packet_rows,
)


def _source(record_id: str, *, question_key: str) -> dict:
    return {
        "record_id": record_id,
        "question_key": question_key,
        "language": "en",
        "metric_id": "metric-a",
        "metric_canonical": "Metric A",
        "question": f"Question {record_id}",
        "answer": f"Answer {record_id}",
        "rubric": f"Rubric {record_id}",
        "label_5": 2,
        "human_1_5": 2,
        "human_2_5": 2,
        "human_3_5": 3,
        "prediction": {"score": 4},
    }


def test_packet_is_blind_and_answer_key_keeps_private_mapping() -> None:
    packet, answer = _packet_rows([_source("r1", question_key="q1")], "A")
    assert len(packet) == len(answer) == 1
    visible = packet[0]
    assert "record_id" not in visible
    assert "question_key" not in visible
    assert "label_5" not in visible
    assert not any(key.startswith("human_") for key in visible)
    assert "prediction" not in visible
    assert visible["presentation_id"].startswith("L2-A-")
    assert all(visible[field] == "" for field in RESPONSE_FIELDS)
    assert answer[0]["record_id"] == "r1"
    assert answer[0]["automatic_measurement_ambiguous"] is True


def test_reviewer_packets_have_distinct_ids_and_orders() -> None:
    rows = [_source(f"r{index}", question_key=f"q{index}") for index in range(20)]
    packet_a, answer_a = _packet_rows(rows, "A")
    packet_b, answer_b = _packet_rows(rows, "B")
    assert [row["record_id"] for row in answer_a] != [
        row["record_id"] for row in answer_b
    ]
    assert {row["presentation_id"] for row in packet_a}.isdisjoint(
        {row["presentation_id"] for row in packet_b}
    )


def test_packet_build_is_deterministic() -> None:
    rows = [_source(f"r{index}", question_key=f"q{index}") for index in range(5)]
    first = _packet_rows(rows, "A")
    second = _packet_rows(list(reversed(rows)), "A")
    assert first == second

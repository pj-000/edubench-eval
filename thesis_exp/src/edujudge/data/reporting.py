"""Compatibility wrapper for Exp 0 final report generation."""

from __future__ import annotations

from thesis_exp.src.edujudge.data.build_dataset import PROCESSED_PATH
from thesis_exp.src.edujudge.plots.plot_distributions import generate_reports
from thesis_exp.src.edujudge.utils.io import read_jsonl


def write_final_reports() -> None:
    generate_reports(read_jsonl(PROCESSED_PATH))


def main() -> None:
    write_final_reports()


if __name__ == "__main__":
    main()

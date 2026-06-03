"""Postprocess available Exp3 input-ablation results."""

from __future__ import annotations

import argparse

from thesis_exp.src.edujudge.exp03 import EXP03_OUTPUT_DIR, ensure_exp03_dirs
from thesis_exp.src.edujudge.exp03.collect_exp03_results import collect_exp03_results
from thesis_exp.src.edujudge.exp03.compute_input_ablation_metrics import compute_input_ablation_metrics
from thesis_exp.src.edujudge.exp03.plot_input_ablation_figures import plot_input_ablation_figures
from thesis_exp.src.edujudge.exp03.sanity_check_exp03_outputs import run_output_sanity
from thesis_exp.src.edujudge.exp03.write_exp03_report import write_exp03_report
from thesis_exp.src.edujudge.utils.io import relpath


def postprocess_exp03_results(strict: bool = False) -> None:
    ensure_exp03_dirs()
    collect_exp03_results()
    compute_input_ablation_metrics()
    plot_input_ablation_figures()
    write_exp03_report()
    run_output_sanity(allow_pending=not strict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Postprocess Exp3 input-ablation outputs.")
    parser.add_argument("--strict", action="store_true", help="Require all A0-A4 runs to be complete.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    postprocess_exp03_results(strict=args.strict)
    print(f"Exp3 postprocess outputs written to {relpath(EXP03_OUTPUT_DIR)}")


if __name__ == "__main__":
    main()

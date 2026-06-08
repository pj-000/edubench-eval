"""Postprocess Exp5 low-score loss ablation outputs."""

from __future__ import annotations

import argparse

from thesis_exp.src.edujudge.exp05 import EXP05_OUTPUT_DIR, ensure_exp05_dirs
from thesis_exp.src.edujudge.exp05.collect_exp05_results import collect_exp05_results
from thesis_exp.src.edujudge.exp05.sanity_check_exp05_outputs import run_output_sanity
from thesis_exp.src.edujudge.exp05.write_exp05_report import write_exp05_report
from thesis_exp.src.edujudge.utils.io import relpath


def postprocess_exp05_results(strict: bool = False, include_l3b: bool = False) -> None:
    ensure_exp05_dirs()
    collect_exp05_results()
    write_exp05_report()
    run_output_sanity(allow_pending=not strict, include_l3b=include_l3b)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Postprocess Exp5 outputs.")
    parser.add_argument("--strict", action="store_true", help="Require L1 formal output.")
    parser.add_argument("--include-l3b", action="store_true", help="Include L3b in output sanity checks.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    postprocess_exp05_results(strict=args.strict, include_l3b=args.include_l3b)
    print(f"Exp5 postprocess outputs written to {relpath(EXP05_OUTPUT_DIR)}")


if __name__ == "__main__":
    main()

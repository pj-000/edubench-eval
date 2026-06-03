"""Postprocess available Exp4 objective-comparison results."""

from __future__ import annotations

import argparse

from thesis_exp.src.edujudge.exp04 import EXP04_OUTPUT_DIR, ensure_exp04_dirs
from thesis_exp.src.edujudge.exp04.collect_exp04_results import collect_exp04_results
from thesis_exp.src.edujudge.exp04.sanity_check_exp04_outputs import run_output_sanity
from thesis_exp.src.edujudge.exp04.write_exp04_report import write_exp04_report
from thesis_exp.src.edujudge.utils.io import relpath


def postprocess_exp04_results(strict: bool = False) -> None:
    ensure_exp04_dirs()
    collect_exp04_results()
    write_exp04_report()
    run_output_sanity(allow_pending=not strict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Postprocess Exp4 objective-comparison outputs.")
    parser.add_argument("--strict", action="store_true", help="Require all O1-O3 runs to be complete.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    postprocess_exp04_results(strict=args.strict)
    print(f"Exp4 postprocess outputs written to {relpath(EXP04_OUTPUT_DIR)}")


if __name__ == "__main__":
    main()

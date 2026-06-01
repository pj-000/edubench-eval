"""Run the complete Exp 0 pipeline in order."""

from __future__ import annotations

from thesis_exp.src.edujudge.data import build_dataset, inventory_sources, leakage_check, make_splits, profile_schema
from thesis_exp.src.edujudge.plots import plot_distributions
from thesis_exp.src.edujudge.utils.io import ensure_exp_dirs


def main() -> None:
    ensure_exp_dirs()
    steps = [
        ("inventory_sources", inventory_sources.main),
        ("profile_schema", profile_schema.main),
        ("build_dataset", build_dataset.main),
        ("make_splits", make_splits.main),
        ("leakage_check", leakage_check.main),
        ("plot_distributions", plot_distributions.main),
    ]
    for name, func in steps:
        print(f"\n=== Running {name} ===")
        func()
    print("\nExp 0 complete.")


if __name__ == "__main__":
    main()

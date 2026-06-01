"""Run Exp 0.1 reference alignment and hardening pipeline."""

from __future__ import annotations

from thesis_exp.src.edujudge.data import audit_official_sources, build_dataset, inventory_sources, leakage_check, make_splits, profile_schema, sanity_check_exp00_reference
from thesis_exp.src.edujudge.plots import plot_distributions
from thesis_exp.src.edujudge.utils.io import ensure_exp_dirs


def main() -> None:
    ensure_exp_dirs()
    steps = [
        ("inventory_sources", inventory_sources.main),
        ("profile_schema", profile_schema.main),
        ("audit_official_sources", audit_official_sources.main),
        ("build_dataset", build_dataset.main),
        ("make_splits", make_splits.main),
        ("leakage_check", leakage_check.main),
        ("sanity_check_exp00_reference", sanity_check_exp00_reference.main),
        ("plot_distributions", plot_distributions.main),
    ]
    for name, func in steps:
        print(f"\n=== Running {name} ===")
        if name == "audit_official_sources":
            func([])
        else:
            func()
    print("\nExp 0.1 reference alignment complete.")


if __name__ == "__main__":
    main()

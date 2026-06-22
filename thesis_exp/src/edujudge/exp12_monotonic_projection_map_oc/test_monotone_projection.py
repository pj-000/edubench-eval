"""Sanity checks for Exp12 monotonic projection."""

from __future__ import annotations

import numpy as np

from thesis_exp.src.edujudge.exp12_monotonic_projection_map_oc.monotone_projection import (
    project_nonincreasing_probs,
)


def _assert_nonincreasing(values: np.ndarray) -> None:
    if not bool(np.all(values[..., :-1] >= values[..., 1:] - 1e-12)):
        raise AssertionError(f"Projection is not non-increasing: {values}")
    if not bool(np.all((values >= -1e-12) & (values <= 1.0 + 1e-12))):
        raise AssertionError(f"Projection escaped [0,1]: {values}")


def main() -> None:
    keep = np.array([0.9, 0.8, 0.7, 0.2])
    projected_keep = project_nonincreasing_probs(keep)
    if not np.allclose(projected_keep, keep):
        raise AssertionError(f"Already monotone case changed unexpectedly: {projected_keep}")
    for case in [
        np.array([0.1, 0.2, 0.3, 0.4]),
        np.array([0.2, 0.8, 0.6, 0.4]),
        np.array([0.0, 1.0, 0.0, 1.0]),
        np.array([1.0, 0.0, 1.0, 0.0]),
    ]:
        projected = project_nonincreasing_probs(case)
        _assert_nonincreasing(projected)
        if np.isnan(projected).any():
            raise AssertionError(f"Projection produced NaN for {case}: {projected}")
    batch = np.stack([keep, [0.1, 0.2, 0.3, 0.4], [0.2, 0.8, 0.6, 0.4]], axis=0)
    projected_batch = project_nonincreasing_probs(batch)
    if projected_batch.shape != batch.shape:
        raise AssertionError(f"Batch shape changed: {projected_batch.shape} != {batch.shape}")
    _assert_nonincreasing(projected_batch)
    print("Exp12 monotone projection sanity PASS")


if __name__ == "__main__":
    main()

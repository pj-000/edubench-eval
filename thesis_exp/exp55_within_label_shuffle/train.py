"""Run the shuffled-soft control through the locked Exp51 trainer."""

from __future__ import annotations

import importlib
import json

from thesis_exp.exp55_within_label_shuffle import VARIANT
from thesis_exp.exp55_within_label_shuffle.build_targets import load_split


def main() -> None:
    exp51_train = importlib.import_module("thesis_exp.exp51_hmsa.train")
    # Patch only the training/dev row provider and reader-facing variant name.
    # The model, optimizer, checkpoint rule, hard-head inference, and all other
    # behavior remain the locked Exp51 implementation.
    exp51_train.load_split = load_split
    exp51_train.VARIANT = VARIANT
    config = exp51_train.parse_args()
    result = exp51_train.train(config)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()

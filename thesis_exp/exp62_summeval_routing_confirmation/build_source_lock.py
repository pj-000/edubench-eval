from __future__ import annotations

import argparse
import json
from pathlib import Path

from thesis_exp.exp62_summeval_routing_confirmation.contract import (
    SOURCE_LOCK_PATH,
    build_source_lock,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=Path, required=True)
    args = parser.parse_args()
    lock = build_source_lock(args.model_path)
    SOURCE_LOCK_PATH.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": lock["status"], "path": str(SOURCE_LOCK_PATH)}, indent=2))


if __name__ == "__main__":
    main()


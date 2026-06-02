#!/usr/bin/env bash
set -euo pipefail

python -m thesis_exp.src.edujudge.exp02.build_exp02_dataset --print-summary
python -m thesis_exp.src.edujudge.exp02.sanity_check_exp02_train_setup

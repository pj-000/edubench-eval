# Thesis Exp 0: EduBench Data Audit

This directory contains the reproducible Exp 0 pipeline for EduBench data
verification, normalized human-scored dataset construction, split creation,
leakage checks, and thesis-ready distribution figures.

The experiment is data-only. It does not train models, call APIs, or use GPU.
Existing repository data files are treated as read-only inputs. Generated code,
processed data, tables, reports, and figures are written under `thesis_exp/`.

## Run All

```bash
python -m thesis_exp.src.edujudge.data.run_exp00
```

Exp2 training setup lives in [`README_exp02.md`](README_exp02.md). It prepares the locked
paper-like split for a 0.6B 5-class CE baseline and provides the GPU training command template.

## Run Step By Step

```bash
python -m thesis_exp.src.edujudge.data.inventory_sources
python -m thesis_exp.src.edujudge.data.profile_schema
python -m thesis_exp.src.edujudge.data.build_dataset
python -m thesis_exp.src.edujudge.data.make_splits
python -m thesis_exp.src.edujudge.data.leakage_check
python -m thesis_exp.src.edujudge.plots.plot_distributions
```

## Main Outputs

- `thesis_exp/data/processed/edubench_scoring_all.jsonl`
- `thesis_exp/data/splits/paper_like_triple_seed42/`
- `thesis_exp/data/splits/question_seed42/`
- `thesis_exp/outputs/exp00_data/data_card.md`
- `thesis_exp/outputs/exp00_data/leakage_report.md`
- `thesis_exp/outputs/exp00_data/report.md`
- `thesis_exp/outputs/exp00_data/review_package.md`
- `thesis_exp/outputs/exp00_data/figures/`
- `thesis_exp/outputs/exp00_data/tables/`

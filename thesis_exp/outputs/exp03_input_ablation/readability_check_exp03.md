# Exp3 Readability Check

Overall status: **PASS**

| check | path | status | observed | expected | notes |
| --- | --- | --- | --- | --- | --- |
| script line count | thesis_exp/scripts/run_exp03_smoke.sh | PASS | 64 | >20 |  |
| script line count LF line endings | thesis_exp/scripts/run_exp03_smoke.sh | PASS | LF | LF |  |
| script line count | thesis_exp/scripts/run_exp03_train_a3_a4.sh | PASS | 190 | >20 |  |
| script line count LF line endings | thesis_exp/scripts/run_exp03_train_a3_a4.sh | PASS | LF | LF |  |
| bash -n | thesis_exp/scripts/run_exp03_smoke.sh | PASS | ok | ok |  |
| bash -n | thesis_exp/scripts/run_exp03_train_a3_a4.sh | PASS | ok | ok |  |
| python module line count | thesis_exp/src/edujudge/exp03/__init__.py | PASS | 86 | >10 |  |
| python module line count LF line endings | thesis_exp/src/edujudge/exp03/__init__.py | PASS | LF | LF |  |
| python module line count | thesis_exp/src/edujudge/exp03/build_exp03_datasets.py | PASS | 401 | >10 |  |
| python module line count LF line endings | thesis_exp/src/edujudge/exp03/build_exp03_datasets.py | PASS | LF | LF |  |
| python module line count | thesis_exp/src/edujudge/exp03/collect_exp03_results.py | PASS | 211 | >10 |  |
| python module line count LF line endings | thesis_exp/src/edujudge/exp03/collect_exp03_results.py | PASS | LF | LF |  |
| python module line count | thesis_exp/src/edujudge/exp03/compute_input_ablation_metrics.py | PASS | 118 | >10 |  |
| python module line count LF line endings | thesis_exp/src/edujudge/exp03/compute_input_ablation_metrics.py | PASS | LF | LF |  |
| python module line count | thesis_exp/src/edujudge/exp03/plot_input_ablation_figures.py | PASS | 264 | >10 |  |
| python module line count LF line endings | thesis_exp/src/edujudge/exp03/plot_input_ablation_figures.py | PASS | LF | LF |  |
| python module line count | thesis_exp/src/edujudge/exp03/postprocess_exp03_results.py | PASS | 40 | >10 |  |
| python module line count LF line endings | thesis_exp/src/edujudge/exp03/postprocess_exp03_results.py | PASS | LF | LF |  |
| python module line count | thesis_exp/src/edujudge/exp03/readability_check_exp03.py | PASS | 158 | >10 |  |
| python module line count LF line endings | thesis_exp/src/edujudge/exp03/readability_check_exp03.py | PASS | LF | LF |  |
| python module line count | thesis_exp/src/edujudge/exp03/rubric_quality_audit.py | PASS | 193 | >10 |  |
| python module line count LF line endings | thesis_exp/src/edujudge/exp03/rubric_quality_audit.py | PASS | LF | LF |  |
| python module line count | thesis_exp/src/edujudge/exp03/rubric_repair.py | PASS | 576 | >10 |  |
| python module line count LF line endings | thesis_exp/src/edujudge/exp03/rubric_repair.py | PASS | LF | LF |  |
| python module line count | thesis_exp/src/edujudge/exp03/rubric_sources.py | PASS | 138 | >10 |  |
| python module line count LF line endings | thesis_exp/src/edujudge/exp03/rubric_sources.py | PASS | LF | LF |  |
| python module line count | thesis_exp/src/edujudge/exp03/run_exp03.py | PASS | 202 | >10 |  |
| python module line count LF line endings | thesis_exp/src/edujudge/exp03/run_exp03.py | PASS | LF | LF |  |
| python module line count | thesis_exp/src/edujudge/exp03/sanity_check_exp03_outputs.py | PASS | 100 | >10 |  |
| python module line count LF line endings | thesis_exp/src/edujudge/exp03/sanity_check_exp03_outputs.py | PASS | LF | LF |  |
| python module line count | thesis_exp/src/edujudge/exp03/sanity_check_exp03_setup.py | PASS | 293 | >10 |  |
| python module line count LF line endings | thesis_exp/src/edujudge/exp03/sanity_check_exp03_setup.py | PASS | LF | LF |  |
| python module line count | thesis_exp/src/edujudge/exp03/templates.py | PASS | 213 | >10 |  |
| python module line count LF line endings | thesis_exp/src/edujudge/exp03/templates.py | PASS | LF | LF |  |
| python module line count | thesis_exp/src/edujudge/exp03/train_input_ablation.py | PASS | 264 | >10 |  |
| python module line count LF line endings | thesis_exp/src/edujudge/exp03/train_input_ablation.py | PASS | LF | LF |  |
| python module line count | thesis_exp/src/edujudge/exp03/write_exp03_report.py | PASS | 360 | >10 |  |
| python module line count LF line endings | thesis_exp/src/edujudge/exp03/write_exp03_report.py | PASS | LF | LF |  |
| py_compile exp03 modules | thesis_exp/src/edujudge/exp03 | PASS | ok | ok |  |
| pandas.read_csv | thesis_exp/outputs/exp03_input_ablation/runs/A2_question_answer_metric/logs/training_lo... | PASS | rows=2 cols=32 | readable |  |
| pandas.read_csv | thesis_exp/outputs/exp03_input_ablation/runs/A2_question_answer_metric/tables/high_scor... | PASS | rows=2 cols=5 | readable |  |
| pandas.read_csv | thesis_exp/outputs/exp03_input_ablation/runs/A2_question_answer_metric/tables/low_score... | PASS | rows=2 cols=10 | readable |  |
| pandas.read_csv | thesis_exp/outputs/exp03_input_ablation/runs/A2_question_answer_metric/tables/metric_le... | PASS | rows=24 cols=29 | readable |  |
| pandas.read_csv | thesis_exp/outputs/exp03_input_ablation/runs/A2_question_answer_metric/tables/metrics_s... | PASS | rows=2 cols=32 | readable |  |
| pandas.read_csv | thesis_exp/outputs/exp03_input_ablation/runs/A2_question_answer_metric/tables/per_bin_m... | PASS | rows=10 cols=10 | readable |  |
| pandas.read_csv | thesis_exp/outputs/exp03_input_ablation/runs/A2_question_answer_metric/tables/scenario_... | PASS | rows=17 cols=29 | readable |  |
| pandas.read_csv | thesis_exp/outputs/exp03_input_ablation/tables/a2_exp2_template_equivalence.csv | PASS | rows=3 cols=5 | readable |  |
| pandas.read_csv | thesis_exp/outputs/exp03_input_ablation/tables/corrected_rubric_mapping.csv | PASS | rows=1 cols=11 | readable |  |
| pandas.read_csv | thesis_exp/outputs/exp03_input_ablation/tables/dataset_stats_by_template.csv | PASS | rows=15 cols=9 | readable |  |
| pandas.read_csv | thesis_exp/outputs/exp03_input_ablation/tables/input_ablation_delta_vs_a2.csv | PASS | rows=5 cols=13 | readable |  |
| pandas.read_csv | thesis_exp/outputs/exp03_input_ablation/tables/input_ablation_low_score.csv | PASS | rows=2 cols=13 | readable |  |
| pandas.read_csv | thesis_exp/outputs/exp03_input_ablation/tables/input_ablation_low_score_comparison.csv | PASS | rows=5 cols=7 | readable |  |
| pandas.read_csv | thesis_exp/outputs/exp03_input_ablation/tables/input_ablation_metric_delta_vs_a2.csv | PASS | rows=12 cols=6 | readable |  |
| pandas.read_csv | thesis_exp/outputs/exp03_input_ablation/tables/input_ablation_metric_level.csv | PASS | rows=24 cols=32 | readable |  |
| pandas.read_csv | thesis_exp/outputs/exp03_input_ablation/tables/input_ablation_per_bin.csv | PASS | rows=10 cols=13 | readable |  |
| pandas.read_csv | thesis_exp/outputs/exp03_input_ablation/tables/input_ablation_scenario_level.csv | PASS | rows=17 cols=32 | readable |  |
| pandas.read_csv | thesis_exp/outputs/exp03_input_ablation/tables/input_ablation_summary.csv | PASS | rows=5 cols=21 | readable |  |
| pandas.read_csv | thesis_exp/outputs/exp03_input_ablation/tables/input_ablation_token_length.csv | PASS | rows=15 cols=13 | readable |  |
| pandas.read_csv | thesis_exp/outputs/exp03_input_ablation/tables/label_distribution_by_template.csv | PASS | rows=75 cols=5 | readable |  |
| pandas.read_csv | thesis_exp/outputs/exp03_input_ablation/tables/readability_check_exp03.csv | PASS | rows=100 cols=6 | readable |  |
| pandas.read_csv | thesis_exp/outputs/exp03_input_ablation/tables/rubric_quality_audit.csv | PASS | rows=132 cols=8 | readable |  |
| pandas.read_csv | thesis_exp/outputs/exp03_input_ablation/tables/rubric_repair_candidates.csv | PASS | rows=15 cols=8 | readable |  |
| pandas.read_csv | thesis_exp/outputs/exp03_input_ablation/tables/rubric_source_audit.csv | PASS | rows=24 cols=10 | readable |  |
| pandas.read_csv | thesis_exp/outputs/exp03_input_ablation/tables/sanity_check_exp03_outputs.csv | PASS | rows=14 cols=5 | readable |  |
| pandas.read_csv | thesis_exp/outputs/exp03_input_ablation/tables/sanity_check_exp03_setup.csv | PASS | rows=115 cols=5 | readable |  |
| pandas.read_csv | thesis_exp/outputs/exp03_input_ablation/tables/template_length_stats.csv | PASS | rows=15 cols=13 | readable |  |
| pandas.read_csv | thesis_exp/outputs/exp03_input_ablation/templates/template_lengths.csv | PASS | rows=27680 cols=8 | readable |  |
| pandas.read_csv | thesis_exp/outputs/exp03_input_ablation/templates/template_manifest.csv | PASS | rows=5 cols=7 | readable |  |
| jsonl line json.loads | thesis_exp/outputs/exp03_input_ablation/datasets/A0_answer_only/dev.jsonl | PASS | rows=664 | all nonempty lines parse |  |
| jsonl line json.loads | thesis_exp/outputs/exp03_input_ablation/datasets/A0_answer_only/test.jsonl | PASS | rows=2218 | all nonempty lines parse |  |
| jsonl line json.loads | thesis_exp/outputs/exp03_input_ablation/datasets/A0_answer_only/train.jsonl | PASS | rows=2654 | all nonempty lines parse |  |
| jsonl line json.loads | thesis_exp/outputs/exp03_input_ablation/datasets/A1_question_answer/dev.jsonl | PASS | rows=664 | all nonempty lines parse |  |
| jsonl line json.loads | thesis_exp/outputs/exp03_input_ablation/datasets/A1_question_answer/test.jsonl | PASS | rows=2218 | all nonempty lines parse |  |
| jsonl line json.loads | thesis_exp/outputs/exp03_input_ablation/datasets/A1_question_answer/train.jsonl | PASS | rows=2654 | all nonempty lines parse |  |
| jsonl line json.loads | thesis_exp/outputs/exp03_input_ablation/datasets/A2_question_answer_metric/dev.jsonl | PASS | rows=664 | all nonempty lines parse |  |
| jsonl line json.loads | thesis_exp/outputs/exp03_input_ablation/datasets/A2_question_answer_metric/test.jsonl | PASS | rows=2218 | all nonempty lines parse |  |
| jsonl line json.loads | thesis_exp/outputs/exp03_input_ablation/datasets/A2_question_answer_metric/train.jsonl | PASS | rows=2654 | all nonempty lines parse |  |
| jsonl line json.loads | thesis_exp/outputs/exp03_input_ablation/datasets/A3_question_answer_metric_rubric/dev.j... | PASS | rows=664 | all nonempty lines parse |  |
| jsonl line json.loads | thesis_exp/outputs/exp03_input_ablation/datasets/A3_question_answer_metric_rubric/test.... | PASS | rows=2218 | all nonempty lines parse |  |
| jsonl line json.loads | thesis_exp/outputs/exp03_input_ablation/datasets/A3_question_answer_metric_rubric/train... | PASS | rows=2654 | all nonempty lines parse |  |
| jsonl line json.loads | thesis_exp/outputs/exp03_input_ablation/datasets/A4_question_answer_metric_rubric_metad... | PASS | rows=664 | all nonempty lines parse |  |
| jsonl line json.loads | thesis_exp/outputs/exp03_input_ablation/datasets/A4_question_answer_metric_rubric_metad... | PASS | rows=2218 | all nonempty lines parse |  |
| jsonl line json.loads | thesis_exp/outputs/exp03_input_ablation/datasets/A4_question_answer_metric_rubric_metad... | PASS | rows=2654 | all nonempty lines parse |  |
| jsonl line json.loads | thesis_exp/outputs/exp03_input_ablation/predictions/A2_question_answer_metric_predictio... | PASS | rows=664 | all nonempty lines parse |  |
| jsonl line json.loads | thesis_exp/outputs/exp03_input_ablation/predictions/A2_question_answer_metric_predictio... | PASS | rows=2218 | all nonempty lines parse |  |
| jsonl line json.loads | thesis_exp/outputs/exp03_input_ablation/runs/A2_question_answer_metric/predictions/pred... | PASS | rows=664 | all nonempty lines parse |  |
| jsonl line json.loads | thesis_exp/outputs/exp03_input_ablation/runs/A2_question_answer_metric/predictions/pred... | PASS | rows=2218 | all nonempty lines parse |  |
| markdown max line length | thesis_exp/outputs/exp03_input_ablation/datasets/dataset_card.md | PASS | 90 | <300 |  |
| markdown max line length | thesis_exp/outputs/exp03_input_ablation/figures/figure_manifest.md | PASS | 55 | <300 |  |
| markdown max line length | thesis_exp/outputs/exp03_input_ablation/notion_exp03_paper_notes.md | PASS | 77 | <300 |  |
| markdown max line length | thesis_exp/outputs/exp03_input_ablation/notion_exp03_summary.md | PASS | 160 | <300 |  |
| markdown max line length | thesis_exp/outputs/exp03_input_ablation/readability_check_exp03.md | PASS | 167 | <300 |  |
| markdown max line length | thesis_exp/outputs/exp03_input_ablation/report.md | PASS | 160 | <300 |  |
| markdown max line length | thesis_exp/outputs/exp03_input_ablation/reports/rubric_quality_audit.md | PASS | 87 | <300 |  |
| markdown max line length | thesis_exp/outputs/exp03_input_ablation/reports/rubric_repair_source_trace.md | PASS | 230 | <300 |  |
| markdown max line length | thesis_exp/outputs/exp03_input_ablation/reports/rubric_source_audit.md | PASS | 93 | <300 |  |
| markdown max line length | thesis_exp/outputs/exp03_input_ablation/review_package.md | PASS | 163 | <300 |  |
| markdown max line length | thesis_exp/outputs/exp03_input_ablation/runs/A2_question_answer_metric/run_summary.md | PASS | 63 | <300 |  |
| markdown max line length | thesis_exp/outputs/exp03_input_ablation/sanity_check_exp03_outputs.md | PASS | 296 | <300 |  |
| markdown max line length | thesis_exp/outputs/exp03_input_ablation/sanity_check_exp03_setup.md | PASS | 198 | <300 |  |

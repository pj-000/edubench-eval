# Exp2 Postprocess Check

Overall status: **PASS**

| artifact | path | status | detail |
| --- | --- | --- | --- |
| metrics_summary.csv | thesis_exp/outputs/exp02_ce_baseline/tables/metrics_summary.csv | PASS | rows=2 columns=32 |
| per_bin_metrics.csv | thesis_exp/outputs/exp02_ce_baseline/tables/per_bin_metrics.csv | PASS | rows=10 columns=10 |
| low_score_metrics.csv | thesis_exp/outputs/exp02_ce_baseline/tables/low_score_metrics.csv | PASS | rows=2 columns=10 |
| predictions_test.jsonl | thesis_exp/outputs/exp02_ce_baseline/predictions/predictions_test.jsonl | PASS | rows=2218 missing_fields=[] |
| exp02_dev_test_arrays.npz | thesis_exp/outputs/exp02_ce_baseline/arrays/exp02_dev_test_arrays.npz | PASS | missing_keys=[] shapes={'logits_dev': [664, 5], 'logits_test': [2218, 5], 'probs_dev': [664, 5], 'probs_test': [2218, 5], 'labels_dev': [664], 'labels_test': [2218], 'record_ids_dev': [664], 'record_ids_test': [2218]} |

# Exp8 Setup Sanity Check

Status: `PASS`

Training executed: `no`.
API called: `no`.
Synthetic generated: `no`.

| check | status | details |
| --- | --- | --- |
| Exp8 run id locked | PASS | QD-ER1_EduRisk_human_only |
| train jsonl exists | PASS | thesis_exp/outputs/exp06_synthetic_low_score/training_datasets/QD-S0_human_only/train.jsonl |
| train row count = 3326 | PASS | 3326 |
| train rows are human only | PASS |  |
| train label provenance is human_score | PASS |  |
| train labels are 1..5 | PASS |  |
| train A4 text exists | PASS |  |
| train template_name is A4 | PASS |  |
| train has no synthetic rows | PASS |  |
| dev jsonl exists | PASS | thesis_exp/outputs/exp06_synthetic_low_score/training_datasets/QD-S0_human_only/dev.jsonl |
| dev row count = 1107 | PASS | 1107 |
| dev rows are human only | PASS |  |
| dev label provenance is human_score | PASS |  |
| dev labels are 1..5 | PASS |  |
| dev A4 text exists | PASS |  |
| dev template_name is A4 | PASS |  |
| dev has no synthetic rows | PASS |  |
| test jsonl exists | PASS | thesis_exp/outputs/exp06_synthetic_low_score/training_datasets/QD-S0_human_only/test.jsonl |
| test row count = 1103 | PASS | 1103 |
| test rows are human only | PASS |  |
| test label provenance is human_score | PASS |  |
| test labels are 1..5 | PASS |  |
| test A4 text exists | PASS |  |
| test template_name is A4 | PASS |  |
| test has no synthetic rows | PASS |  |
| train label 1 count = 58 | PASS | {1: 58, 2: 53, 3: 297, 4: 1163, 5: 1755} |
| train label 2 count = 53 | PASS | {1: 58, 2: 53, 3: 297, 4: 1163, 5: 1755} |
| train label 3 count = 297 | PASS | {1: 58, 2: 53, 3: 297, 4: 1163, 5: 1755} |
| train label 4 count = 1163 | PASS | {1: 58, 2: 53, 3: 297, 4: 1163, 5: 1755} |
| train label 5 count = 1755 | PASS | {1: 58, 2: 53, 3: 297, 4: 1163, 5: 1755} |
| QD-B0 baseline run available | PASS | QD-B0_human_only_ordinary_ordinal |
| QD-B1 baseline run available | PASS | QD-B1_human_only_L1_weighted_ordinal |
| QD-R1 baseline run available | PASS | QD-R1_CORAL_human_only |
| no checkpoint/weights tracked | PASS |  |
| no tracked Exp0-Exp7 output modifications | PASS |  |
| config exists thesis_exp/configs/exp08_edurisk/exp08_qder1_edurisk_human_only.yaml | PASS |  |
| config declares run id exp08_qder1_edurisk_human_only.yaml | PASS |  |
| config disables synthetic exp08_qder1_edurisk_human_only.yaml | PASS |  |
| config exists thesis_exp/configs/exp08_edurisk/exp08_qder1_edurisk_smoke.yaml | PASS |  |
| config declares run id exp08_qder1_edurisk_smoke.yaml | PASS |  |
| config disables synthetic exp08_qder1_edurisk_smoke.yaml | PASS |  |
| formal beta is 0.99 | PASS |  |
| formal normalized cost enabled | PASS |  |
| formal decode primary cumulative | PASS |  |
| formal decode secondary argmax_q | PASS |  |
| py_compile thesis_exp/src/edujudge/exp08_edurisk/__init__.py | PASS |  |
| py_compile thesis_exp/src/edujudge/exp08_edurisk/collect_exp08_results.py | PASS |  |
| py_compile thesis_exp/src/edujudge/exp08_edurisk/coral_distribution.py | PASS |  |
| py_compile thesis_exp/src/edujudge/exp08_edurisk/data.py | PASS |  |
| py_compile thesis_exp/src/edujudge/exp08_edurisk/losses.py | PASS |  |
| py_compile thesis_exp/src/edujudge/exp08_edurisk/metrics.py | PASS |  |
| py_compile thesis_exp/src/edujudge/exp08_edurisk/readability_check_exp08.py | PASS |  |
| py_compile thesis_exp/src/edujudge/exp08_edurisk/sanity_check_exp08_outputs.py | PASS |  |
| py_compile thesis_exp/src/edujudge/exp08_edurisk/sanity_check_exp08_setup.py | PASS |  |
| py_compile thesis_exp/src/edujudge/exp08_edurisk/train_qder1_edurisk.py | PASS |  |
| py_compile thesis_exp/src/edujudge/exp08_edurisk/write_exp08_report.py | PASS |  |
| script exists thesis_exp/scripts/run_exp08_qder1_smoke.sh | PASS |  |
| bash -n thesis_exp/scripts/run_exp08_qder1_smoke.sh | PASS |  |
| script exists thesis_exp/scripts/run_exp08_qder1_train.sh | PASS |  |
| bash -n thesis_exp/scripts/run_exp08_qder1_train.sh | PASS |  |
| script exists thesis_exp/scripts/sync_exp08_qder1_to_server.sh | PASS |  |
| bash -n thesis_exp/scripts/sync_exp08_qder1_to_server.sh | PASS |  |
| class weights have five rows | PASS | 5 |
| class weights beta is 0.99 | PASS | 0.99 |
| class weights no clipping | PASS | [1.4626774356901158, 1.5645803874680504, 0.6805084387006645, 0.646119572287026, 0.646114165854144] |
| class weights mean approximately one | PASS | [1.4626774356901158, 1.5645803874680504, 0.6805084387006645, 0.646119572287026, 0.646114165854144] |
| toy CORAL logits shape | PASS | (3, 4) |
| toy q_raw shape | PASS | (3, 5) |
| toy q_safe shape | PASS | (3, 5) |
| toy q_raw sane before guard | PASS | {'ok': True, 'finite': True, 'min_q_raw': 0.01798620996209155, 'mean_q_raw_sum': 1.0, 'max_q_raw_sum_error': 0.0} |
| toy q_safe sums to one | PASS | [1.0, 1.0, 1.0] |
| soft target shape | PASS | (3, 5) |
| soft target rows sum to one | PASS | [0.9999999999999999, 0.9999999999999998, 1.0] |
| cumulative target shape | PASS | (3, 4) |
| normalized risk cost shape | PASS | (3, 5) |
| normalized risk cost max bounded | PASS | 3.0 |
| cumulative predictions finite | PASS | [4, 3, 5] |
| argmax predictions finite | PASS | [4, 3, 5] |
| expected scores finite | PASS | [3.7247719573721656, 3.101524766329384, 4.546443573468229] |
| toy EduRisk loss finite | PASS | 3.36318336852975 |
| toy debug has L_total | PASS | 3.36318336852975 |
| toy debug has L_softCE | PASS | 1.8037009395626236 |
| toy debug has L_risk | PASS | 0.7740273683215708 |
| toy debug has L_cumBCE | PASS | 0.7493594764316956 |
| toy debug has mean_weight | PASS | 1.2233333333333334 |
| toy debug has min_weight | PASS | 0.65 |
| toy debug has max_weight | PASS | 1.56 |
| toy debug has weighted_L_softCE | PASS | 2.491462259640351 |
| toy debug has weighted_L_risk | PASS | 0.3333702570001556 |
| toy debug has weighted_L_cumBCE | PASS | 0.5383508518892431 |
| toy debug has mean_q_raw_min | PASS | 0.05505419075311446 |
| toy debug has mean_q_raw_sum | PASS | 1.0 |
| toy debug has max_q_raw_sum_error | PASS | 0.0 |
| no checkpoint/weights tracked | PASS |  |
| no tracked Exp0-Exp7 output modifications | PASS |  |

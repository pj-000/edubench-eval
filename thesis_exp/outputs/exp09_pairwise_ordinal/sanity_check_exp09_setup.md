# Exp9 Setup Sanity Check

Status: `PASS`

Training executed: `no`.
API called: `no`.
Synthetic generated: `no`.

## Pair Summary

- Train pairs: `20000`
- Dev diagnostic pairs: `5000`
- Train low_high pairs: `8000`
- Toy L_total: `0.746437350922`
- Toy L_pair: `1.228289530371`

## Checks

| check | status | details |
| --- | --- | --- |
| train jsonl exists | PASS | thesis_exp/outputs/exp06_synthetic_low_score/training_datasets/QD-S0_human_only/train.jsonl |
| train row count = 3326 | PASS | 3326 |
| train rows are human only | PASS |  |
| train labels are 1..5 | PASS |  |
| train A4 text exists | PASS |  |
| train has no synthetic rows | PASS |  |
| dev jsonl exists | PASS | thesis_exp/outputs/exp06_synthetic_low_score/training_datasets/QD-S0_human_only/dev.jsonl |
| dev row count = 1107 | PASS | 1107 |
| dev rows are human only | PASS |  |
| dev labels are 1..5 | PASS |  |
| dev A4 text exists | PASS |  |
| dev has no synthetic rows | PASS |  |
| test jsonl exists | PASS | thesis_exp/outputs/exp06_synthetic_low_score/training_datasets/QD-S0_human_only/test.jsonl |
| test row count = 1103 | PASS | 1103 |
| test rows are human only | PASS |  |
| test labels are 1..5 | PASS |  |
| test A4 text exists | PASS |  |
| test has no synthetic rows | PASS |  |
| no checkpoint/weights tracked | PASS |  |
| no tracked Exp0-Exp8 output modifications | PASS |  |
| config exists thesis_exp/configs/exp09_pairwise_ordinal/exp09_qdpr1_pairwise_human_only.yaml | PASS |  |
| config declares run id exp09_qdpr1_pairwise_human_only.yaml | PASS |  |
| config disables synthetic exp09_qdpr1_pairwise_human_only.yaml | PASS |  |
| config disables coral exp09_qdpr1_pairwise_human_only.yaml | PASS |  |
| config disables edurisk exp09_qdpr1_pairwise_human_only.yaml | PASS |  |
| config exists thesis_exp/configs/exp09_pairwise_ordinal/exp09_qdpr1_pairwise_smoke.yaml | PASS |  |
| config declares run id exp09_qdpr1_pairwise_smoke.yaml | PASS |  |
| config disables synthetic exp09_qdpr1_pairwise_smoke.yaml | PASS |  |
| config disables coral exp09_qdpr1_pairwise_smoke.yaml | PASS |  |
| config disables edurisk exp09_qdpr1_pairwise_smoke.yaml | PASS |  |
| formal train pair count is 20000 | PASS |  |
| formal dev pair count is 5000 | PASS |  |
| formal lambda_pair is 0.3 | PASS |  |
| py_compile thesis_exp/src/edujudge/exp09_pairwise_ordinal/__init__.py | PASS |  |
| py_compile thesis_exp/src/edujudge/exp09_pairwise_ordinal/collect_exp09_results.py | PASS |  |
| py_compile thesis_exp/src/edujudge/exp09_pairwise_ordinal/data.py | PASS |  |
| py_compile thesis_exp/src/edujudge/exp09_pairwise_ordinal/losses.py | PASS |  |
| py_compile thesis_exp/src/edujudge/exp09_pairwise_ordinal/metrics.py | PASS |  |
| py_compile thesis_exp/src/edujudge/exp09_pairwise_ordinal/pair_builder.py | PASS |  |
| py_compile thesis_exp/src/edujudge/exp09_pairwise_ordinal/readability_check_exp09.py | PASS |  |
| py_compile thesis_exp/src/edujudge/exp09_pairwise_ordinal/sanity_check_exp09_setup.py | PASS |  |
| py_compile thesis_exp/src/edujudge/exp09_pairwise_ordinal/train_qdpr1_pairwise.py | PASS |  |
| script exists thesis_exp/scripts/run_exp09_qdpr1_smoke.sh | PASS |  |
| bash -n thesis_exp/scripts/run_exp09_qdpr1_smoke.sh | PASS |  |
| script exists thesis_exp/scripts/run_exp09_qdpr1_train.sh | PASS |  |
| bash -n thesis_exp/scripts/run_exp09_qdpr1_train.sh | PASS |  |
| script exists thesis_exp/scripts/sync_exp09_qdpr1_to_server.sh | PASS |  |
| bash -n thesis_exp/scripts/sync_exp09_qdpr1_to_server.sh | PASS |  |
| train pair count = 20000 | PASS | 20000 |
| dev diagnostic pair count = 5000 | PASS | 5000 |
| train pair win_label > lose_label | PASS |  |
| train pair margins positive | PASS |  |
| train pair weights finite | PASS |  |
| train low_high pair count > 0 | PASS |  |
| train low_high proportion close or reported | PASS | rate=0.4000; target=0.4000 |
| train low_mid proportion close or reported | PASS | rate=0.2000; target=0.2000 |
| train adjacent proportion close or reported | PASS | rate=0.3000; target=0.3000 |
| train random_ordinal proportion close or reported | PASS | rate=0.1000; target=0.1000 |
| dev pair win_label > lose_label | PASS |  |
| dev pair margins positive | PASS |  |
| dev pair weights finite | PASS |  |
| dev low_high pair count > 0 | PASS |  |
| dev low_high proportion close or reported | PASS | rate=0.4000; target=0.4000 |
| dev low_mid proportion close or reported | PASS | rate=0.2000; target=0.2000 |
| dev adjacent proportion close or reported | PASS | rate=0.3000; target=0.3000 |
| dev random_ordinal proportion close or reported | PASS | rate=0.1000; target=0.1000 |
| no dev/test records in train pairs | PASS | 0 |
| train max reuse within cap | PASS | {'split': 'train', 'records_used': 3275, 'max_reuse': 208, 'mean_reuse': 12.213740458015268, 'low_records_used': 111, 'max_reuse_low_record': 208, 'mean_reuse_low_record': 108.14414414414415} |
| train low-record max reuse within cap | PASS | {'split': 'train', 'records_used': 3275, 'max_reuse': 208, 'mean_reuse': 12.213740458015268, 'low_records_used': 111, 'max_reuse_low_record': 208, 'mean_reuse_low_record': 108.14414414414415} |
| dev max reuse within cap | PASS | {'split': 'dev', 'records_used': 1068, 'max_reuse': 96, 'mean_reuse': 9.363295880149813, 'low_records_used': 57, 'max_reuse_low_record': 96, 'mean_reuse_low_record': 53.01754385964912} |
| pointwise class weights have six-index vector | PASS | [0.0, 3.0, 3.0, 2.23973063973064, 0.5719690455717971, 0.5] |
| pointwise class weights finite | PASS | [0.0, 3.0, 3.0, 2.23973063973064, 0.5719690455717971, 0.5] |
| pointwise class weights clipped to [0.5, 3.0] | PASS | [0.0, 3.0, 3.0, 2.23973063973064, 0.5719690455717971, 0.5] |
| toy pairwise loss finite | PASS | 0.7464373509224485 |
| toy pair margins positive | PASS | [1.25, 0.75, 0.25] |
| toy pair weights finite | PASS | [2.5, 2.1666666666666665, 1.0] |
| toy scalar scores in [1,5] | PASS | {'win': [3.4739758172912345, 3.2007624659020144, 4.089408970909261], 'lose': [2.161498894179186, 2.5817367171531047, 3.069500112055714]} |
| toy debug has L_total | PASS | 0.746437350922 |
| toy debug has L_point | PASS | 0.377950491811 |
| toy debug has L_pair | PASS | 1.228289530371 |
| toy debug has weighted_L_pair | PASS | 1.228289530371 |
| toy debug has mean_pair_weight | PASS | 1.888888888889 |
| toy debug has mean_pair_margin | PASS | 0.75 |
| toy debug has mean_score_gap | PASS | 0.983803843572 |
| toy debug has low_high_pair_loss | PASS | 1.652170845702 |
| toy debug has adjacent_pair_loss | PASS | 0.380526899709 |
| toy debug has mean_point_base_loss | PASS | 0.404632098502 |
| toy debug has mean_point_sample_weight | PASS | 1.357323015191 |
| pair inventory has train/dev rows | PASS | [{'split': 'train', 'pair_count': '20000', 'target_count': '20000', 'sampling_seed': '42', 'max_pairs_per_record': '80', 'max_pairs_per_low_record': '240'}, {'split': 'dev', 'pair_count': '5000', 'target_count': '5000', 'sampling_seed': '42', 'max_pairs_per_record': '80', 'max_pairs_per_low_record': '240'}] |
| pair comparability audit has train/dev pair-type rows | PASS | [('dev', 'adjacent'), ('dev', 'low_high'), ('dev', 'low_mid'), ('dev', 'random_ordinal'), ('train', 'adjacent'), ('train', 'low_high'), ('train', 'low_mid'), ('train', 'random_ordinal')] |
| no checkpoint/weights tracked | PASS |  |
| no tracked Exp0-Exp8 output modifications | PASS |  |

# Exp33A Independent Model-Reviewed Silver Reference Validation

Overall status: **PASS**.

| check | status | detail |
|---|---|---|
| required_public_files | PASS | missing=none |
| required_private_files | PASS | missing=none |
| private_files_gitignored | PASS | not_ignored=none |
| paper_train_dev_boundary | PASS | {"dev_rows": 664, "dev_unique_question_keys": 184, "future_train_rows_removed_for_question_key_overlap": 0, "future_train_rows_retained": 2654, "processed_rows": 5536, "train_dev_question_key_overlap": 184, "train_dev_sample_overlap": 0, "train_dev_triple_key_overlap": 0, "train_rows": 2654, "train_rows_on_train_dev_shared_question_keys": 2562, "train_unique_question_keys": 196} |
| teacher_input_manifest | PASS | primary=2654:42d4ca48d4ef7ef9bf4bddc91c5652c337bcfba108359527e21354a3247150eb; secondary=1552:07ae301026a6330de682e02fa6f0195eaf71a6f3275f589551e334b48e6c462a |
| deterministic_sampling_and_weights | PASS | failures=none; clean_dev_max_qkeys=175 |
| blind_packets_and_hashes | PASS | failures=none; rows=420 |
| schemas_and_provenance | PASS | draft-2020-12 schemas valid for model/human provenance and staged adjudication |
| paper_protocol_and_pre_review_decision | PASS | failures=none |
| public_distribution_dimensions | PASS | missing_dimensions=none |
| public_private_identity_leakage | PASS | failures=none |
| blind_leakage_zero | PASS | all nine forbidden leakage classes reconstructed as zero |
| old_test_access_count_zero | PASS | sealed test path was never passed to a reader; inherited locked test row count only |
| no_api_gpu_training_inference | PASS | validator is CPU data/schema audit only |
| heavy_private_source_reference_coverage | PASS | rows=420 |
| staged_heavy_commit_boundary | PASS | staged_files=34; failures=none |

The paper protocol remains triple-key-disjoint rather than question-key-disjoint. Question-key overlap is expected and removes zero train rows. The provider-agnostic method claim is blind-first source comparison, conflict adjudication, direction-aware correction, and uncertainty fallback. Actual reviewer provider/model IDs remain mandatory provenance.

No reviewer result was fabricated. `model_silver_reference_complete=false`, `expert_reference_complete=false`, and `teacher_reliability_ready=false` remain locked before review. No API, GPU, training, student inference, or sealed-test access occurred.

# Exp38A Qwen score-range qualification

- Status: **RANGE_PROTOCOL_NOT_QUALIFIED**
- Rows: 196
- Center MAE / QWK: 0.7806 / 0.5729
- Silver point coverage / range overlap: 0.6990 / 0.7959
- Mean / median width: 1.7296 / 2.0000
- Non-singleton rate: 0.6990
- Gate checks: `{"center_mae": false, "center_qwk": true, "human_high_direction": true, "human_low_direction": false, "mean_width": true, "non_singleton_rate": true, "range_overlap": false, "schema_success": true, "silver_high_direction": false, "silver_low_direction": false, "silver_point_coverage": false, "target_scope_success": true}`
- Full-train annotation recommended: `false`
- This frozen 196 set must not be reused to tune the prompt after observing results.
- No reason/failure supervision, student training, dev, or test access occurred.

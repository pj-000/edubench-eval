# Exp3 Rubric Repair Source Trace

Selected rubric mode: **proposed**

## Conclusion

No official/original Chinese per-score Scenario Element Integration rubric was found.
The repair is therefore a proposed Chinese translation from the official English SEI anchors in the
paper/PDF.
Formal A3/A4 training must remain blocked until the proposed rubric is reviewed and confirmed.

## Checked Sources

| source_file | source_field | language | metric | confidence | identical_to_IFTC | notes |
| --- | --- | --- | --- | --- | --- | --- |
| thesis_exp/outputs/exp03_input_ablation/tables/rubric_source_audit.csv | rubric | zh | Instruction Following & Task Completion | low | False | Split-level source. zh SEI is suspect if identical to IFTC. |
| thesis_exp/outputs/exp03_input_ablation/tables/rubric_source_audit.csv | rubric | zh | Scenario Element Integration | low | True | Split-level source. zh SEI is suspect if identical to IFTC. |
| thesis_exp/outputs/exp03_input_ablation/tables/rubric_source_audit.csv | rubric | en | Scenario Element Integration | high | False | Split-level source. zh SEI is suspect if identical to IFTC. |
| thesis_exp/data/processed/edubench_scoring_all.jsonl | rubric | zh | Scenario Element Integration | low | True | Processed data row rubric; it reproduces the defective zh SEI text. |
| edu-data-synthesis-main/data/criteria/metrics_zh_whiten.json | 1.4.rules | zh | Scenario Element Integration | low | True | Local criteria file. zh SEI matches IFTC content, so it is not a reliable correction source. |
| edu-data-synthesis-main/data/criteria/metrics_en_whiten.json | 1.4.rules | en | Scenario Element Integration | high | False | Local English criteria file contains the expected SEI-specific anchors. |
| 5-grades/5_metrics_zh.json | 场景要素融合度.rules | zh | Scenario Element Integration | low | True | Local five-grade criteria file. zh SEI matches IFTC content, so it is not a reliable correction source. |
| 5-grades/5_metrics_en.json | Scenario Element Integration.rules | en | Scenario Element Integration | high | False | Local five-grade English criteria contains the expected SEI-specific anchors. |
| EduBench.pdf | Appendix F.1.4 | en | Scenario Element Integration | high | False | Official paper PDF contains English 10-point SEI scoring anchors, not a Chinese per-score rubric. |
| results_merge.jsonl | all fields | zh | Scenario Element Integration | none | False | Contains metric/sample records but no rubric text field. |
| metrics_map.json | all fields | zh | Scenario Element Integration | none | False | Contains scenario-to-metric mapping only, no scoring anchors. |
| thesis_exp/outputs/exp00_data/tables/metric_mapping.csv | all fields | zh | Scenario Element Integration | none | False | Contains canonical metric mapping only, no rubric text. |
| thesis_exp/configs/reference_contract.yaml | all fields | zh | Scenario Element Integration | none | False | Contains reference names/contracts only, no rubric text. |
| thesis_exp/.cache/official_edubench/README.md and README_zh.md | Evaluation Metrics Design | zh | Scenario Element Integration | none | False | Official GitHub README lists metric names/descriptions but no per-score rubric. |
| EduBench.zip | README, README.md, anthology.bib.txt, distribution-radar.pdf, distribution-radar | zh | Scenario Element Integration | none | False | Archive contains README and figure PDFs only; no criteria/rubric JSON or CSV. |

## zh SEI Finding

- zh SEI candidate rows with rubric text: 4
- zh SEI rows identical to zh IFTC: 4
- Mapping CSV: `thesis_exp/outputs/exp03_input_ablation/tables/proposed_corrected_rubric_mapping.csv`
- Candidate CSV: `thesis_exp/outputs/exp03_input_ablation/tables/rubric_repair_candidates.csv`

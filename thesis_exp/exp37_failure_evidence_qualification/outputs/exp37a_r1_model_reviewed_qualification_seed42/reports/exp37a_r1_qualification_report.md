# Exp37A-R1 qualification report

## Decision
- Status: `NO_GO`
- Reference complete: `True`
- Reference gate: `True`
- Semantic gate: `False`
- Utility gate: `False`
- Score-range pilot gate: `True`

## Reference quality
- Reviewer A/B score QWK: `0.8112382331643737`
- Major-failure-presence agreement: `0.8673469387755102`
- Selective Reviewer C conflicts: `125` of 196

## Qwen semantic qualification
- Major-failure F1: `0.6551724137931034`
- Low-tail major-failure recall: `0.8333333333333334`
- Supported subtype macro-F1: `0.5587499693649977`
- Evidence syntax validity: `0.75`
- Semantic support partial-or-better: `0.6020408163265306`

## OOF utility
- Train-only OOF input available: `True`
- Human-anchor severe-error AUPRC: `0.46105801475519786`
- Human-anchor aligned minus permutation mean: `-0.013482304541534873`
- Human-anchor permutation p: `0.8671328671328671`
- Human-anchor bootstrap lower CI: `-0.08638114796966159`
- Silver-anchor severe-error AUPRC: `0.6173469387755103`
- Human-anchor and model-reviewed silver-anchor targets remain strictly separate.
- Strong silver-anchor utility without human-anchor utility is not treated as benchmark improvement evidence.
- A missing train-only OOF input is reported as unavailable, never fabricated.

## Boundary
- No API, GPU, training, student inference, dev, or test access occurred.

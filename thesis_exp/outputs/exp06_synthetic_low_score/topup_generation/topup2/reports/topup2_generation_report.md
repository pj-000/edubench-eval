# Exp6-11 Topup-2 Synthetic Generation Report

## Status

- Static checks status: **PASS**
- API called: **YES**
- API generation status: **API_CALLED**
- Planned count: **160**
- Generated count: **160**
- Normalized count: **160**
- Filtered pass count: **157**
- Leakage status: **PASS**

## Filtered Distribution

- Label distribution: `{'3': 24, '2': 66, '1': 67}`
- Language distribution: `{'en': 80, 'zh': 77}`
- Metric coverage: **12**
- Error type coverage: **7**
- Error type distribution: `{'reasoning_gap': 20, 'scenario_mismatch': 35, 'factual_error': 21, 'instruction_violation': 12, 'superficial_fluency': 26, 'overconfident_wrong': 13, 'rubric_violation': 30}`
- Manual review required count: **0**

## Source Reuse

- Prior source question reuse max/mean: **2 / 1.1625**
- Topup2 within-batch source question reuse max/mean: **4 / 1.7937**
- Source split: **question_seed42/train only**

## Gates

- Can full 384 generation start: **NO**
- Can final synthetic pool build start: **YES**
- Exp6 training can start: **NO**

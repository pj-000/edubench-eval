# Exp28E Three-Seed Dev-Only Results

- runs: 15/15
- model: Qwen3-Reranker-0.6B
- input: question + answer + evaluation dimension
- loss: ordinary cross-entropy
- epochs: 10
- effective batch size: 128
- checkpoint selection: validation accuracy
- test read: no
- decision: **DEV_SUCCESS_CRITERIA_NOT_MET**

The selective method is compared with the original-label baseline, full primary-teacher labels,
uncertainty filtering, and a matched random-transition control. Test remains closed until the
paired clustered bootstrap and final dev lock pass.

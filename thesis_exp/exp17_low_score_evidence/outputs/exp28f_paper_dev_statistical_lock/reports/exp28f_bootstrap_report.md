# Exp28F Dev Statistical Lock

- two-level bootstrap: seed and paper triple cluster
- resamples: 2000
- main comparison: B2 selective dual teacher vs B0 original labels
- targeting control: B2 vs B4 random transition control
- decision: **BOOTSTRAP_SUCCESS_CRITERIA_NOT_MET**
- test read: no

The held-out test remains closed. A separate lock step verifies this decision, the multiseed dev
decision, dataset hashes, and checkpoint availability before authorizing final evaluation.
